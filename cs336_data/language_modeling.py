from __future__ import annotations

import os
import re
from typing import Any
import numpy as np
from cs336_data.data_quality import *
from cs336_data.deduplication import *
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import learning_rate_schedule
from cs336_basics.data import get_batch
from fastwarc.warc import ArchiveIterator
from pathlib import Path
import timeit

import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
logger = logging.getLogger(__name__)

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

## Process input WET file and keep only English documents with probability over threshold
def english_wet(infile: str, outfile: str, lang_threshold: float = 0.7):
    total_count = 0
    en_count = 0
    with open(infile, "rb") as fin, open(outfile, "w") as fout:
        records = ArchiveIterator(fin)
        for record in records:
            total_count += 1
            text = extract_text_from_html_bytes(record.reader.read())
            text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
            text = re.sub(r"\s+", " ", text).strip()
            lang, lang_conf = identify_language(text)
            if lang == "en" and lang_conf >= lang_threshold and gopher_quality_filter(text):
                en_count += 1
                # label, conf = classify_quality(text)
                fout.write(f"{text}\n")
    # print(f"Total number of records: {total_count}")
    # print(f"Number of English records retained: {en_count}")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    model_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(model_dict, out)


def load_checkpoint(src: str, model: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None):
    checkpoint = torch.load(src)
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def model_train(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    betas: tuple[float, float],
    lr_min: float,
    lr_max: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    weight_decay: float,
    eps: float,
    max_iter: int,
    batch_size: int,
    gradient_accumulation_step: int,
    train_data_filename: str,
    valid_data_filename: str,
    max_grad_norm: float,
    training_dtype: str,
    checkpoint_path: str,
    model_prefix: str,
    continue_from: int = 0
):
    os.makedirs(checkpoint_path, exist_ok=True)
    train_data = np.memmap(train_data_filename, dtype=np.uint16, mode="r")
    valid_data = np.memmap(valid_data_filename, dtype=np.uint16, mode="r")
    model_hyperparams = {
        "vocab_size": vocab_size, 
        "context_length": context_length, 
        "d_model": d_model, 
        "num_layers": num_layers, 
        "num_heads": num_heads, 
        "d_ff": d_ff, 
        "rope_theta": rope_theta
    }
    model = BasicsTransformerLM(**model_hyperparams).to(device)
    # Set up the AdamW optimizer.
    # First, we need to group the parameters that should
    # be decayed and those that shouldn't.
    # In particular, we do not apply decay on 1D parameters (e.g., biases and RMSNorms)
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr = lr_max,
        betas = betas,
        eps = eps,
        fused = True,
    )
    if continue_from > 0:
        if Path(os.path.join(checkpoint_path, f"./{model_prefix}_{continue_from}.pt")).is_file():
            load_checkpoint(os.path.join(checkpoint_path, f"./{model_prefix}_{continue_from}.pt"), model, optimizer)
        else:
            raise FileNotFoundError(f"Checkpoint file {os.path.join(checkpoint_path, f'./{model_prefix}_{continue_from}.pt')} not found.")
    start_step = continue_from if continue_from > 0 else 0


    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[training_dtype]

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
    X_train, y_train = get_batch(
        train_data,
        batch_size = batch_size,
        context_length = context_length,
        device = str(device)
    )

    ## Model training
    train_step = start_step
    while train_step <= max_iter:
        lr_curr = learning_rate_schedule(lr_max=lr_max, lr_min=lr_min, warmup_iters=warmup_iters, cosine_cycle_iters=cosine_cycle_iters, current_iter=train_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_curr

        for micro_step in range(gradient_accumulation_step):

            with amp_ctx:
                logits = model(X_train)

                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                next_batch_x, next_batch_y = get_batch(
                    train_data,
                    batch_size = batch_size,
                    context_length = context_length,
                    device = str(device)
                )

                train_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_train.view(-1)) / gradient_accumulation_step
                train_loss.backward()

                X_train = next_batch_x
                y_train = next_batch_y

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if train_step > 0 and train_step % 100 == 0:  # Evaluate every 100 steps
            model.eval()
            with torch.no_grad():
                valid_loss = list()
                for _ in range(gradient_accumulation_step):
                    X_valid, y_valid = get_batch(
                        valid_data,
                        batch_size = batch_size,
                        context_length = context_length,
                        device = str(device)
                    )
                    y_pred = model(X_valid)
                    valid_loss.append(F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y_valid.view(-1)).item())
                valid_loss = sum(valid_loss) / len(valid_loss)
            logger.info(f"Validation loss at step {train_step}: {valid_loss:.6f}")
            # logger.info(f"Training loss at step {train_step}: {train_loss.item():.6f}")
            model.train()
            save_checkpoint(model, optimizer, train_step, os.path.join(checkpoint_path, f"./{model_prefix}_{train_step}.pt"))
        train_step += 1
    save_checkpoint(model, optimizer, train_step, os.path.join(checkpoint_path, f"./{model_prefix}_{train_step}.pt"))