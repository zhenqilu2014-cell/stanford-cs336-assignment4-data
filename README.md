# CS336 Spring 2026 Assignment 4: Data

For a full description of the assignment, see the assignment handout at
[cs336_assignment4_data.pdf](./cs336_assignment4_data.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336_basics`](./cs336_basics): This module contains the staff 
  implementation of the language model from assignment 1. You will use this training code
  to train an LM on your filtered data. You should not modify the training logic, since
  your leaderboard submission must use it exactly.
- [`./cs336_data`](./cs336_data): This folder is basically empty! This is the
  module where you will implement code to filter and process the data.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   └── ... an optimized training implementation ...
├── cs336_data  # TODO(you): code that you'll write for assignment 4
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 4 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 4 ...
```

As in previous assignments, we use `uv` to manage dependencies.

## Downloading data

### For students

Data is available at `/shared-data`, such as `uv run modal shell ./scripts/download_data.py::main --cmd "ls -l /shared-data"`

To only download the files needed for running offline, run `uv run scripts/download_data.py --offline-only`.
To download all data for the students, run `uv run modal run scripts/download_data.py`

### For non-students

To only download the files needed for running offline, run `uv run scripts/download_data.py --offline-only`.

Implement the method `is_english` in [`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) before downloading the full non-offline data.

#### Modal

Remove `environment_name` from `shared_data_volume` in [`./cs336_data/modal_utils.py`](./cs336_data/modal_utils.py)

```python
uv run modal run scripts/download_data.py
```

#### Non-modal

Change the path in [`./cs336_data/common.py`](./cs336_data/common.py) and run `uv run scripts/download_data.py`.

Consider changing `n_files` for `EnglishWetFiles` in [`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) if you want to download less than 2.5k WET files.

## Training on Modal

The Modal entrypoint in [`./scripts/train.py`](./scripts/train.py) contains the full training config; pass the path to your GPT-2-tokenized training data with `--train-bin`.

The final training run uses 8 B200 GPUs:

```sh
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a zip with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.

## My Contributions

This assignment involved building a complete data pipeline for language model training, including data quality filtering, deduplication, tokenization, training, and evaluation.

### Data Quality Filtering
- `cs336_data/data_quality.py` — Implemented data quality filters and heuristics to identify high-quality text documents from Common Crawl.
- `cs336_data/train_quality_classifier.ipynb` — Trained a quality classifier to score and filter documents based on learned quality signals.

### Deduplication
- `cs336_data/deduplication.py` — Implemented document-level deduplication to remove near-duplicate documents from the training corpus, reducing redundancy and improving data diversity.

### Language Modeling Pipeline
- `cs336_data/language_modeling.py` — Built the full language modeling pipeline: data loading, tokenization with GPT-2 tokenizer, batching, and training/evaluation loop integration.
- `cs336_data/language_modeling.ipynb` — Companion notebook for interactive exploration of the language modeling pipeline, including data inspection, tokenization visualization, and evaluation on Paloma validation sets.

### Common Crawl Processing
- `cs336_data/common_crawl.py` — Implemented processing logic for Common Crawl WET files, including extraction, filtering, and serialization into binary format for efficient training.

### Baseline Model Integration
- `cs336_basics/optimizer.py` — Extended the baseline optimizer to support the updated training pipeline.

### Testing
- `tests/adapters.py` — Updated test adapters to validate the new data filtering, deduplication, and language modeling implementations.
