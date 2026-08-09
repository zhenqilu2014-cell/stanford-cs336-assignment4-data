import os
import re
import random
from collections import defaultdict
from nltk.util import ngrams
import numpy as np
from typing import List

PRIME_NUMBER = 2 ** 61 - 1


def exact_line_deduplication(
    input_files: List[os.PathLike], output_directory: os.PathLike
):
    # count number of files each unique line appears
    count = defaultdict(int)
    for filename in input_files:
        seen = set()
        with open(filename, "r") as fin:
            for line in fin.readlines():
                key = hash(line)
                if key in seen:
                    continue
                seen.add(key)
                count[key] += 1

    # keep lines unique to each file, only once
    for filename in input_files:
        output_filename = os.path.join(output_directory, os.path.basename(filename))
        with open(filename, "r") as fin, open(output_filename, "w") as fout:
            seen = set()
            for line in fin.readlines():
                key = hash(line)
                if key in seen or count[key] > 1:
                    continue
                seen.add(key)
                fout.write(line)
    return


def _text_clean(text: str) -> List[str]:
    return re.sub(r"\s+", " ", text.lower()).strip().split()


def _hash_func(grams: List[List[str]], hash_values: np.array) -> int:
    base_hash = [hash(" ".join(gram)) for gram in grams]
    return [min((int(a) * h + int(b)) % PRIME_NUMBER for h in base_hash) for a, b in hash_values]


def _jaccard_similarity(set1: set, set2: set) -> float:
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    num_ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be divisible by num_bands")
    # compute min-hash signature for each document
    hash_values = np.random.randint(1, PRIME_NUMBER, size=(num_hashes, 2))
    hash_signature = dict()
    index = 0
    nonempty_files = list()
    total_count = len(input_files)
    for filename in input_files:
        with open(filename, "r") as fin:
            words = _text_clean(fin.read())
        grams = list(ngrams(words, num_ngrams))
        if len(grams) <= 0:
            continue
        sign = _hash_func(grams, hash_values)
        hash_signature[index] = sign
        index += 1
        if index % 30 == 0:
            print(f"Processed {index} out of {total_count} files")
        nonempty_files.append(filename)
    print(f"Number of non-empty files: {len(nonempty_files)} out of {total_count} input files.")

    # apply locality-sensitive hashing
    n = len(hash_signature)
    found_groups = {i: {i} for i in range(n)}
    found_keys = {i: i for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            flag = False
            for k in range(num_bands):
                lo = k * (num_hashes // num_bands)
                hi = (k + 1) * (num_hashes // num_bands)
                if tuple(hash_signature[i][lo:hi]) == tuple(hash_signature[j][lo:hi]):
                    flag = True
                    break
            if flag:
                with open(nonempty_files[i], "r") as fin1:
                    words = _text_clean(fin1.read())
                grams1 = list(ngrams(words, num_ngrams))
                with open(nonempty_files[j], "r") as fin2:
                    words = _text_clean(fin2.read())
                grams2 = list(ngrams(words, num_ngrams))
                if _jaccard_similarity(set(grams1), set(grams2)) > jaccard_threshold:
                    key_i = found_keys[i]
                    key_j = found_keys[j]
                    if key_i == key_j:
                        continue
                    for key in found_groups[key_j]:
                        found_keys[key] = key_i
                    found_groups[key_i] = found_groups[key_i] | found_groups[key_j]
                    found_groups.pop(key_j)

    # output one file from each group
    for group in found_groups.values():
        rep = random.choice(list(group))
        filename = nonempty_files[rep]
        output_filename = os.path.join(output_directory, os.path.basename(filename))
        with open(filename, "r") as fin, open(output_filename, "w") as fout:
            fout.write(fin.read())