import os

from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import fasttext
import re
from nltk.tokenize import word_tokenize
import numpy as np
from typing import Any

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

lang_detect_model = fasttext.load_model(os.path.join(SCRIPT_DIR, "lid.176.bin"))
nsfw_model = fasttext.load_model(os.path.join(SCRIPT_DIR, "dolma_fasttext_nsfw_jigsaw_model.bin"))
hatespeech_model = fasttext.load_model(os.path.join(SCRIPT_DIR, "dolma_fasttext_hatespeech_jigsaw_model.bin"))
quality_classify_model = fasttext.load_model(os.path.join(SCRIPT_DIR, "text_classifier.bin"))


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        html_decode = html_bytes.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        html_type = detect_encoding(html_bytes)
        html_decode = html_bytes.decode(html_type, errors="ignore")
    return extract_plain_text(html_decode)


def identify_language(text: str) -> tuple[str, float]:
    label, conf = lang_detect_model.predict(text.replace("\n", " "), k = 1)
    return label[0].replace("__label__", ""), conf[0]


def mask_emails(text: str) -> tuple[str, int]:
    # Pattern to extract emails bounded by word boundaries
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return re.sub(pattern, "|||EMAIL_ADDRESS|||", text), len(re.findall(pattern, text))


def mask_phone_numbers(text: str) -> tuple[str, int]:
    # Pattern to extract US phone numbers bounded by non-word boundaries
    pattern = r'(?<!\w)\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})(?!\w)'
    return re.sub(pattern, "|||PHONE_NUMBER|||", text), len(re.findall(pattern, text))


def mask_ips(text: str) -> tuple[str, int]:
    # Pattern to extract IPv4 bounded by word boundaries
    pattern = r'\b(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)){3}\b'
    return re.sub(pattern, "|||IP_ADDRESS|||", text), len(re.findall(pattern, text))


def classify_nsfw(text: str) -> tuple[str, float]:
    label, conf = nsfw_model.predict(text.replace("\n", " "), k = 1)
    return label[0].replace("__label__", ""), conf[0]


def classify_toxic_speech(text: str) -> tuple[str, float]:
    label, conf = hatespeech_model.predict(text.replace("\n", " "), k = 1)
    return label[0].replace("__label__", ""), conf[0]


def gopher_quality_filter(text: str) -> bool:
    tokens = word_tokenize(text)
    word_count = sum([token.isalpha() for token in tokens])
    if word_count < 50 or word_count > 100_000:
        return False
    avg_word_len = np.mean([len(token) for token in tokens if token.isalpha()])
    if avg_word_len < 3 or avg_word_len > 10:
        return False
    ellipsis_count = len([token for token in tokens if token == "..."])
    sentence_count = len([token for token in tokens if not token.isalpha()])
    if sentence_count > 0 and ellipsis_count / sentence_count > 0.3:
        return False
    alpha_count = len([token for token in tokens if sum(1 for char in token if char.isalpha()) > 0])
    if alpha_count / word_count < 0.8:
        return False
    return True


def classify_quality(text: str, model = quality_classify_model) -> tuple[str, float]:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    label, conf = model.predict(text)
    return label[0].replace("__label__", ""), conf[0]