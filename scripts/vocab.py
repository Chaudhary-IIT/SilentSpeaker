# scripts/vocab.py

import string
import json

# BLANK_ID = 0 for CTC loss: idx 0 is "blank".
# Space, a-z, and apostrophe are allowed tokens in GRID/caption style
CHARS = list(" " + string.ascii_lowercase + "'")  # 28 chars: space + a-z + apostrophe

PAD = ""
BLANK_ID = 0

def build_vocab():
    """
    Builds vocabulary mappings for CTC training.
    Returns:
        char2idx: dict mapping char->int
        idx2char: list of chars with index=position
    """
    idx2char = [""] + CHARS  # idx 0 is blank, then space, a, ..., z, '
    char2idx = {c: i for i, c in enumerate(idx2char)}
    return char2idx, idx2char

def text_to_int(s, char2idx):
    """
    Converts a string into a list of integer character indices.
    Skips characters not in char2idx.
    """
    s = s.lower()
    return [char2idx[c] for c in s if c in char2idx]

def int_to_text(ids, idx2char):
    """
    Converts a list of indices into a string using idx2char mapping.
    """
    return "".join(idx2char[i] for i in ids if i < len(idx2char))

def save_vocab(path, idx2char):
    """
    Saves idx2char mapping as JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"idx2char": idx2char}, f)

def load_vocab(path):
    """
    Loads idx2char from JSON, rebuilds char2idx.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    idx2char = obj["idx2char"]
    char2idx = {c: i for i, c in enumerate(idx2char)}
    return char2idx, idx2char
