# scripts/vocab.py
import string, json
# Blank id = 0 for CTC
CHARS = list(" " + string.ascii_lowercase + "'")  # 28 tokens: space + a-z + apostrophe
PAD = "<pad>"
BLANK_ID = 0

def build_vocab():
    idx2char = ["<blank>"] + CHARS
    char2idx = {c:i for i,c in enumerate(idx2char)}
    return char2idx, idx2char

def text_to_int(s, char2idx):
    s = s.lower()
    return [char2idx[c] for c in s if c in char2idx]

def int_to_text(ids, idx2char):
    return "".join(idx2char[i] for i in ids if i < len(idx2char))

def save_vocab(path, idx2char):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"idx2char": idx2char}, f)

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    idx2char = obj["idx2char"]
    char2idx = {c:i for i,c in enumerate(idx2char)}
    return char2idx, idx2char
