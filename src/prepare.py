"""
Data preparation for URL classification with Autoresearch.
Uses tiktoken cl100k_base (no custom BPE training needed).

Usage:
    python prepare.py
"""

import json
import math
import os
import pickle
import random

import torch
import tiktoken

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 64           # URLs average ~30-50 tokens; 64 covers everything, 256 is overkill
TIME_BUDGET = 1800
VOCAB_SIZE = 100277        # cl100k_base vocab size

DEFAULT_DATASET = "iowacat"
DATASET_CHOICES = ("iowacat",)


def _cache_dir():
    lap = os.environ.get("LOCALAPPDATA")
    base = os.path.join(lap, "autoresearch", "url") if lap \
        else os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "url")
    os.makedirs(base, exist_ok=True)
    return base


CACHE_DIR = _cache_dir()


# ---------------------------------------------------------------------------
# Tokenizer (wraps tiktoken cl100k_base)
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, encoder, dataset):
        self._enc = encoder
        self._ds = dataset
        # Token IDs for our classification tokens:
        # bos=1 (tiktoken already uses 1 for <|im_start|> etc)
        self.pad_id = 0
        self.bos_id = 1
        # After bos, we put the URL tokens, then eos
        self._eos_id = 1  # We'll use bos as eos too for simplicity

    @property
    def dataset(self):
        return self._ds

    def get_vocab_size(self):
        return VOCAB_SIZE

    def encode(self, text, add_bos=True, add_eos=False):
        ids = self._enc.encode(text, allowed_special="all")
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.bos_id]
        return ids

    def decode(self, ids):
        return self._enc.decode([i for i in ids if i not in (0, 1)])

    @classmethod
    def from_directory(cls, dataset=None):
        dataset = dataset or DEFAULT_DATASET
        cache = os.path.join(CACHE_DIR, f"tokenizer_{dataset}.pkl")
        if os.path.exists(cache):
            print(f"Loading cached tokenizer from {cache}")
            with open(cache, "rb") as f:
                return cls(**pickle.load(f))
        enc = tiktoken.get_encoding("cl100k_base")
        obj = cls(enc, dataset)
        with open(cache, "wb") as f:
            pickle.dump({"encoder": enc, "dataset": dataset}, f)
        return obj


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(dataset=None):
    """Load raw (url, label) pairs from local JSON or HuggingFace."""
    dataset = dataset or DEFAULT_DATASET
    cache = os.path.join(CACHE_DIR, f"data_{dataset}.pkl")

    if os.path.exists(cache):
        print(f"Loading cached data from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print(f"Loading dataset: {dataset}")

    if dataset == "iowacat":
        pairs = _load_from_json()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    with open(cache, "wb") as f:
        pickle.dump(pairs, f)
    print(f"  Saved cache: {len(pairs)} samples")
    return pairs


def _load_from_json():
    """Extract URLs and labels from train.json (conversation format)."""
    path = "C:/Users/windlx/Projects/url-classifier/url-classifier/data/train.json"
    print(f"Reading: {path}")

    urls, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        text = item.get("text", "")
        # Extract URL from the user message
        url = _extract_url(text)
        label = _extract_label(text)
        if url and label is not None:
            urls.append(url)
            labels.append(label)

    n_a = labels.count(0)
    n_b = labels.count(1)
    print(f"  Loaded {len(urls)} samples — A={n_a}, B={n_b}")
    return urls, labels


def _extract_url(text):
    """Pull URL from conversation text field."""
    marker = "URL:\n\n"
    idx = text.find(marker)
    if idx < 0:
        marker = "URL: "
        idx = text.find(marker)
        if idx < 0:
            return None
        idx += len(marker)
    else:
        idx += len(marker)
    end = text.find("<|", idx)
    url = text[idx:end if end > 0 else None].strip()
    return url if url else None


def _extract_label(text):
    """Extract A=0 or B=1 from assistant response."""
    marker = "assistant\n"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    idx += len(marker)
    ch = text[idx:idx+1].strip().upper()
    return 0 if ch == "A" else (1 if ch == "B" else None)


# ---------------------------------------------------------------------------
# Datasets (train / val split)
# ---------------------------------------------------------------------------

def _split(urls, labels, val_frac=0.1, seed=42):
    paired = list(zip(urls, labels))
    random.shuffle(paired)
    n = int(len(paired) * val_frac)
    val = paired[:n]
    train = paired[n:]
    return list(zip(*train)), list(zip(*val))


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class URLDataLoader:
    """Yields (x, y_seq) batches; x = token IDs, y_seq = shifted for LM loss."""

    PAD = 0

    def __init__(self, urls, labels, tokenizer, batch_size, seq_len, shuffle=True, device=None):
        self.urls, self.labels = urls, labels
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self._device = device
        self._pos = 0
        self._epoch = 0
        self._val_urls, self._val_labels = None, None  # for eval accuracy

    def _shuffle_data(self):
        paired = list(zip(self.urls, self.labels))
        random.shuffle(paired)
        self.urls, self.labels = zip(*paired)
        self.urls, self.labels = list(self.urls), list(self.labels)

    def _encode(self, url):
        ids = self.tokenizer.encode(url)
        ids = ids[:self.seq_len - 1]  # leave room for eos/bos
        ids.append(self.tokenizer.bos_id)  # eos ≈ bos for us
        return ids

    def __iter__(self):
        self._pos = 0
        if self.shuffle:
            self._shuffle_data()
        return self

    def __next__(self):
        if self._pos >= len(self.urls):
            self._epoch += 1
            self._pos = 0
            if self.shuffle:
                self._shuffle_data()
        bx, by, batch_labels = [], [], []
        for _ in range(self.batch_size):
            if self._pos >= len(self.urls):
                break
            url = self.urls[self._pos]
            label = self.labels[self._pos]
            x = self._encode(url)
            y = x[1:] + [self.PAD]  # shift right
            x = x + [self.PAD] * (self.seq_len - len(x))
            y = y + [self.PAD] * (self.seq_len - len(y))
            bx.append(x)
            by.append(y)
            batch_labels.append(label)
            self._pos += 1
        dev = self._device
        bx_t = torch.tensor(bx, dtype=torch.long, device=dev)
        by_t = torch.tensor(by, dtype=torch.long, device=dev)
        labels_t = torch.tensor(batch_labels, dtype=torch.long, device=dev)
        return bx_t, by_t, labels_t, self._epoch


def make_dataloader(tokenizer, batch_size, seq_len, split, device, dataset=None):
    """Create a DataLoader for train or val split."""
    urls, labels = _load_data(dataset)
    train_pairs, val_pairs = _split(urls, labels)
    which = train_pairs if split == "train" else val_pairs
    urls_s, labels_s = which
    shuffle = split == "train"
    return URLDataLoader(urls_s, labels_s, tokenizer, batch_size, seq_len, shuffle=shuffle, device=device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bpb(model, tokenizer, device, val_batch_size=32, dataset=None):
    """Compute bits-per-byte (LM loss) on validation set."""
    val_loader = make_dataloader(tokenizer, val_batch_size, MAX_SEQ_LEN, "val", device, dataset)
    total_bits = 0.0
    total_tokens = 0

    model.eval()
    autocast = torch.amp.autocast("cuda", dtype=torch.float16)
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast:
                _, loss = model(x, y)
            if loss is not None:
                total_bits += loss.item() * x.numel()
                total_tokens += x.numel()

    bpb = (total_bits / max(total_tokens, 1)) / math.log(2)
    model.train()
    return bpb


def evaluate_accuracy(model, tokenizer, device, val_batch_size=64, dataset=None, max_batches=30):
    """
    Compute classification accuracy on validation set.
    Strategy: for each URL, encode it and look at the logits of the model.
    Since the model is trained as LM, we use a proxy:
    - Encode URL with [BOS] prefix
    - Look at logits at each position
    - Predict label based on some heuristic (e.g., sequence length, certain tokens)
    Real approach: fine-tune with classification head (see train.py).
    """
    val_loader = make_dataloader(tokenizer, val_batch_size, MAX_SEQ_LEN, "val", device, dataset)
    correct, total = 0, 0

    model.eval()
    autocast = torch.amp.autocast("cuda", dtype=torch.float16)
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            with autocast:
                logits, _ = model(x)
            # LM model doesn't have a classification head yet;
            # proxy: predict based on whether URL contains '/product/' vs '/list' etc.
            # We'll use a simple heuristic until proper classification is wired up
            total += x.size(0)

    # Return 0 until we wire up proper classification
    return correct / max(total, 1)
