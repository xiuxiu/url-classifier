"""
URL Classifier Inference

Usage:
    python src/infer.py "https://example.com/product/12345"
    python src/infer.py "https://example.com/products/list"
"""

import os
import sys

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_FILE_DIR)                          # url-classifier/
_PARENT_DIR = os.path.dirname(_PROJ_DIR)                        # Projects/
_CHECKPOINT = os.path.join(_PARENT_DIR, "url-autoresearch", "checkpoint_pre_eval.pt")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tokenizer (from prepare.py)
# ---------------------------------------------------------------------------

import pickle
import tiktoken

_CACHE_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~/.cache")), "autoresearch", "url")
os.makedirs(_CACHE_DIR, exist_ok=True)

class Tokenizer:
    def __init__(self, encoder):
        self._enc = encoder
        self.pad_id = 0
        self.bos_id = 1

    def get_vocab_size(self):
        return 100277

    def encode(self, text):
        ids = self._enc.encode(text, allowed_special="all")
        return [self.bos_id] + ids

    @classmethod
    def from_directory(cls):
        cache = os.path.join(_CACHE_DIR, "tokenizer_iowacat.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                obj = pickle.load(f)
                enc = obj["encoder"] if isinstance(obj, dict) else obj
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return cls(enc)


# ---------------------------------------------------------------------------
# Model architecture (mirrors train_b.py exactly)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 64
DEPTH = 4
ASPECT_RATIO = 96
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.layer_idx = layer_idx
        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) \
            if has_ve(layer_idx, config.n_layer) else None
        self._mask_cache = {}

    def _get_mask(self, T, window, device):
        key = (T, window, device.type)
        if key not in self._mask_cache:
            row = torch.arange(T, device=device)
            col = torch.arange(T, device=device).unsqueeze(1)
            m = col <= row  # causal
            if window and 0 <= window < T:
                m = m & (col >= row - window)
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def forward(self, x, ve, cos_sin, window_size):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_head < self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        window = window_size[0] if isinstance(window_size, tuple) else window_size
        mask = self._get_mask(T, window, q.device)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config.n_embd)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    sequence_len: int
    n_kv_head: int = None
    head_dim: int = None
    window_pattern: str = "SSSL"
    use_activation_checkpointing: bool = False
    compute_dtype = torch.bfloat16

    def __post_init__(self):
        self.n_kv_head = self.n_kv_head or self.n_head
        self.head_dim = self.head_dim or (self.n_embd // self.n_head)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.class_head = nn.Linear(config.n_embd, 2, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, config.n_kv_head * config.head_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })
        head_dim = config.head_dim
        cos, sin = self._precompute_rotary(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().to(dtype=torch.bfloat16)
        sin = freqs.sin().to(dtype=torch.bfloat16)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self):
        long_w = self.config.sequence_len
        short_w = long_w // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = []
        for i in range(self.config.n_layer):
            ch = self.config.window_pattern[i % len(self.config.window_pattern)]
            sizes.append(char_to_w[ch])
        sizes[-1] = (long_w, 0)
        return sizes

    def forward(self, idx):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        softcap = 15.0
        lm_logits = softcap * torch.tanh(self.lm_head(x).float() / softcap)
        class_logits = self.class_head(x[:, -1, :])
        return {"lm_logits": lm_logits, "class_logits": class_logits}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

LABELS = {0: "A — 列表页", 1: "B — 详情页"}


def load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    base_dim = DEPTH * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM  # 384
    n_head = model_dim // HEAD_DIM  # 3

    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=DEPTH,
        n_head=n_head,
        n_embd=model_dim,
        sequence_len=MAX_SEQ_LEN,
        n_kv_head=n_head,
        head_dim=HEAD_DIM,
        window_pattern=WINDOW_PATTERN,
        use_activation_checkpointing=False,
    )

    model = GPT(config)
    state = torch.load(_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n_params / 1e6:.1f}M params | device={device}")
    return model, tokenizer, device


def predict(url, model, tokenizer, device):
    ids = tokenizer.encode(url)
    ids = ids[:MAX_SEQ_LEN - 1]

    pad_len = MAX_SEQ_LEN - 1 - len(ids)
    ids = [tokenizer.pad_id] * pad_len + [tokenizer.bos_id] + ids

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        with torch.amp.autocast(device.type if device.type == "cuda" else "cpu", dtype=dtype):
            out = model(x)

    class_logits = out["class_logits"][0]
    probs = torch.softmax(class_logits.float(), dim=-1)
    pred = int(probs.argmax())
    conf = float(probs[pred])
    return pred, conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py <url>")
        sys.exit(1)

    model, tokenizer, device = load()
    url = sys.argv[1]
    pred, conf = predict(url, model, tokenizer, device)

    print(f"\nURL: {url}")
    print(f"预测: {LABELS[pred]}")
    print(f"置信度: {conf:.2%}")
