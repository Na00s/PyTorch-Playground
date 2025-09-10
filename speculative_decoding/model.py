import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, List, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class GEMMA3_270M_CONFIG:
    vocab_size: int = 262_144
    context_length: int = 32_768
    emb_dim: int = 640
    n_heads: int = 4
    n_layers: int = 18
    hidden_dim: int = 2048
    head_dim: int = 256
    n_kv_groups: int = 1
    sliding_window: int = 512
    rope_local_base: float = 10_000.0
    rope_base: float = 1_000_000.0
    qk_norm: bool = True
    layer_types: Tuple[str, ...] = ("sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention")
    dtype: torch.dtype = torch.bfloat16

class TransformerEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_wrd_embed = nn.Embedding(config.vocab_size, config.emb_dim)
    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.context_length
        x = self.tok_wrd_embed(idx)
        x = x * math.sqrt(self.config.emb_dim)
        return x

class GroupedQueryAttention(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.proj_q = nn.Linear(config.emb_dim, config.n_heads * config.head_dim)
        self.proj_k = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim)
        self.proj_v = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim)
        self.proj_o = nn.Linear(config.n_heads * config.head_dim, config.emb_dim)
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(config.head_dim, eps=1e-12)
            self.k_norm = nn.RMSNorm(config.head_dim, eps=1e-12)
        else:
            self.q_norm = None
            self.k_norm = None
    @staticmethod
    def get_rope_params(length, angle, dim, device):
        wavelength = (torch.arange(0, dim, 2, device = device) / dim).unsqueeze(0) #1, dim//2
        freq = 1 / (angle ** wavelength) #1, dim//2
        positions = torch.arange(0, length, device = device).unsqueeze(1) #T, 1
        angles = freq * positions #T, dim//2
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin
    @staticmethod
    def apply_rope(x, cos, sin):
        x_even = x[..., 0::2] #B, T, C//2
        x_odd = x[..., 1::2] #B, T, C//2
        a = x_even * cos - x_odd * sin
        b = x_even * sin + x_odd * cos
        x = torch.stack([a, b], dim=-1) #B, T, C//2, 2
        x = x.flatten(-2) #B, T, C
        return x
    def forward(self, x):
        B, T, C = x.shape
        q = self.proj_q(x) #B, T, n_heads * head_dim
        k = self.proj_k(x) #B, T, n_kv_groups * head_dim
        v = self.proj_v(x) #B, T, n_kv_groups * head_dim

        q = q.reshape(B, T, self.config.n_heads, self.config.head_dim)
        k = k.reshape(B, T, self.config.n_kv_groups, self.config.head_dim)
        v = v.reshape(B, T, self.config.n_kv_groups, self.config.head_dim)

        q = q.transpose(1, 2) #B, n_heads, T, head_dim
        k = k.transpose(1, 2) #B, n_kv_groups, T, head_dim
        v = v.transpose(1, 2) #B, n_kv_groups, T, head_dim

        repeat_factor = self.config.n_heads // self.config.n_kv_groups

        q = q.reshape(B, self.config.n_kv_groups, repeat_factor, T, self.config.head_dim) #B, n_kv_groups, repeat_factor, T, head_dim
        k = k.reshape(B, self.config.n_kv_groups, 1, T, self.config.head_dim) #B, n_kv_groups, 1, T, head_dim
        v = v.reshape(B, self.config.n_kv_groups, 1, T, self.config.head_dim) #B, n_kv_groups, 1, T, head_dim

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.layer_type == "sliding_attention":
            theta = self.config.rope_local_base
            mask = torch.triu(torch.tril(torch.ones(T, T, device=x.device)), diagonal=-self.config.sliding_window)
        else:
            theta = self.config.rope_base
            mask = torch.tril(torch.ones(T, T, device=x.device))
        
        cos, sin = self.get_rope_params(T, theta, self.config.head_dim, device=x.device)
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        scores = q @ k.transpose(-1, -2) #B, n_kv_groups, repeat_factor, T, T
        scores = scores / math.sqrt(self.config.head_dim)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(scores, dim=-1)

        ctx = scores @ v #B, n_kv_groups, repeat_factor, T, head_dim

        ctx = ctx.reshape(B, self.config.n_heads, T, self.config.head_dim) #B, n_heads, T, head_dim
        ctx = ctx.transpose(1, 2) #B, T, n_heads, head_dim
        ctx = ctx.reshape(B, T, self.config.n_heads * self.config.head_dim) #B, T, n_heads*head_dim

        x = self.proj_o(ctx)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.emb_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.emb_dim, config.hidden_dim)
        self.fc3 = nn.Linear(config.hidden_dim, config.emb_dim)
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        act = x1 * F.silu(x2)
        out = self.fc3(act)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.config = config
        self.attn = GroupedQueryAttention(config, layer_type)
        self.ffn = FeedForwardNetwork(config)
        self.n1 = nn.RMSNorm(config.emb_dim, eps=1e-12)
        self.n2 = nn.RMSNorm(config.emb_dim, eps=1e-12)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = TransformerEmbedder(config)
        self.blocks = nn.ModuleList([TransformerBlock(config, layer_type) for layer_type in config.layer_types])
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)
    def forward(self, idx, targets=None):
        x = self.embedder(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        if targets is not None:
            B, T, C = logits.shape
            assert (targets.shape == (B, T))
            loss = F.cross_entropy(logits.reshape(B*T, C), targets.reshape(B*T))
        else:
            loss = None
        return logits, loss
    def generate(self, idx, max_new_tokens=500):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=-1)
        return idx