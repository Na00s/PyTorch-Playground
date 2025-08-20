import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 262_144
    context_length: int = 32_768
    emb_dim: int = 640
    n_heads: int = 4
    n_layers: int = 18
    hidden_dim: int = 2048
    head_dim: int = 256
    sliding_window: int = 512
    n_kv_groups: int = 1
    rope_local_base: float = 10_000.0
    rope_base: float = 1_000_000.0
    qk_norm: bool = True

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # B, T, C = x.shape
    x0 = x[..., 0::2] 
    x1 = x[..., 1::2]
    x_rotated_even = x0 * cos - x1 * sin
    x_rotated_odd = x0 * sin + x1 * cos
    x = torch.stack((x_rotated_even, x_rotated_odd), dim=-1) #B, T, C//2, 2
    x = x.flatten(-2) #B, T, C
    return x


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.emb_dim, config.hidden_dim)
        self.w2 = nn.Linear(config.emb_dim, config.hidden_dim)
        self.w3 = nn.Linear(config.hidden_dim, config.emb_dim)

    def forward(self, x):
        B, T, C = x.shape
        x1 = self.w1(x)  # B, T, hidden_dim
        x2 = self.w2(x)  # B, T, hidden_dim
        x1 = F.gelu(x1)  # B, T, hidden_dim
        x = x1 * x2      # B, T, hidden_dim
        x = self.w3(x)   # B, T, emb_dim
        return x


class TransformerEmbedder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb_table = nn.Embedding(config.vocab_size, config.emb_dim)
        self.proj_in = nn.Linear(config.emb_dim, config.emb_dim)
    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.context_length
        x = self.tok_emb_table(idx)  # B, T, emb_dim
        x = self.proj_in(x)
        return x


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        assert config.emb_dim % config.n_heads == 0
        assert config.n_heads % config.n_kv_groups == 0
        assert config.head_dim % 2 == 0

        self.proj_q = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.proj_k = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim, bias=False)
        self.proj_v = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim, bias=False)
        self.proj_o = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.register_buffer("mask", torch.triu(torch.tril(torch.ones(config.context_length, config.context_length)),diagonal=-config.sliding_window,),persistent=False)

        inv_freq = 1.0 / (config.rope_base ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))  # (D/2,)
        t = torch.arange(config.context_length, dtype=torch.float)  # (T,)
        freqs = torch.outer(t, inv_freq)  # (T, D/2)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (T, D)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # (T, D)
        
        self.register_buffer("rope_cos", cos[None, None, :, :], persistent=False)
        self.register_buffer("rope_sin", sin[None, None, :, :], persistent=False)

        if config.qk_norm:
            self.q_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x):
        B, T, C = x.shape #B, T, emb_dim
        
        repeat_factor = self.config.n_heads // self.config.n_kv_groups

        q = self.proj_q(x).reshape(B, T, self.config.n_heads, self.config.head_dim)   # B, T, n_heads, head_dim
        k = self.proj_k(x).reshape(B, T, self.config.n_kv_groups, self.config.head_dim)   # B, T, n_kv_groups, head_dim
        v = self.proj_v(x).reshape(B, T, self.config.n_kv_groups, self.config.head_dim)   # B, T, n_kv_groups, head_dim

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        
        cos = self.rope_cos[:, :, :T, :].to(dtype=q.dtype, device=q.device)  # 1, 1, T, head_dim
        sin = self.rope_sin[:, :, :T, :].to(dtype=q.dtype, device=q.device)  # 1, 1, T, head_dim
        q = apply_rope(q, cos, sin)  # B, T, n_heads, head_dim
        k = apply_rope(k, cos, sin)  # B, T, n_kv_groups, head_dim

        q = q.transpose(1, 2)  # B, n_heads, T, head_dim
        k = k.transpose(1, 2)  # B, n_kv_groups, T, head_dim
        v = v.transpose(1, 2)  # B, n_kv_groups, T, head_dim


        k = k.repeat_interleave(repeat_factor, dim=1)  # B, H, T, head_dim
        v = v.repeat_interleave(repeat_factor, dim=1)  # B, H, T, head_dim

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.config.head_dim)  # B, H, T, T
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = attn @ v                             # B, n_heads, T, head_dim
        out = out.transpose(1, 2).reshape(B, T, C) # B, T, emb_dim
        out = self.proj_o(out)                     # B, T, emb_dim
        return out



class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.n1 = nn.RMSNorm(config.emb_dim)
        self.n2 = nn.RMSNorm(config.emb_dim)
        self.n3 = nn.RMSNorm(config.emb_dim)
        self.n4 = nn.RMSNorm(config.emb_dim)
        self.attn = GroupedQueryAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.n2(x + self.attn(self.n1(x)))
        x = self.n4(x + self.ffn(self.n3(x)))
        return x


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed = TransformerEmbedder(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.n_f = nn.RMSNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        x = self.embed(idx)                  # B, T, C
        for blk in self.blocks:
            x = blk(x)
        x = self.n_f(x)
        logits = self.lm_head(x)             # B, T, V
        return logits
    
#config = TransformerConfig()
#m = TransformerModel(config)

#num_parameters = sum([p.numel() for p in m.parameters()])
#print(num_parameters)


