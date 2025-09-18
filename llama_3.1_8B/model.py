import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Literal, List, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class LLAMA3_CONFIG:
    attention_bias: bool = False
    hidden_act: Literal["relu", "gelu", "swiglu", "silu", "geglu", "glu"] = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 14336
    max_position_embeddings: int = 131072
    mlp_bias: bool = False
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 8
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-05
    rope_scaling: bool = True
    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 128256
    head_size: int = 128
    use_cache: bool = False

class TransformerEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb_table = nn.Embedding(config.vocab_size, config.hidden_size)
    def forward(self, idx):
        B, T = x.shape
        assert (T <= self.config.original_max_position_embeddings)
        x = self.tok_emb_table(idx) #B, T, C
        return x

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj_q = nn.Linear(config.hidden_size, config.num_attention_heads*config.head_size)
        self.proj_k = nn.Linear(config.hidden_size, config.num_key_value_heads*config.head_size)
        self.proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads*config.head_size)
        self.proj_o = nn.Linear(config.num_attention_heads*config.head_size, config.hidden_size)
    @staticmethod
    def get_rope_params(theta, length, head_dim, past_len=0):
        freq = theta ** ((-torch.arange(0, head_dim, 2))/head_dim).unsqueeze(0) # 1, head_dim//2
        pos = torch.arange(past_len, past_len + length).unsqueeze(1) #T, 1
        angles = pos * freq #T, head_dim//2
        cos = torch.cos(angles) #T, head_dim//2
        sin = torch.sin(angles) #T, head_dim//2
        return cos, sin
    @staticmethod
    def apply_rope(x, cos, sin):
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        x_even = x[..., 0::2] #..., C//2
        x_odd = x[..., 1::2] #..., C//2
        u = x_even*cos - x_odd*sin
        v = x_even*sin + x_odd*cos
        x = torch.stack([u, v], dim=-1) #..., C//2, 2
        x = x.flatten(-2) #..., C
        return x
    def forward(self, x, past_k=None, past_v=None, use_cache=False):
        B, T, C = x.shape
        q = self.proj_q(x) #B, T, num_attention_heads*head_size
        k = self.proj_k(x) #B, T, num_key_value_heads*head_size
        v = self.proj_v(x) #B, T, num_key_value_heads*head_size

        q = q.reshape(B, T, self.config.num_attention_heads, self.config.head_size) #B, T, num_attention_heads, head_size
        k = k.reshape(B, T, self.config.num_key_value_heads, self.config.head_size) #B, T, num_key_value_heads, head_size
        v = v.reshape(B, T, self.config.num_key_value_heads, self.config.head_size) #B, T, num_key_value_heads, head_size

        q = q.transpose(1, 2) #B, num_attention_heads, T, head_size
        k = k.transpose(1, 2) #B, num_key_value_heads, T, head_size
        v = v.transpose(1, 2) #B, num_key_value_heads, T, head_size

        past_length = 0 if past_k is None else past_k.size(2)
        cos, sin = self.get_rope_params(self.config.rope_theta, T, self.config.head_size, past_len=past_length)
        q = self.apply_rope(q, cos, sin) #B, num_key_value_heads, T, head_size
        k = self.apply_rope(k, cos, sin) #B, num_key_value_heads, T, head_size

        if past_k is not None:
            k_all = torch.cat([past_k, k], dim=2) #B, num_key_value_heads, T+past_length, head_size
            v_all = torch.cat([past_v, v], dim=2) #B, num_key_value_heads, T+past_length, head_size
        else:
            k_all, v_all = k, v

        repeat_factor = self.config.num_attention_heads // self.config.num_key_value_heads

        q = q.reshape(B, self.config.num_key_value_heads, repeat_factor, T, self.config.head_size) #B, num_key_value_heads, repeat_factor, T, head_size
        k_all = k_all.unsqueeze(2) #B, num_key_value_heads, 1, T + past_length, head_size
        v_all = v_all.unsqueeze(2) #B, num_key_value_heads, 1, T + past_length, head_size

        scores = q @ k_all.transpose(-1, -2) #B, num_key_value_heads, repeat_factor, T, T+past_length
        mask = torch.tril(torch.ones(T, T+past_length, device=x.device))
        scores = scores.masked_fill(mask==0, float("-inf"))
        scores = scores / math.sqrt(self.config.head_size)
        scores = F.softmax(scores, dim=-1)
        ctx = scores @ v_all #B, num_key_value_heads, repeat_factor, T, head_size
        ctx = ctx.reshape(B, self.config.num_attention_heads, T, self.config.head_size).transpose(1, 2) #B, T, num_attention_heads, head_size
        ctx = ctx.reshape(B, T, C)

        x = self.proj_o(ctx) #B, T, C

        k_all = k_all.squeeze(2)
        v_all = v_all.squeeze(2)

        present = (k_all, v_all) if use_cache else None

        return x, present

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc3 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    def forward(self, x):
        return self.fc3(F.silu(self.fc2(x)) * self.fc1(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gqa = GroupedQueryAttention(config)  
        self.ffn = FeedForwardNetwork(config)
        self.n1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.n2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, past=None, use_cache=False):
        residual = x
        x = self.n1(x)
        attn_out, present = self.gqa(
            x,
            past_k=None if past is None else past[0],
            past_v=None if past is None else past[1],
            use_cache=self.config.use_cache,
        )
        x = residual + attn_out

        residual = x
        x = self.n2(x)
        x = residual + self.ffn(x)

        return x, present


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = TransformerEmbedder(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj_o = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.proj_o.weight = self.embedder.tok_emb_table.weight

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        x = self.embedder(idx)
        presents = [] if use_cache else None

        if past_kv is None:
            past_kv = [None] * len(self.blocks)

        for block, past in zip(self.blocks, past_kv):
            x, present = block(x, past=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.proj_o(x)

        loss = None
        if targets is not None:
            # shift logits and targets for causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )

        return (logits, loss, presents if use_cache else None)







class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # x: [B, T, C]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normed


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: [B, T, C]
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_normed = (x - mean) / torch.sqrt(var + self.eps)

        if self.weight is not None:
            x_normed = self.weight * x_normed
        if self.bias is not None:
            x_normed = x_normed + self.bias

        return x_normed

            

            

