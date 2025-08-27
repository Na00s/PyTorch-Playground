import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Literal

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

class TransformerEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb_table = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.max_position_embeddings
        x = self.tok_emb_table(idx) #B, T, hidden_size
        return x
    
    
class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj_q = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_size, bias=False)
        self.proj_k = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_size, bias=False)
        self.proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_size, bias=False)
        self.proj_o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    @staticmethod
    def get_rope_params(theta, head_size, length, device):
        wavelength = theta ** (torch.arange(0, head_size, 2, device=device) / head_size).unsqueeze(0).float() #1, head_dim//2
        freq = 1 / wavelength
        positions = torch.arange(length, device=device).unsqueeze(1).float() #length, 1
        angles = positions @ freq #length, head_dim//2
        cos = torch.cos(angles) #length, head_dim//2
        sin = torch.sin(angles) #length, head_dim//2
        return cos, sin
    
    @staticmethod
    def apply_rope(x, cos, sin):
        x_even = x[..., 0::2] #B, T, head_dim//2
        x_odd = x[..., 1::2] #B, T, head_dim//2
        a = x_even * cos - x_odd * sin #B, T, head_dim//2
        b = x_even * sin + x_odd * cos #B, T, head_dim//2
        x = torch.stack([a, b], dim=-1) #B, T, head_dim//2, 2
        x = x.flatten(-2) #B, T, head_dim
        return x
    
    def forward(self, x):
        B, T, C = x.shape #B, T, hidden_size
        q = self.proj_q(x) #B, T, hidden_size
        k = self.proj_k(x) #B, T, n_kv_heads * head_size
        v = self.proj_v(x) #B, T, n_kv_heads * head_size

        q = q.reshape(B, T, self.config.num_attention_heads, self.config.head_size)
        k = k.reshape(B, T, self.config.num_key_value_heads, self.config.head_size)
        v = v.reshape(B, T, self.config.num_key_value_heads, self.config.head_size)

        q = q.transpose(1, 2) #B, n_heads, T, head_size
        k = k.transpose(1, 2) #B, n_kv_heads, T, head_size
        v = v.transpose(1, 2) #B, n_kv_heads, T, head_size

        repeat_factor = self.config.num_attention_heads // self.config.num_key_value_heads

        q = q.unsqueeze(2).reshape(B, self.config.num_key_value_heads, repeat_factor, T, self.config.head_size) #B, n_kv_heads, repeat_factor, T, head_size
        k = k.unsqueeze(2) #B, n_kv_heads, 1, T, head_size
        v = v.unsqueeze(2) #B, n_kv_heads, 1, T, head_size
        
        cos, sin = self.get_rope_params(self.config.rope_theta, self.config.head_size, T, device=x.device) #T, head_dim//2
        q, k = self.apply_rope(q, cos, sin), self.apply_rope(k, cos, sin)

        scores = q @ k.transpose(3, 4) #B, n_kv_heads, repeat_factor, T, T
        scores = scores / math.sqrt(self.config.head_size)

        mask = torch.tril(torch.ones((T, T), device=x.device))

        scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(scores, dim=-1)
        scores = scores.to(dtype=self.config.torch_dtype)
        ctx = scores @ v #B, n_kv_heads, repeat_factor, T, head_size
        ctx = ctx.reshape(B, self.config.num_attention_heads, T, self.config.head_size)
        ctx = ctx.transpose(1, 2) #B, T, n_heads, head_size
        ctx = ctx.reshape(B, T, self.config.hidden_size)

        x = self.proj_o(ctx)
        
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc3 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = x2 * F.silu(x1)
        x = self.fc3(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gqa = GroupedQueryAttention(config)
        self.ffn = FeedForwardNetwork(config)
        self.n1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.n2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x):
        x = x + self.gqa(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = TransformerEmbedder(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.proj_o = nn.Linear(config.hidden_size, config.vocab_size)
    def forward(self, idx, targets=None):
        x = self.embedder(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.proj_o(x)
        if targets is not None:
            B, T, C = logits.shape
            assert targets.shape == idx.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(-1))
        else:
            loss = None
        return logits, loss
    

config = LLAMA3_CONFIG()    
m = TransformerModel(config).to(device=device, dtype=config.torch_dtype)

num_params = sum([p.numel() for p in m.parameters()])
print(num_params)

            

            

