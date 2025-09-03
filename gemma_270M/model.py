import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Tuple

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
    layer_type: Tuple[str, ...] = ("sliding_attention",
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
        self.embed_table = nn.Embedding(config.vocab_size, config.emb_dim)

    def forward(self, idx):
        x = self.embed_table(idx)
        x = x * math.sqrt(self.config.emb_dim)
        return x

class GroupedQueryAttention(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.proj_q = nn.Linear(config.emb_dim, config.n_heads * config.head_dim, bias=False)
        self.proj_k = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim, bias=False)
        self.proj_v = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim, bias=False)
        self.proj_o = nn.Linear(config.n_heads * config.head_dim, config.emb_dim, bias=False)
        if config.qk_norm == True:
            self.q_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None
    
    @staticmethod
    def get_rope_params(head_dim, theta, length, dtype=torch.float32):
        freq = theta ** (-torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim).unsqueeze(0) #1, head/dim//2
        positions = torch.arange(length, dtype=dtype, device=device).unsqueeze(1) #length, 1
        angles = positions * freq #length, head_dim//2
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin
    
    @staticmethod
    def apply_rope(x, cos, sin):
        #B, T, C = x.shape
        x_even = x[..., 0::2] #B, T, C//2
        x_odd = x[..., 1::2] #B, T, C//2
        a = x_even*cos - x_odd*sin #B, T, C//2
        b = x_even*sin + x_odd*cos #B, T, C//2
        x = torch.stack([a, b], dim = -1) #B, T, C//2, 2
        x = x.flatten(-2) #B, T, C
        return x

    def forward(self, x):
        B, T, C = x.shape #B, T, emb_dim
        q = self.proj_q(x) #B, T, n_heads * head_dim
        k = self.proj_k(x) #B, T, n_kv_groups * head_dim
        v = self.proj_v(x) #B, T, n_kv_groups * head_dim

        q = q.reshape(B, T, self.config.n_heads, self.config.head_dim) #B, T, n_heads, head_dim
        k = k.reshape(B, T, self.config.n_kv_groups, self.config.head_dim) #B, T, n_kv_groups, head_dim
        v = v.reshape(B, T, self.config.n_kv_groups, self.config.head_dim) #B, T, n_kv_groups, head_dim

        q = q.transpose(1, 2) #B, n_heads, T, head_dim
        k = k.transpose(1, 2) #B, n_kv_groups, T, head_dim
        v = v.transpose(1, 2) #B, n_kv_groups, T, head_dim

        repeat_factor = self.config.n_heads // self.config.n_kv_groups
        q = q.reshape(B, self.config.n_kv_groups, repeat_factor, T, self.config.head_dim) #B, n_kv_groups, repeat_factor, T, head_dim
        k = k.unsqueeze(2) #B, n_kv_groups, 1, T, head_dim
        v = v.unsqueeze(2) #B, n_kv_groups, 1, T, head_dim

        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        if self.layer_type == "sliding_attention":
            theta = self.config.rope_local_base
            offset = -self.config.sliding_window
            mask = torch.triu(torch.tril(torch.ones((T, T), device=x.device)), diagonal = offset) #Not registering a gull context length buffer for memory constraints' reasons
        elif self.layer_type == "full_attention":
            theta = self.config.rope_base
            mask = torch.tril(torch.ones((T, T), device=x.device)) #Not registering a gull context length buffer for memory constraints' reasons
        
        cos, sin = self.get_rope_params(self.config.head_dim, theta, T, dtype=q.dtype)
        cos, sin = cos.to(x.device), sin.to(x.device)
        
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        attn_scores = q.float() @ k.transpose(-1, -2).float() #B, n_kv_groups, repeat_factor, T, T
        attn_scores = attn_scores / math.sqrt(self.config.head_dim)
        
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = attn_scores.to(dtype=self.config.dtype) #going back to bf16 after accumulation in full precision

        ctx = attn_scores @ v #B, n_kv_groups, repeat_factor, T, head_dim
        ctx = ctx.reshape(B, self.config.n_heads, T, self.config.head_dim)
        ctx = ctx.transpose(1, 2) #B, T, n_heads, head_dim
        ctx = ctx.reshape(B, T, self.config.n_heads * self.config.head_dim) #B, T, n_heads*head_dim
        x = self.proj_o(ctx)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.emb_dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.emb_dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, config.emb_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x1 = self.w1(x)  # B, T, hidden_dim
        x2 = self.w2(x)  # B, T, hidden_dim
        x1 = F.gelu(x1)  # B, T, hidden_dim
        x = x1 * x2      # B, T, hidden_dim
        x = self.w3(x)   # B, T, emb_dim
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.attn = GroupedQueryAttention(config, layer_type)
        self.ffn = FeedForward(config)
        self.n1 = nn.RMSNorm(config.emb_dim, eps=1e-6)
        self.n2 = nn.RMSNorm(config.emb_dim, eps=1e-6)
        self.n3 = nn.RMSNorm(config.emb_dim, eps=1e-6)
        self.n4 = nn.RMSNorm(config.emb_dim, eps=1e-6)
    def forward(self, x):
        x = x + self.n2(self.attn(self.n1(x)))
        x = x + self.n4(self.ffn(self.n3(x)))
        return x

class TransformerModel(nn.Module):
    def __init__(self, config, layer_types):
        super().__init__()
        self.config = config
        self.layer_types = layer_types
        self.embedder = TransformerEmbedder(config)
        self.blocks = nn.ModuleList([TransformerBlock(config, layer_type) for layer_type in layer_types])
        self.n5 = nn.RMSNorm(config.emb_dim, eps=1e-6)
        self.proj_o = nn.Linear(config.emb_dim, config.vocab_size, bias=False) #can also be tied to the embeddinga
    def forward(self, idx, targets=None):
        idx = idx.to(device)
        x = self.embedder(idx)
        for block in self.blocks:
            x = block(x)
        x = self.n5(x)
        logits = self.proj_o(x)
        if targets is not None:
            targets = targets.to(device)
            assert idx.shape == targets.shape
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.float().view(B*T, C), targets.view(-1))
        else:
            loss = None
        return logits, loss
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length:]
            with torch.no_grad():
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :] #B, C
            probs = F.softmax(logits, dim=-1) #B, C
            idx_next = torch.multinomial(probs, num_samples=1) #B, 1
            idx = torch.cat((idx, idx_next), dim=1) #B, T+1
        self.train() 
        return idx

config = GEMMA3_270M_CONFIG()
layer_types = config.layer_type
m = TransformerModel(config, layer_types).to(device=device, dtype=config.dtype)

num_parameters = sum([p.numel() for p in m.parameters()])
print(f"total number of parameters: {num_parameters}, If we tie input and output projections: {num_parameters - m.embedder.embed_table.weight.numel()}")


