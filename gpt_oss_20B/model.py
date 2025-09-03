import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class GPT_OSS_20B:
    attention_bias: bool = True
    attention_dropout: float = 0.0
    eos_token_id: int = 200002
    experts_per_token: int = 4
    head_dim: int = 64
    hidden_act: Literal["gelu", "silu", "relu", "swiglu", "glu", "geglu"] = "silu"
    hidden_size: int = 2880
    initial_context_length: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 2880
    layer_types: Tuple[str, ...] = ("sliding_attention", 
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention")
    max_position_embeddings: int = 131072
    num_attention_heads: int = 64
    num_experts_per_tok: int = 4
    num_hidden_layers: int = 24
    num_key_value_heads: int = 8
    num_local_experts: int = 32
    output_router_logits: bool = False
    pad_token_id: int = 199999
    modules_to_not_convert: Tuple[str, ...] = ("model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head")
    quant_method: str = "mxfp4"
    rms_norm_eps: float = 1e-05
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_factor: float = 32.0
    original_max_position_embeddings: int = 4096
    rope_theta: int = 150000
    router_aux_loss_coef: float = 0.9
    sliding_window: int = 128
    swiglu_limit: float = 7.0
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 201088

class TransformerEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wrd_tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
    def forward(self, idx):
        B, T = idx.shape
        assert (T <= self.config.max_position_embeddings)
        x = self.wrd_tok_embed(idx) #B, T, hidden_size
        return x

class GroupedQueryAttention(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.proj_q = nn.Linear(config.hidden_size, config.num_attention_heads*config.head_dim, bias=False)
        self.proj_k = nn.Linear(config.hidden_size, config.num_key_value_heads*config.head_dim, bias=False)
        self.proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads*config.head_dim, bias=False)
        self.proj_o = nn.Linear(config.num_attention_heads*config.head_dim, config.hidden_size)
    @staticmethod
    def get_rope_params(theta, length, head_dim, device):
        freqs = theta ** (-torch.arange(0, head_dim, 2) / head_dim).unsqueeze(0) #1, D//2
        freqs = freqs.to(device)
        positions = torch.arange(length).unsqueeze(1) #T, 1
        positions = positions.to(device)
        angles = positions * freqs #T, D//2
        cos = torch.cos(angles) #T, D//2
        sin = torch.sin(angles) #T, D//2
        return cos, sin
    @staticmethod
    def apply_rope(x, cos, sin):
        x_even = x[..., 0::2] #B, T, C//2
        x_odd = x[..., 1::2] #B, T, C//2
        u = x_even*cos + x_odd*sin #B, T, C//2
        v = -x_even*sin + x_odd*cos #B, T, C//2
        x_rotated = torch.stack([u, v], dim=-1) #B, T, C//2, 2
        x_rotated = x_rotated.flatten(-2) #B, T, C
        return x_rotated
    def forward(self, x):
        B, T, C = x.shape
        q = self.proj_q(x) #B, T, num_attention_heads * head_dim
        k = self.proj_k(x) #B, T, num_key_value_heads * head_dim
        v = self.proj_v(x) #B, T, num_key_value_heads * head_dim

        q = q.reshape(B, T, self.config.num_attention_heads, self.config.head_dim) #B, T, num_attention_heads, head_dim
        k = k.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim) #B, T, num_key_value_heads, head_dim
        v = v.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim) #B, T, num_key_value_heads, head_dim

        q = q.transpose(1, 2) #B, num_attention_heads, T, head_dim
        k = k.transpose(1, 2) #B, num_key_value_heads, T, head_dim
        v = v.transpose(1, 2) #B, num_key_value_heads, T, head_dim

        cos, sin = self.get_rope_params(self.config.rope_theta, q.size(2), self.config.head_dim, device=x.device)
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        repeat_factor = self.config.num_attention_heads // self.config.num_key_value_heads

        q = q.reshape(B , self.config.num_key_value_heads, repeat_factor, T, self.config.head_dim)
        k = k.reshape(B , self.config.num_key_value_heads, 1, T, self.config.head_dim)
        v = v.reshape(B , self.config.num_key_value_heads, 1, T, self.config.head_dim)

        if self.layer_type == "sliding_attention":
            mask = torch.triu(torch.tril(torch.ones(T, T, device=x.device)), diagonal=-self.config.sliding_window)
        else:
            mask = torch.tril(torch.ones(T, T, device=x.device))
        
        scores = q @ k.transpose(-1, -2) #B, num_key_value_heads, repeat_factor, T, T
        scores = scores / math.sqrt(self.config.head_dim)
        scores = scores.masked_fill(mask==0, float("-inf"))
        scores = F.softmax(scores, dim=-1)

        ctx = scores @ v #B, num_key_value_heads, repeat_factor, T, head_dim
        ctx = ctx.reshape(B, self.config.num_attention_heads, T, self.config.head_dim)
        ctx = ctx.transpose(1, 2)
        ctx = ctx.reshape(B, T, self.config.num_attention_heads*self.config.head_dim)

        x = self.proj_o(ctx)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_local_experts)
        self.fc1 = nn.ModuleList([nn.Linear(config.hidden_size, config.intermediate_size) for _ in range(config.num_local_experts)])
        self.fc2 = nn.ModuleList([nn.Linear(config.hidden_size, config.intermediate_size) for _ in range(config.num_local_experts)])
        self.fc3 = nn.ModuleList([nn.Linear(config.intermediate_size, config.hidden_size) for _ in range(config.num_local_experts)])
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)

        router_logits = self.router(x_flat) #B*T, num_local_experts
        gate_logits, gate_idxs = router_logits.topk(self.config.experts_per_token, dim=-1) #(B*T, num_local_experts)*2
        gate_probs = F.softmax(gate_logits, dim=-1)

        y_flat = torch.zeros_like(x_flat) #B*T, C
        gate_idxs_flat = gate_idxs.reshape(B*T*self.config.experts_per_token) #B*T*experts_per_token
        gate_probs_flat = gate_probs.reshape(B*T*self.config.experts_per_token) #B*T*experts_per_token
        token_indices = torch.arange(B*T, device=x.device).repeat_interleave(self.config.experts_per_token)

        for expert_id in range(self.config.num_local_experts):
            mask = (gate_idxs_flat == expert_id) #B*T*experts_per_token
            if not mask.any():
                continue
            expert_tokens = token_indices[mask] #num_selected_tokens
            expert_weights = gate_probs_flat[mask] #num_selected_tokens
            x_selected = x_flat[expert_tokens] #num_selected_tokens, C
            u = self.fc1[expert_id](x_selected) #num_selected_tokens, intermediate_size
            v = self.fc2[expert_id](x_selected) #num_selected_tokens, intermediate_size
            act = F.silu(u) * v
            if self.config.swiglu_limit is not None:
                act = torch.clamp(act, -self.config.swiglu_limit, self.config.swiglu_limit)
            h = self.fc3[expert_id](act) #num_selected_tokens, hidden_size
            y_flat.index_add_(0, expert_tokens, expert_weights.unsqueeze(1) * h)
        return y_flat.reshape(B, T, C)










