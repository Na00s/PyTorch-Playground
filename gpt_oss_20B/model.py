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
        self.tok_word_embed = nn.Embedding(config.vocab_size, config.hidden_size)
    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.max_position_embeddings
        x = self.tok_word_embed(idx)
        return x

class GroupedQueryAttention(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.proj_q = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.proj_k = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.proj_o = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        # could register mask as a buffer here but won't do it for memory reasons

    @staticmethod
    def get_rope_params(theta, head_dim, length, device):
        wavelength = theta ** (torch.arange(0, head_dim, 2, device=device) / head_dim).unsqueeze(0) #1, head_dim//2
        freqs = 1 / wavelength
        positions = torch.arange(length, device=device).unsqueeze(1) #T, 1
        angles = positions @ freqs #T, head_dim//2
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin

    @staticmethod
    def apply_rope(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        a = x_even * cos - x_odd * sin
        b = x_even * sin + x_odd * cos
        x = torch.stack([a, b], dim=-1) #B, T, C//2, 2
        x = x.flatten(-2) #B, T, C
        return x

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.config.hidden_size
        q = self.proj_q(x) #B, T, num_attention_heads * head_dim
        k = self.proj_k(x) #B, T, num_kv_heads * head_dim
        v = self.proj_v(x) #B, T, num_kv_heads * head_dim

        q = q.reshape(B, T, self.config.num_attention_heads, self.config.head_dim)
        k = k.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim)
        v = v.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim)

        q = q.transpose(1, 2) #B, n_attn_heads, T, head_dim
        k = k.transpose(1, 2) #B, n_kv_heads, T, head_dim
        v = v.transpose(1, 2) #B, n_kv_heads, T, head_dim

        cos, sin = self.get_rope_params(theta=self.config.rope_theta, head_dim=self.config.head_dim, length=q.size(3), device=x.device)
        q = self.apply_rope(q, cos, sin).to(dtype=x.dtype)
        k = self.apply_rope(k, cos, sin).to(dtype=x.dtype)

        repeat_factor = self.config.num_attention_heads // self.config.num_key_value_heads

        q = q.reshape(B, self.config.num_key_value_heads, repeat_factor, T, self.config.head_dim) #B, n_kv_heads, repeat_factor, T, head_dim
        k = k.unsqueeze(2) #B, n_kv_heads, 1, T, head_dim
        v = v.unsqueeze(2) #B, n_kv_heads, 1, T, head_dim

        scores = q @ k.transpose(-1, -2) #B, n_kv_heads, repeat_factor, T, T
        scores = scores / math.sqrt(self.config.head_dim)
        
        if self.layer_type == "full_attention":
            mask = torch.tril(torch.ones(T, T, device=x.device))
        if self.layer_type == "sliding_attention":
            mask = torch.triu(torch.tril(torch.ones(T, T, device=x.device)), diagonal=-self.config.sliding_window)
        
        scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(scores, dim=-1)
        ctx = scores @ v #B, n_kv_heads, repeat_factor, T, head_dim

        ctx = ctx.reshape(B, self.config.num_attention_heads, T, self.config.head_dim)
        ctx = ctx.transpose(1, 2)
        ctx = ctx.reshape(B, T, self.config.num_attention_heads*self.config.head_dim)

        x = self.proj_o(ctx)
        return x
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.fc1 = nn.ModuleList([nn.Linear(config.hidden_size, config.intermediate_size) for _ in range(config.num_local_experts)])
        self.fc2 = nn.ModuleList([nn.Linear(config.hidden_size, config.intermediate_size) for _ in range(config.num_local_experts)])
        self.fc3 = nn.ModuleList([nn.Linear(config.intermediate_size, config.hidden_size) for _ in range(config.num_local_experts)])
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)

        router_logits = self.router(x_flat) #B*T, num_local_experts
        gate_logits, gate_idxs = router_logits.topk(self.config.experts_per_token, dim=-1) #(B*T, experts_per_token)*2
        gate_probs = F.softmax(gate_logits, dim=-1) #B*T, experts_per_token

        if self.training:
            router_probs = F.softmax(router_logits, dim=-1) #B*T, num_local_experts
            flat_ids = gate_idxs.reshape(-1) #B*T*experts_per_token
            counts = torch.zeros(self.config.num_local_experts, device=x_flat.device, dtype=x_flat.dtype) #(num_local_experts, )
            counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=x_flat.dtype)) #(num_local_experts, )
            fraction_per_expert = counts / (B * T * self.config.experts_per_token) #(num_local_experts, )
            mean_prob_per_expert = router_probs.mean(dim=0) #(num_local_experts, )
            self.aux_loss = self.config.num_local_experts * (mean_prob_per_expert * fraction_per_expert).sum() #Scalar Tensor
        else:
            self.aux_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
        
        y_flat = torch.zeros_like(x_flat) #B*T, C
        flat_gate_idxs = gate_idxs.reshape(-1) #B*T*experts_per_token
        flat_gate_probs = gate_probs.reshape(-1) #B*T*experts_per_token
        token_indices = torch.arange(B*T, device=x_flat.device).repeat_interleave(self.config.experts_per_token) #B*T*experts_per_token

        for expert_id in range(self.config.num_local_experts):
            mask = (flat_gate_idxs == expert_id) #B*T*experts_per_token
            if not mask.any():
                continue
            expert_tokens = token_indices[mask] #num_selected_tokens
            expert_weights = flat_gate_probs[mask] #num_selected_tokens

            x_sel = x_flat[expert_tokens] #num_selected_tokens, C
            
            u = self.fc1[expert_id](x_sel) #num_selected_tokens, intermediate_size
            v = self.fc2[expert_id](x_sel) #num_selected_tokens, intermediate_size
            act = F.silu(u) * v #num_selected_tokens, intermediate_size
            if self.config.swiglu_limit is not None:
                act = torch.clamp(act, -self.config.swiglu_limit, self.config.swiglu_limit)
            h = self.fc3[expert_id](act) #num_selected_tokens, C

            y_flat.index_add_(0, expert_tokens, expert_weights.unsqueeze(1) * h) #num_selected_tokens, C elemets gets populated in the B*T, C buffer
        
        return y_flat.view(B, T, C)




class TransformerBlock(nn.Module):
    def __init__(self, config, layer_types):
        super().__init__()
        self.config = config
        self.attn = GroupedQueryAttention(config, layer_types)
        self.ffn = FeedForwardNetwork(config)
        self.n1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.n2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = TransformerEmbedder(config)
        self.blocks = nn.ModuleList([TransformerBlock(config, layer_type) for layer_type in config.layer_types])
        self.n = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj_o = nn.Linear(config.hidden_size, config.vocab_size)
    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.proj_o(x)
        if targets is not None:
            B, T, C = logits.shape
            assert (targets.shape == (B, T))
            loss = F.cross_entropy(logits.reshape(B*T, C), targets.reshape(-1))
        else:
            loss = None
        return logits, loss
    def generate(self, idx, max_new_tokens = 500):
        B, T = idx.shape
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.config.max_position_embeddings:]
            logits, _ = self(idx_context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx 


config = GPT_OSS_20B() 
m = TransformerModel(config)       

num_params = sum([p.numel() for p in m.parameters()])
print(num_params)
        



