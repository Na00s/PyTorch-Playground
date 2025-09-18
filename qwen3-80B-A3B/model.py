import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class QWEN3_80B_A3B:
    attention_dropout: float = 0.0
    decoder_sparse_step: int = 1
    full_attention_interval: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 5120
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_value_head_dim: int = 128
    max_position_embeddings: int = 262144
    moe_intermediate_size: int = 512
    norm_topk_prob: bool = True
    num_attention_heads: int = 16
    num_experts: int = 512
    num_experts_per_tok: int = 10
    num_hidden_layers: int = 48
    num_key_value_heads: int = 2
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-06
    rope_theta: int = 10000000
    router_aux_loss_coef: float = 0.001
    shared_expert_intermediate_size: int = 512
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    use_cache: bool = False
    vocab_size: int = 151936

class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.config = config
        self.dim = dim
        self.gamma = nn.Parameter(torch.randn((dim))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    def forward(self, x):
        mean_x = torch.mean(x, dim=-1, keepdim=True) #B, T, 1
        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.config.rms_norm_eps)
        x = ((x - mean_x) / rms_x) * self.gamma
        return x


class TransformerEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.max_position_embeddings
        x = self.vocab_tok_embed(idx) #B, T, C
        return x

class GatedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj_q = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.proj_k = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.proj_o = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size)
        self.gate = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.norm_q = ZeroCenteredRMSNorm(config, config.head_dim)
        self.norm_k = ZeroCenteredRMSNorm(config, config.head_dim)
    @staticmethod
    def get_rope_params(head_dim, length, rope_angle, device):
        freq =  rope_angle ** (-(torch.arange(0, head_dim, 2, device=device)) / head_dim).unsqueeze(0) #1, head_dim//2
        length = torch.arange(0, length, device=device).unsqueeze(1) #T, 1
        angles = length * freq
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin
    @staticmethod
    def apply_rope(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        a = x_even * cos - x_odd * sin
        b = x_even * sin + x_odd * cos
        x = torch.stack([a, b], dim=-1)
        x = x.flatten(-2)
        return x
    def forward(self, x):
        B, T, C = x.shape

        q = self.proj_q(x) #B, T, num_attn_heads * head_dim
        k = self.proj_k(x) #B, T, num_kv_heads * head_dim
        v = self.proj_v(x) #B, T, num_kv_heads  * head_dim
        g = self.gate(x) #B, T, C

        g = torch.sigmoid(g)

        q = q.reshape(B, T, self.config.num_attention_heads, self.config.head_dim)
        k = k.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim)
        v = v.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim)

        q = q.transpose(1, 2) #B, num_attn_heads, T, head_dim
        k = k.transpose(1, 2) #B, num_kv_heads, T, head_dim
        v = v.transpose(1, 2) #B, num_kv_heads, T, head_dim

        q = self.norm_q(q)
        k = self.norm_k(k)

        cos, sin = self.get_rope_params(self.config.head_dim, T, self.config.rope_theta, device=x.device)
        q, k = self.apply_rope(q, cos, sin), self.apply_rope(k, cos, sin)

        repeat_factor = self.config.num_attention_heads // self.config.num_key_value_heads

        q = q.reshape(B, self.config.num_key_value_heads, repeat_factor, T, self.config.head_dim)
        k = k.reshape(B, self.config.num_key_value_heads, 1, T, self.config.head_dim)
        v = v.reshape(B, self.config.num_key_value_heads, 1, T, self.config.head_dim)

        scores = q @ k.transpose(-1, -2)
        scores = scores / math.sqrt(self.config.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        ctx = scores @ v #B, num_kv_heads, repeat_factor, T, head_dim

        ctx = ctx.reshape(B, self.config.num_attention_heads, T, self.config.head_dim)
        ctx = ctx.reshape(B, T, self.config.num_attention_heads, self.config.head_dim)
        ctx = ctx.reshape(B, T, self.config.num_attention_heads * self.config.head_dim)

        ctx = ctx * g 
        
        x = self.proj_o(ctx)

        return x

class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj_q = nn.Linear(config.hidden_size, config.linear_num_key_heads * config.linear_key_head_dim)
        self.proj_k = nn.Linear(config.hidden_size, config.linear_num_key_heads * config.linear_key_head_dim)
        self.proj_v = nn.Linear(config.hidden_size, config.linear_num_value_heads * config.linear_value_head_dim)
        self.conv_q = nn.Conv1d(in_channels=config.linear_key_head_dim, out_channels=config.linear_key_head_dim, kernel_size=config.linear_conv_kernel_dim, padding = config.linear_conv_kernel_dim//2)
        self.conv_k = nn.Conv1d(in_channels=config.linear_key_head_dim, out_channels=config.linear_key_head_dim, kernel_size=config.linear_conv_kernel_dim, padding = config.linear_conv_kernel_dim//2)
        self.conv_v = nn.Conv1d(in_channels=config.linear_value_head_dim, out_channels=config.linear_value_head_dim, kernel_size=config.linear_conv_kernel_dim, padding = config.linear_conv_kernel_dim//2)
        self.proj_alpha = nn.Linear(config.hidden_size, config.linear_num_value_heads * config.linear_value_head_dim)
        self.proj_beta = nn.Linear(config.hidden_size, config.linear_num_value_heads * config.linear_value_head_dim)
        self.gate = nn.Linear(config.hidden_size, config.linear_num_value_heads * config.linear_value_head_dim)
        self.norm_delta = ZeroCenteredRMSNorm(config, config.linear_value_head_dim)
        self.proj_o = nn.Linear(config.linear_num_value_heads * config.linear_value_head_dim, config.hidden_size)
    def forward(self, x):
        B, T, C = x.shape

        q = self.proj_q(x) #B, T, config.linear_num_key_heads * config.linear_key_head_dim
        k = self.proj_k(x) #B, T, config.linear_num_key_heads * config.linear_key_head_dim
        v = self.proj_v(x) #B, T, config.linear_num_value_heads * config.linear_value_head_dim

        q = q.reshape(B, T, self.config.linear_num_key_heads, self.config.linear_key_head_dim)
        k = k.reshape(B, T, self.config.linear_num_key_heads, self.config.linear_key_head_dim)
        v = v.reshape(B, T, self.config.linear_num_value_heads, self.config.linear_value_head_dim)

        q = q.permute(0, 2, 3, 1) #B, config.linear_num_key_heads, config.linear_key_head_dim, T
        k = k.permute(0, 2, 3, 1) #B, config.linear_num_key_heads, config.linear_key_head_dim, T
        v = v.permute(0, 2, 3, 1) #B, config.linear_num_value_heads, config.linear_value_head_dim, T

        q = q.reshape(B*self.config.linear_num_key_heads, self.config.linear_key_head_dim, T)
        k = k.reshape(B*self.config.linear_num_key_heads, self.config.linear_key_head_dim, T)
        v = v.reshape(B*self.config.linear_num_value_heads, self.config.linear_value_head_dim, T)

        q = self.conv_q(q) #B*self.config.linear_num_key_heads, self.config.linear_key_head_dim, T
        k = self.conv_k(k) #B*self.config.linear_num_key_heads, self.config.linear_key_head_dim, T
        v = self.conv_v(v) #B*self.config.linear_num_value_heads, self.config.linear_value_head_dim, T

        q = q.reshape(B, self.config.linear_num_key_heads, self.config.linear_key_head_dim, T).transpose(-1, -2) #B, self.config.linear_num_key_heads, T, self.config.linear_key_head_dim
        k = k.reshape(B, self.config.linear_num_key_heads, self.config.linear_key_head_dim, T).transpose(-1, -2) #B, self.config.linear_num_key_heads, T, self.config.linear_key_head_dim
        v = v.reshape(B, self.config.linear_num_value_heads, self.config.linear_value_head_dim, T).transpose(-1, -2) #B, self.config.linear_num_value_heads, T, self.config.linear_value_head_dim

        q = F.silu(q)
        k = F.silu(k)

        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

        v = F.silu(v)
        
        alpha = self.proj_alpha(x) #B, T, config.linear_num_value_heads * config.linear_value_head_dim
        beta = self.proj_beta(x) #B, T, config.linear_num_value_heads * config.linear_value_head_dim

        alpha = alpha.reshape(B, T, self.config.linear_num_value_heads, self.config.linear_value_head_dim).transpose(1, 2)
        beta = beta.reshape(B, T, self.config.linear_num_value_heads, self.config.linear_value_head_dim).transpose(1, 2)

        g = self.gate(x) #B, T, config.linear_num_value_heads * config.linear_value_head_dim

        g = g.reshape(B, T, self.config.linear_num_value_heads, self.config.linear_value_head_dim).transpose(1, 2) #B, config.linear_num_value_heads, T, config.linear_value_head_dim
        g = F.silu(g)

        repeat_factor = self.config.linear_num_value_heads // self.config.linear_num_key_heads

        alpha = alpha.reshape(B, self.config.linear_num_key_heads, repeat_factor, T, self.config.linear_value_head_dim)
        beta = beta.reshape(B, self.config.linear_num_key_heads, repeat_factor, T, self.config.linear_value_head_dim)
        g = g.reshape(B, self.config.linear_num_key_heads, repeat_factor, T, self.config.linear_value_head_dim)
        v = v.reshape(B, self.config.linear_num_key_heads, repeat_factor, T, self.config.linear_value_head_dim)

        q = q.unsqueeze(2)
        k = k.unsqueeze(2)
        

        delta = alpha * v + beta * (q-k)

        x = self.norm_delta(delta)

        x = g * x
        x = x.reshape(B, self.config.linear_num_value_heads, T, self.config.linear_value_head_dim)
        x = x.transpose(1, 2) #B, T, self.config_linear.num_value_heads, self.config.linear_value_head_dim
        x = x.reshape(B, T, self.config.linear_num_value_heads * self.config.linear_value_head_dim)
        x = self.proj_o(x)
        return x


class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.ModuleList([nn.Linear(config.hidden_size, config.intermediate_size) for _ in range(config.num_experts)])
        self.fc2 = nn.ModuleList([nn.Linear(config.hidden_size, config.intermediate_size) for _ in range(config.num_experts)])
        self.fc3 = nn.ModuleList([nn.Linear(config.intermediate_size, config.hidden_size) for _ in range(config.num_experts)])
        self.router = nn.Linear(config.hidden_size, config.num_experts)
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)
        
        router_logits = self.router(x) #B, T, num_experts
        
        gate_logits, gate_idxs = router_logits.topk(self.config.num_experts_per_tok, dim=-1) #(B*T, num_experts_per_tok)*2
        gate_probs = F.softmax(gate_logits, dim=-1) #B*T, num_experts_per_tok

        y_flat = torch.zeros_like(x_flat) #B*T, C
        gate_idxs_flat = gate_idxs.reshape(B*T*self.config.num_experts_per_tok) #B*T*config.num_experts_per_tok
        gate_probs_flat = gate_probs.reshape(B*T*self.config.num_experts_per_tok) #B*T*config.num_experts_per_tok
        token_indices = torch.arange(B*T, device=x.device).repeat_interleave(self.config.num_experts_per_tok) #B*T*config.num_experts_per_tok

        for expert_id in range(self.config.num_experts):
            mask = (gate_idxs_flat == expert_id)
            if not mask.any():
                continue
            expert_tokens = token_indices[mask] #num_selected_tokens_by_current_expert
            expert_weights = gate_probs_flat[mask] #num_selected_tokens_by_current_expert
            x_selected = x_flat[expert_tokens] #num_selected_tokens_by_current_expert, C
            u = self.fc1[expert_id](x_selected)
            v = self.fc2[expert_id](x_selected)
            act = F.silu(u) * v
            h = self.fc3[expert_id](act) #num_selected_tokens_by_current_expert, C
            y_flat.index_add_(0, expert_tokens, expert_weights.unsqueeze(1) * h)
        return y_flat.reshape(B, T, C)

class GatedDeltaNetBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm_1 = ZeroCenteredRMSNorm(config, config.hidden_size)
        self.gdn = GatedDeltaNet(config)
        self.norm_2 = ZeroCenteredRMSNorm(config, config.hidden_size)
        self.moe = MixtureOfExperts(config)
    def forward(self, x):
        x = x + self.gdn(self.norm_1(x))
        x = x + self.moe(self.norm_2(x))
        return x

class GatedAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm_1 = ZeroCenteredRMSNorm(config, config.hidden_size)
        self.ga = GatedAttention(config)
        self.norm_2 = ZeroCenteredRMSNorm(config, config.hidden_size)
        self.moe = MixtureOfExperts(config)
    def forward(self, x):
        x = x + self.ga(self.norm_1(x))
        x = x + self.moe(self.norm_2(x))
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = TransformerEmbedder(config)
        self.GdnBlocks = nn.ModuleList([GatedDeltaNetBlock(config) for _ in range(config.num_hidden_layers//2)])
        self.GaBlocks = nn.ModuleList([GatedAttentionBlock(config) for _ in range(config.num_hidden_layers//2)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    def forward(self, idx, targets=None):
        x = self.embedder(idx)
        for block in self.GdnBlocks:
            x = block(x)
        for block in self.GaBlocks:
            x = block(x)
        logits = self.lm_head(x)
        if targets is not None:
            assert targets.shape == idx.shape
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, C), targets.reshape(B*T))
        else:
            loss = None
        return logits, loss
    def generate(self, idx, max_new_tokens=500):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_position_embeddings:]
            with torch.no_grad():
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        self.train()
        return idx

config = QWEN3_80B_A3B()
m = Model(config).to(device=device, dtype=config.torch_dtype)
    




            




