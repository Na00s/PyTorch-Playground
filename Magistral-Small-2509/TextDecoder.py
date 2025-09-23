import math
import torch

import torch.nn as nn
import torch.nn.functional as F


from dataclasses import dataclass
from typing import Optional, Tuple, Literal, List
from VisionEncoder import VisionEncoder, PixtralConfig

vis_cfg = PixtralConfig()

@dataclass
class MagistralConfig:
    image_token_index: int = 10
    attention_dropout: float = 0.0
    head_dim: int = 128
    hidden_act: Literal["silu"] = "silu"
    hidden_size: int = 5120
    initializer_range: float = 0.02
    intermediate_size: int = 32768
    max_position_embeddings: int = 131072
    num_attention_heads: int = 32
    num_hidden_layers: int = 40
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-05
    rope_theta: float = 1000000000.0
    use_cache: bool = True
    vocab_size: int = 131072

class MagistralEmbedder(nn.Module):
    def __init__(self, config: MagistralConfig):
        super().__init__()
        self.config = config
        self.VisionEncoder = VisionEncoder(vis_cfg)
        self.text_embedder = nn.Embedding(config.vocab_size, config.hidden_size)

        nn.init.normal_(self.text_embedder.weight, mean=0.0, std=0.2)
    def forward(self, idx, imgs=None):
        B, T = idx.shape
        assert T <= self.config.max_position_embeddings
        x = self.text_embedder(idx)
        if imgs is not None:
            assert imgs.shape[0] == B
            vision_projection = self.VisionEncoder(imgs)
            x = torch.cat([x[:, :self.config.image_token_index, :], vision_projection, x[:, self.config.image_token_index:, :]], dim=1)
        return x

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: MagistralConfig):
        super().__init__()
        self.config = config
        self.proj_q = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.proj_k = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.proj_o = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size)

    @staticmethod
    def get_rope_params(head_dim, length, theta, device, start=0):
        frequencies = 1 / (theta ** torch.arange(0, head_dim, 2, device=device) / head_dim).unsqueeze(0)
        positions = torch.arange(start, start + length, device = device).unsqueeze(1)
        angles = frequencies * positions
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin
    
    @staticmethod
    def apply_rope(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        a = (x_even*cos) - (x_odd*sin)
        b = (x_even*sin) + (x_odd*cos)
        x = torch.stack([a, b], dim=-1)
        x = x.flatten(-2)
        return x
        
    def forward(self, x, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.shape
        q = self.proj_q(x) #B, T, num_attention_heads * head_dim
        k_new = self.proj_k(x) #B, T, num_key_value_heads * head_dim
        v_new = self.proj_v(x) #B, T, num_key_value_heads * head_dim

        q = q.reshape(B, T, self.config.num_attention_heads, self.config.head_dim)
        k_new = k_new.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim)
        v_new = v_new.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim)
        
        q = q.permute(0, 2, 1, 3) #B, num_attention_heads, T, head_dim
        k_new = k_new.permute(0, 2, 1, 3) #B, num_key_value_heads, T, head_dim
        v_new = v_new.permute(0, 2, 1, 3) #B, num_key_value_heads, T, head_dim

        if kv_cache is None:
            L = 0
            k_prev = v_prev = None
        else:
            k_prev, v_prev = kv_cache #(B, num_key_value_heads, L, head_dim)*2
            L = k_prev.shape[2]

        cos, sin = self.get_rope_params(self.config.head_dim, T, self.config.rope_theta, device=x.device, start=L)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        q = self.apply_rope(q, cos, sin)
        k_new = self.apply_rope(k_new, cos, sin)

        if L > 0:
            k = torch.cat([k_prev, k_new], dim=2)
            v = torch.cat([v_prev, v_new], dim=2)
        else:
            k, v = k_new, v_new

        repeat_factor = self.config.num_attention_heads // self.config.num_key_value_heads

        q = q.reshape(B, self.config.num_key_value_heads, repeat_factor, T, self.config.head_dim)
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)

        mask = torch.tril(torch.ones(T, L + T, device=x.device), diagonal=L)

        scores = q @ k.transpose(-1, -2)
        scores = scores / math.sqrt(self.config.head_dim)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        scores = F.softmax(scores, dim=-1)
        ctx = scores @ v #B, config.num_key_value_heads, repeat_factor, T, config.head_dim

        ctx = ctx.reshape(B, self.config.num_attention_heads, T, self.config.head_dim)
        ctx = ctx.permute(0, 2, 1, 3)
        ctx = ctx.reshape(B, T, self.config.num_attention_heads*self.config.head_dim)

        x = self.proj_o(ctx)
        if self.config.use_cache:
            return x, (k.squeeze(2), v.squeeze(2))
        else:
            return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, config: MagistralConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc3 = nn.Linear(config.intermediate_size, config.hidden_size)
    def forward(self, x):
        x = self.fc1(x) * F.silu(self.fc2(x))
        return self.fc3(x)

class TransformerBlock(nn.Module):
    def __init__(self, config: MagistralConfig):
        super().__init__()
        self.config = config
        self.gqa = GroupedQueryAttention(config)
        self.ffn = FeedForwardNetwork(config)
        self.n1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.n2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if kv_cache is None:
            attn_out = self.gqa(x)                  
            x = x + self.n1(attn_out)
            x = x + self.n2(self.ffn(x))
            return x, None
        else:
            attn_out, new_cache = self.gqa(x, kv_cache=kv_cache) 
            x = x + self.n1(attn_out)
            x = x + self.n2(self.ffn(x))
            return x, new_cache

class Magistral(nn.Module):
    def __init__(self, config: MagistralConfig):
        super().__init__()
        self.config = config
        self.embedder = MagistralEmbedder(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.cache = None

    def forward(self, x, kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        x = self.embedder(x)
        new_caches = []
        for i, block in enumerate(self.blocks):
            if kv_caches is None:
                x, cache = block(x)
            else:
                x, cache = block(x, kv_cache=kv_caches[i])
            new_caches.append(cache)
        logits = self.lm_head(x)
        if self.config.use_cache:
            return logits, new_caches
        else:
            return logits

    def clear_cache(self):
        self.cache = None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature, top_k):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_position_embeddings:]
            if self.cache is None:
                logits, self.cache = self.forward(idx_cond)
            else:
                logits, self.cache = self.forward(idx_cond[:, -1:], kv_caches=self.cache)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(values, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(indices, 1, sampled)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx




        
    




        