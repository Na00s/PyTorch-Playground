import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal

@dataclass
class PixtralConfig:
    multimodal_projector_bias: bool = False
    projector_hidden_act: Literal["gelu"] = "gelu"
    attention_dropout: float = 0.0
    head_dim: int = 64
    hidden_act: Literal["silu"] = "silu"
    hidden_size: int = 1024
    image_size: int = 1540
    initializer_range: int = 0.02
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 24
    patch_size: int = 14
    rope_theta: float = 10000.0
    text_hidden_size: int = 5120

class VisionEmbedder(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.config = config
        self.p = config.patch_size
        in_dim = config.num_channels * self.p * self.p
        self.proj_in = nn.Linear(in_dim, config.hidden_size)
        nn.init.normal_(self.proj_in.weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.proj_in.bias)
    def forward(self, imgs):
        assert imgs.ndim == 4
        B, C, H, W = imgs.shape
        assert C == self.config.num_channels
        Hc = H // 2
        Wc = W // 2
        Hc = (Hc // self.p) * self.p
        Wc = (Wc // self.p) * self.p
        imgs = imgs[:, :, :Hc, :Wc]
        imgs = imgs.reshape(B, C, Hc//self.p, self.p, Wc//self.p, self.p)
        imgs = imgs.permute(0, 2, 4, 1, 3, 5) #B, Hc//self.p, Wc//self.p, C, self.p, self.p
        imgs = imgs.reshape(B, Hc//self.p * Wc//self.p, C*self.p*self.p)
        x = self.proj_in(imgs)
        return x

class VisionMultiHeadSelfAttention(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.config = config
        self.proj_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_o = nn.Linear(config.hidden_size, config.hidden_size)
    @staticmethod
    def get_rope_params(head_dim, length, theta, device):
        wavelength = 1 / (theta ** (torch.arange(0, head_dim, 2, device=device) / head_dim)).unsqueeze(0)
        positions = torch.arange(0, length, device=device).unsqueeze(1)
        angles = wavelength * positions
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin
    @staticmethod
    def apply_rope(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        a = (x_even * cos) - (x_odd * sin)
        b = (x_even * sin) + (x_odd * cos)
        x = torch.stack([a, b], dim=-1)
        x = x.flatten(-2)
        return x
    def forward(self, x):
        B, T, C = x.shape
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        q = q.reshape(B, T, self.config.num_attention_heads, self.config.head_dim)
        k = k.reshape(B, T, self.config.num_attention_heads, self.config.head_dim)
        v = v.reshape(B, T, self.config.num_attention_heads, self.config.head_dim)

        q = q.transpose(1, 2) #B, n_heads, T, head_size
        k = k.transpose(1, 2) #B, n_heads, T, head_size
        v = v.transpose(1, 2) #B, n_heads, T, head_size

        cos, sin = self.get_rope_params(self.config.head_dim, T, self.config.rope_theta, x.device)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        scores = q @ k.transpose(-1, -2)
        scores = scores / math.sqrt(self.config.head_dim)
        scores = F.softmax(scores, dim=-1)

        ctx = scores @ v
        ctx = ctx.transpose(1, 2) #B, T, n_heads, head_size
        ctx = ctx.reshape(B, T, C)

        x = self.proj_o(ctx)
        return x

class VisionFeedForwardNetwork(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc3 = nn.Linear(config.intermediate_size, config.hidden_size)
    def forward(self, x):
        x = self.fc1(x) * F.silu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class VisionBlock(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.config = config
        self.n1 = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.n2 = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.vmha = VisionMultiHeadSelfAttention(config)
        self.vffn = VisionFeedForwardNetwork(config)
    def forward(self, x):
        x = x + self.n1(self.vmha(x))
        x = x + self.n2(self.vffn(x))
        return x

class VisionEncoder(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.config = config
        self.embedder = VisionEmbedder(config)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.projector = nn.Linear(config.hidden_size, config.text_hidden_size, bias=False)
    def forward(self, imgs):
        x = self.embedder(imgs)
        for block in self.blocks:
            x = block(x)
        logits = self.projector(x)
        return F.gelu(logits)




        







        