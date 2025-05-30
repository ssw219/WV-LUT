"""
Global Adjustment Module (GAM) implementation based on IAT (Illumination-Adaptive Transformer).
Reference: https://github.com/cuiziteng/Illumination-Adaptive-Transformer
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from typing import Optional, Tuple, Union


class Mlp(nn.Module):
    """MLP with GELU activation and dropout."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AffChannel(nn.Module):
    """Affine transformation layer for channel-wise operations."""
    
    def __init__(self, dim: int, channel_first: bool = True) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_first:
            x1 = torch.tensordot(x, self.color, dims=[[-1], [-1]])
            x2 = x1 * self.alpha + self.beta
        else:
            x1 = x * self.alpha + self.beta
            x2 = torch.tensordot(x1, self.color, dims=[[-1], [-1]])
        return x2


class QueryAttention(nn.Module):
    """Attention mechanism with learnable query parameters."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.q = nn.Parameter(torch.ones((1, 10, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 10, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QuerySABlock(nn.Module):
    """Self-attention block with query mechanism."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = QueryAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvEmbedding(nn.Module):
    """Convolutional embedding layer."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class BaseGAM(nn.Module):
    """Base class for GAM modules."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        num_heads: int = 4,
        num_chunks: int = 2
    ) -> None:
        super().__init__()
        self.num_chunks = num_chunks
        self.gamma_base = nn.Parameter(torch.ones(1), requires_grad=True)
        self.color_base = nn.Parameter(torch.eye(3), requires_grad=True)
        
        self.conv_large = ConvEmbedding(in_channels, out_channels)
        self.generator = QuerySABlock(dim=out_channels // num_chunks, num_heads=num_heads)
        self.gamma_linear = nn.Linear(out_channels, 1)
        self.color_linear = nn.Linear(out_channels, 1)
        self.norm = nn.LayerNorm(out_channels)
        
        self.apply(self._init_weights)
        self._init_special_weights()

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_special_weights(self) -> None:
        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_large(x)
        chunks = torch.chunk(x, self.num_chunks, dim=1)
        x_generators = [self.generator(chunk) for chunk in chunks]
        x = torch.cat(x_generators, dim=2)
        x = self.norm(x)
        
        gamma, color = x[:, 0].unsqueeze(1), x[:, 1:]
        gamma = self.gamma_linear(gamma).squeeze(-1) + self.gamma_base
        color = self.color_linear(color).squeeze(-1).view(-1, 3, 3) + self.color_base
        return gamma, color


class GAM_pll_2(BaseGAM):
    """GAM with 2 parallel branches."""
    def __init__(self, in_channels: int = 3, out_channels: int = 64, num_heads: int = 4):
        super().__init__(in_channels=in_channels, out_channels=out_channels, num_heads=num_heads, num_chunks=2)


class GAM_pll_4(BaseGAM):
    """GAM with 4 parallel branches."""
    def __init__(self, in_channels: int = 3, out_channels: int = 64, num_heads: int = 4):
        super().__init__(in_channels=in_channels, out_channels=out_channels, num_heads=num_heads, num_chunks=4)


class GAM_pll_8(BaseGAM):
    """GAM with 8 parallel branches."""
    def __init__(self, in_channels: int = 3, out_channels: int = 64, num_heads: int = 4):
        super().__init__(in_channels=in_channels, out_channels=out_channels, num_heads=num_heads, num_chunks=8)
