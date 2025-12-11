# -*- coding: utf-8 -*-
"""
Swin Transformer骨干网络

支持Swin-Tiny/Small/Base/Large。
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import BACKBONES
from ..layers import DropPath, WindowAttention, MLP


SWIN_CONFIGS = {
    'tiny': {'embed_dim': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24]},
    'small': {'embed_dim': 96, 'depths': [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24]},
    'base': {'embed_dim': 128, 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32]},
    'large': {'embed_dim': 192, 'depths': [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48]},
}


class PatchMerging(nn.Module):
    """Patch合并层（下采样）"""
    
    def __init__(self, dim: int, norm_cfg: Dict[str, Any] = dict(type='LN')):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        # 下采样
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        
        return x, H // 2, W // 2


class SwinBlock(nn.Module):
    """Swin Transformer块"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (window_size, window_size), num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=drop)
        
    def forward(self, x: torch.Tensor, H: int, W: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
        # Window partition
        x = self._window_partition(x)
        x = x.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        x = self.attn(x, mask)
        
        # Window reverse
        x = x.view(-1, self.window_size, self.window_size, C)
        x = self._window_reverse(x, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
        
    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size, self.window_size, C)
        return x
        
    def _window_reverse(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = x.shape[0] // (H // self.window_size * W // self.window_size)
        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x


class SwinStage(nn.Module):
    """Swin Transformer Stage"""
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: List[float] = None,
        downsample: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim, num_heads, window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path[i] if drop_path else 0.0,
            ) for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        for block in self.blocks:
            x = block(x, H, W)
        if self.downsample:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


@BACKBONES.register(name='SwinTransformer')
class SwinTransformer(nn.Module):
    """Swin Transformer骨干网络"""
    
    def __init__(
        self,
        arch: str = 'tiny',
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        config = SWIN_CONFIGS.get(arch, SWIN_CONFIGS['tiny'])
        embed_dim = config['embed_dim']
        depths = config['depths']
        num_heads = config['num_heads']
        
        self.out_indices = out_indices
        self.num_stages = len(depths)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        self.out_channels = []
        
        for i in range(self.num_stages):
            stage = SwinStage(
                dim=embed_dim * (2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                downsample=i < self.num_stages - 1,
            )
            self.stages.append(stage)
            self.out_channels.append(embed_dim * (2 ** i))
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.pos_drop(x)
        
        outs = []
        for i, stage in enumerate(self.stages):
            x, H, W = stage(x, H, W)
            if i in self.out_indices:
                out = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs
