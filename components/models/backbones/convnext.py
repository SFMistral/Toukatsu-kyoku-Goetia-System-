# -*- coding: utf-8 -*-
"""
ConvNeXt骨干网络

支持ConvNeXt-Tiny/Small/Base/Large/XLarge。
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn

from registry.model_registry import BACKBONES
from ..layers import DropPath, LayerNorm2d


CONVNEXT_CONFIGS = {
    'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
    'small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768]},
    'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
    'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
    'xlarge': {'depths': [3, 3, 27, 3], 'dims': [256, 512, 1024, 2048]},
}


class ConvNeXtBlock(nn.Module):
    """ConvNeXt块"""
    
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = shortcut + self.drop_path(x)
        return x


@BACKBONES.register(name='ConvNeXt')
class ConvNeXt(nn.Module):
    """
    ConvNeXt骨干网络
    
    Args:
        arch: 架构变体 tiny/small/base/large/xlarge
        in_channels: 输入通道数
        out_indices: 输出stage索引
        drop_path_rate: DropPath概率
        layer_scale_init_value: LayerScale初始值
    """
    
    def __init__(
        self,
        arch: str = 'tiny',
        in_channels: int = 3,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        config = CONVNEXT_CONFIGS.get(arch, CONVNEXT_CONFIGS['tiny'])
        depths = config['depths']
        dims = config['dims']
        
        self.out_indices = out_indices
        self.out_channels = dims
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 4, stride=4),
            LayerNorm2d(dims[0]),
        )
        
        # Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            # Downsample
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i-1]),
                    nn.Conv2d(dims[i-1], dims[i], 2, stride=2),
                )
            else:
                downsample = nn.Identity()
                
            # Blocks
            blocks = [
                ConvNeXtBlock(
                    dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                ) for j in range(depths[i])
            ]
            cur += depths[i]
            
            stage = nn.Sequential(downsample, *blocks)
            self.stages.append(stage)
            
        # Norms for output
        self.norms = nn.ModuleList([
            LayerNorm2d(dims[i]) for i in out_indices
        ])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                idx = self.out_indices.index(i)
                outs.append(self.norms[idx](x))
                
        return outs
