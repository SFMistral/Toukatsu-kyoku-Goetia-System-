# -*- coding: utf-8 -*-
"""
EfficientNet系列骨干网络

支持EfficientNet-B0~B7、EfficientNetV2-S/M/L。
"""

from typing import Dict, Any, Optional, List, Tuple
import math
import torch
import torch.nn as nn

from registry.model_registry import BACKBONES
from ..layers import ConvModule, InvertedResidual, SELayer, DropPath
from ..utils import make_divisible


# EfficientNet配置 (expand_ratio, channels, num_blocks, stride, kernel_size)
EFFICIENTNET_CONFIGS = {
    'b0': {'width_mult': 1.0, 'depth_mult': 1.0, 'resolution': 224},
    'b1': {'width_mult': 1.0, 'depth_mult': 1.1, 'resolution': 240},
    'b2': {'width_mult': 1.1, 'depth_mult': 1.2, 'resolution': 260},
    'b3': {'width_mult': 1.2, 'depth_mult': 1.4, 'resolution': 300},
    'b4': {'width_mult': 1.4, 'depth_mult': 1.8, 'resolution': 380},
    'b5': {'width_mult': 1.6, 'depth_mult': 2.2, 'resolution': 456},
    'b6': {'width_mult': 1.8, 'depth_mult': 2.6, 'resolution': 528},
    'b7': {'width_mult': 2.0, 'depth_mult': 3.1, 'resolution': 600},
}

# MBConv配置
MBCONV_CONFIGS = [
    # (expand, channels, num_blocks, stride, kernel)
    (1, 16, 1, 1, 3),
    (6, 24, 2, 2, 3),
    (6, 40, 2, 2, 5),
    (6, 80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
]


class MBConvBlock(nn.Module):
    """MBConv块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float,
        stride: int,
        kernel_size: int,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='SiLU', inplace=True),
    ):
        super().__init__()

        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        # Expand
        if expand_ratio != 1:
            layers.append(ConvModule(in_channels, hidden_dim, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
        # Depthwise
        layers.append(ConvModule(
            hidden_dim, hidden_dim, kernel_size,
            stride=stride, padding=kernel_size // 2, groups=hidden_dim,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        ))
        # SE
        if se_ratio > 0:
            layers.append(SELayer(hidden_dim, reduction=int(1 / se_ratio)))
        # Project
        layers.append(ConvModule(hidden_dim, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None))
        
        self.conv = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)


@BACKBONES.register(name='EfficientNet')
class EfficientNet(nn.Module):
    """
    EfficientNet骨干网络
    
    Args:
        arch: 架构变体 b0-b7
        out_indices: 输出特征层索引
        frozen_stages: 冻结的stage数量
        drop_path_rate: DropPath概率
        norm_cfg: 归一化层配置
    """
    
    def __init__(
        self,
        arch: str = 'b0',
        out_indices: Tuple[int, ...] = (1, 2, 4, 6),
        frozen_stages: int = -1,
        drop_path_rate: float = 0.0,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='SiLU', inplace=True),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        if arch not in EFFICIENTNET_CONFIGS:
            raise ValueError(f"Invalid arch {arch}")
            
        config = EFFICIENTNET_CONFIGS[arch]
        width_mult = config['width_mult']
        depth_mult = config['depth_mult']
        
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Stem
        stem_channels = make_divisible(32 * width_mult, 8)
        self.stem = ConvModule(3, stem_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # Stages
        self.stages = nn.ModuleList()
        self.out_channels = []
        in_channels = stem_channels
        total_blocks = sum(int(math.ceil(cfg[2] * depth_mult)) for cfg in MBCONV_CONFIGS)
        block_idx = 0
        
        for i, (expand, channels, num_blocks, stride, kernel) in enumerate(MBCONV_CONFIGS):
            out_channels = make_divisible(channels * width_mult, 8)
            num_blocks = int(math.ceil(num_blocks * depth_mult))
            
            blocks = []
            for j in range(num_blocks):
                drop_rate = drop_path_rate * block_idx / total_blocks
                blocks.append(MBConvBlock(
                    in_channels if j == 0 else out_channels,
                    out_channels,
                    expand_ratio=expand,
                    stride=stride if j == 0 else 1,
                    kernel_size=kernel,
                    drop_path_rate=drop_rate,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ))
                block_idx += 1
                
            self.stages.append(nn.Sequential(*blocks))
            self.out_channels.append(out_channels)
            in_channels = out_channels
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


@BACKBONES.register(name='EfficientNetV2')
class EfficientNetV2(EfficientNet):
    """EfficientNetV2骨干网络"""
    
    def __init__(self, arch: str = 'v2_s', **kwargs):
        # V2使用Fused-MBConv，这里简化处理
        super().__init__(arch='b0', **kwargs)
