# -*- coding: utf-8 -*-
"""
MobileNet系列骨干网络

支持MobileNetV2、MobileNetV3-Small/Large。
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn

from registry.model_registry import BACKBONES
from ..layers import ConvModule, InvertedResidual, SELayer
from ..utils import make_divisible


@BACKBONES.register(name='MobileNetV2')
class MobileNetV2(nn.Module):
    """
    MobileNetV2骨干网络
    
    Args:
        out_indices: 输出特征层索引
        frozen_stages: 冻结的stage数量
        width_mult: 宽度乘数
        norm_cfg: 归一化层配置
    """
    
    # (expand_ratio, channels, num_blocks, stride)
    INVERTED_RESIDUAL_SETTING = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    
    def __init__(
        self,
        out_indices: Tuple[int, ...] = (1, 2, 4, 6),
        frozen_stages: int = -1,
        width_mult: float = 1.0,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='ReLU6', inplace=True),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Stem
        in_channels = make_divisible(32 * width_mult, 8)
        self.stem = ConvModule(3, in_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # Stages
        self.stages = nn.ModuleList()
        self.out_channels = []
        
        for i, (t, c, n, s) in enumerate(self.INVERTED_RESIDUAL_SETTING):
            out_channels = make_divisible(c * width_mult, 8)
            blocks = []
            for j in range(n):
                stride = s if j == 0 else 1
                blocks.append(InvertedResidual(
                    in_channels, out_channels,
                    stride=stride, expand_ratio=t,
                    norm_cfg=norm_cfg, act_cfg=act_cfg,
                ))
                in_channels = out_channels
            self.stages.append(nn.Sequential(*blocks))
            self.out_channels.append(out_channels)
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


@BACKBONES.register(name='MobileNetV3')
class MobileNetV3(nn.Module):
    """
    MobileNetV3骨干网络
    
    Args:
        arch: 架构 'small' / 'large'
        out_indices: 输出特征层索引
        frozen_stages: 冻结的stage数量
        width_mult: 宽度乘数
        norm_cfg: 归一化层配置
    """
    
    # (kernel, expand, out, se, act, stride)
    LARGE_SETTING = [
        (3, 16, 16, False, 'RE', 1),
        (3, 64, 24, False, 'RE', 2),
        (3, 72, 24, False, 'RE', 1),
        (5, 72, 40, True, 'RE', 2),
        (5, 120, 40, True, 'RE', 1),
        (5, 120, 40, True, 'RE', 1),
        (3, 240, 80, False, 'HS', 2),
        (3, 200, 80, False, 'HS', 1),
        (3, 184, 80, False, 'HS', 1),
        (3, 184, 80, False, 'HS', 1),
        (3, 480, 112, True, 'HS', 1),
        (3, 672, 112, True, 'HS', 1),
        (5, 672, 160, True, 'HS', 2),
        (5, 960, 160, True, 'HS', 1),
        (5, 960, 160, True, 'HS', 1),
    ]
    
    SMALL_SETTING = [
        (3, 16, 16, True, 'RE', 2),
        (3, 72, 24, False, 'RE', 2),
        (3, 88, 24, False, 'RE', 1),
        (5, 96, 40, True, 'HS', 2),
        (5, 240, 40, True, 'HS', 1),
        (5, 240, 40, True, 'HS', 1),
        (5, 120, 48, True, 'HS', 1),
        (5, 144, 48, True, 'HS', 1),
        (5, 288, 96, True, 'HS', 2),
        (5, 576, 96, True, 'HS', 1),
        (5, 576, 96, True, 'HS', 1),
    ]
    
    def __init__(
        self,
        arch: str = 'large',
        out_indices: Tuple[int, ...] = (3, 6, 12, 15),
        frozen_stages: int = -1,
        width_mult: float = 1.0,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        setting = self.LARGE_SETTING if arch == 'large' else self.SMALL_SETTING
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Stem
        in_channels = make_divisible(16 * width_mult, 8)
        self.stem = ConvModule(
            3, in_channels, 3, stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='HardSwish'),
        )
        
        # Layers
        self.layers = nn.ModuleList()
        self.out_channels = []
        
        for i, (k, exp, c, se, act, s) in enumerate(setting):
            out_channels = make_divisible(c * width_mult, 8)
            act_cfg = dict(type='ReLU', inplace=True) if act == 'RE' else dict(type='HardSwish')
            
            self.layers.append(InvertedResidual(
                in_channels, out_channels,
                stride=s, expand_ratio=exp / in_channels,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
                use_se=se, se_ratio=0.25,
            ))
            in_channels = out_channels
            self.out_channels.append(out_channels)
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
