# -*- coding: utf-8 -*-
"""
Darknet系列骨干网络（YOLO系列）

支持Darknet-53、CSPDarknet等。
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn

from registry.model_registry import BACKBONES
from ..layers import ConvModule, CSPLayer, C2f, SPPF


class DarknetBlock(nn.Module):
    """Darknet基础块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='LeakyReLU', negative_slope=0.1),
    ):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = ConvModule(in_channels, hidden_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(hidden_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


@BACKBONES.register(name='Darknet')
class Darknet(nn.Module):
    """
    Darknet-53骨干网络（YOLOv3）
    
    Args:
        out_indices: 输出特征层索引
        frozen_stages: 冻结的stage数量
        norm_cfg: 归一化层配置
        act_cfg: 激活函数配置
    """
    
    # (channels, num_blocks)
    STAGE_CONFIGS = [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)]
    
    def __init__(
        self,
        out_indices: Tuple[int, ...] = (2, 3, 4),
        frozen_stages: int = -1,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Stem
        self.stem = ConvModule(3, 32, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # Stages
        self.stages = nn.ModuleList()
        self.out_channels = []
        in_channels = 32
        
        for i, (channels, num_blocks) in enumerate(self.STAGE_CONFIGS):
            layers = [ConvModule(in_channels, channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)]
            for _ in range(num_blocks):
                layers.append(DarknetBlock(channels, channels, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.stages.append(nn.Sequential(*layers))
            self.out_channels.append(channels)
            in_channels = channels
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


@BACKBONES.register(name='CSPDarknet')
class CSPDarknet(nn.Module):
    """
    CSPDarknet骨干网络（YOLOv5/v8）
    
    Args:
        arch: 架构 'yolov5' / 'yolov8'
        deepen_factor: 深度因子
        widen_factor: 宽度因子
        out_indices: 输出特征层索引
        frozen_stages: 冻结的stage数量
        norm_cfg: 归一化层配置
        act_cfg: 激活函数配置
    """
    
    def __init__(
        self,
        arch: str = 'yolov5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Tuple[int, ...] = (2, 3, 4),
        frozen_stages: int = -1,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='SiLU', inplace=True),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # 基础通道数
        base_channels = int(64 * widen_factor)
        base_depth = max(round(3 * deepen_factor), 1)
        
        # Stem
        self.stem = ConvModule(3, base_channels, 6, stride=2, padding=2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # Stages
        self.stages = nn.ModuleList()
        self.out_channels = []
        
        # Stage configs: (out_channels_mult, num_blocks_mult, use_sppf)
        stage_configs = [
            (2, 1, False),   # P2
            (4, 2, False),   # P3
            (8, 3, False),   # P4
            (16, 1, True),   # P5
        ]
        
        in_channels = base_channels
        for i, (ch_mult, depth_mult, use_sppf) in enumerate(stage_configs):
            out_channels = int(base_channels * ch_mult)
            num_blocks = max(round(base_depth * depth_mult), 1)
            
            layers = [
                ConvModule(in_channels, out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ]
            
            if arch == 'yolov8':
                layers.append(C2f(out_channels, out_channels, num_blocks=num_blocks, norm_cfg=norm_cfg, act_cfg=act_cfg))
            else:
                layers.append(CSPLayer(out_channels, out_channels, num_blocks=num_blocks, norm_cfg=norm_cfg, act_cfg=act_cfg))
                
            if use_sppf:
                layers.append(SPPF(out_channels, out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg))
                
            self.stages.append(nn.Sequential(*layers))
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
