# -*- coding: utf-8 -*-
"""
YOLO系列专用Neck

支持YOLOv5/v7/v8 Neck。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import NECKS
from ..layers import ConvModule, CSPLayer, C2f


@NECKS.register(name='YOLONeck')
class YOLONeck(nn.Module):
    """
    YOLO Neck (PANet变体)
    
    Args:
        in_channels: 输入各层通道数
        out_channels: 输出各层通道数
        deepen_factor: 深度因子
        widen_factor: 宽度因子
        num_csp_blocks: CSP块数量
        norm_cfg: 归一化层配置
        act_cfg: 激活函数配置
    """
    
    def __init__(
        self,
        in_channels: List[int] = [256, 512, 1024],
        out_channels: List[int] = [256, 512, 1024],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        arch: str = 'yolov5',
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='SiLU', inplace=True),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_blocks = max(round(num_csp_blocks * deepen_factor), 1)
        
        Block = C2f if arch == 'yolov8' else CSPLayer

        # 自顶向下
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        
        for i in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(in_channels[i], in_channels[i-1], 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
            self.top_down_blocks.append(
                Block(in_channels[i-1] * 2, in_channels[i-1], num_blocks=num_blocks, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
            
        # 自底向上
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        
        for i in range(len(in_channels) - 1):
            self.downsample_layers.append(
                ConvModule(in_channels[i], in_channels[i], 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
            self.bottom_up_blocks.append(
                Block(in_channels[i] * 2, in_channels[i+1], num_blocks=num_blocks, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
            
        # 输出卷积
        self.out_convs = nn.ModuleList([
            ConvModule(ch, out_ch, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for ch, out_ch in zip(in_channels, out_channels)
        ])
        
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # inputs: [P3, P4, P5] 从小到大
        
        # 自顶向下
        inner_outs = [inputs[-1]]
        for i in range(len(inputs) - 1, 0, -1):
            feat_high = self.reduce_layers[len(inputs) - 1 - i](inner_outs[0])
            feat_high = F.interpolate(feat_high, size=inputs[i-1].shape[2:], mode='nearest')
            feat_low = inputs[i - 1]
            inner_outs.insert(0, self.top_down_blocks[len(inputs) - 1 - i](torch.cat([feat_low, feat_high], dim=1)))
            
        # 自底向上
        outs = [inner_outs[0]]
        for i in range(len(inputs) - 1):
            feat_low = self.downsample_layers[i](outs[-1])
            feat_high = inner_outs[i + 1]
            outs.append(self.bottom_up_blocks[i](torch.cat([feat_low, feat_high], dim=1)))
            
        # 输出
        return [self.out_convs[i](outs[i]) for i in range(len(outs))]
