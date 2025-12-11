# -*- coding: utf-8 -*-
"""
Bi-directional Feature Pyramid Network (BiFPN)

双向跨尺度连接，可学习的特征融合权重。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import NECKS
from ..layers import ConvModule, DepthwiseSeparableConv


class BiFPNLayer(nn.Module):
    """单层BiFPN"""
    
    def __init__(
        self,
        channels: int,
        num_levels: int,
        eps: float = 1e-4,
        norm_cfg: Optional[Dict[str, Any]] = None,
        act_cfg: Optional[Dict[str, Any]] = dict(type='SiLU', inplace=True),
    ):
        super().__init__()
        self.eps = eps
        self.num_levels = num_levels
        
        # 自顶向下权重
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(num_levels - 1)
        ])
        # 自底向上权重
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3 if i > 0 else 2)) for i in range(num_levels - 1)
        ])
        
        # 卷积
        self.td_convs = nn.ModuleList([
            DepthwiseSeparableConv(channels, channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(channels, channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(num_levels - 1)
        ])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # 自顶向下
        td_feats = [inputs[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.td_weights[i])
            w = w / (w.sum() + self.eps)
            up = F.interpolate(td_feats[0], size=inputs[i].shape[2:], mode='nearest')
            td_feats.insert(0, self.td_convs[self.num_levels - 2 - i](w[0] * inputs[i] + w[1] * up))
            
        # 自底向上
        bu_feats = [td_feats[0]]
        for i in range(self.num_levels - 1):
            w = F.relu(self.bu_weights[i])
            w = w / (w.sum() + self.eps)
            down = F.interpolate(bu_feats[-1], size=td_feats[i + 1].shape[2:], mode='nearest')
            if i == 0:
                feat = w[0] * td_feats[i + 1] + w[1] * down
            else:
                feat = w[0] * inputs[i + 1] + w[1] * td_feats[i + 1] + w[2] * down
            bu_feats.append(self.bu_convs[i](feat))
            
        return bu_feats


@NECKS.register(name='BiFPN')
class BiFPN(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    
    Args:
        in_channels: 输入各层通道数
        out_channels: 输出通道数
        num_outs: 输出层数
        num_repeats: BiFPN重复次数
        norm_cfg: 归一化层配置
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int = 5,
        num_repeats: int = 3,
        norm_cfg: Optional[Dict[str, Any]] = None,
        act_cfg: Optional[Dict[str, Any]] = dict(type='SiLU', inplace=True),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.num_levels = len(in_channels)
        
        # 输入投影
        self.input_convs = nn.ModuleList([
            ConvModule(ch, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
            for ch in in_channels
        ])
        
        # BiFPN层
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, self.num_levels, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(num_repeats)
        ])
        
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # 输入投影
        feats = [conv(x) for conv, x in zip(self.input_convs, inputs)]
        
        # BiFPN
        for bifpn in self.bifpn_layers:
            feats = bifpn(feats)
            
        return feats
