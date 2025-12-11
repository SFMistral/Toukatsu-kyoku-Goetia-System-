# -*- coding: utf-8 -*-
"""
Path Aggregation Network (PAN)

在FPN基础上增加自底向上路径。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import NECKS
from ..layers import ConvModule


@NECKS.register(name='PAN')
class PAN(nn.Module):
    """
    Path Aggregation Network
    
    Args:
        in_channels: 输入各层通道数列表
        out_channels: 输出统一通道数
        num_outs: 输出特征层数量
        start_level: 起始输入层级
        end_level: 结束输入层级
        norm_cfg: 归一化层配置
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int = 5,
        start_level: int = 0,
        end_level: int = -1,
        norm_cfg: Optional[Dict[str, Any]] = None,
        act_cfg: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        
        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level

        # FPN部分：横向连接 + 自顶向下
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
        # PAN部分：自底向上
        self.downsample_convs = nn.ModuleList()
        self.pan_convs = nn.ModuleList()
        
        for i in range(self.backbone_end_level - start_level - 1):
            d_conv = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            pan_conv = ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.downsample_convs.append(d_conv)
            self.pan_convs.append(pan_conv)
            
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # FPN: 横向连接
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # FPN: 自顶向下
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
            
        # FPN输出
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        # PAN: 自底向上
        pan_outs = [fpn_outs[0]]
        for i in range(len(self.downsample_convs)):
            down = self.downsample_convs[i](pan_outs[-1])
            pan_outs.append(self.pan_convs[i](down + fpn_outs[i + 1]))
            
        return pan_outs
