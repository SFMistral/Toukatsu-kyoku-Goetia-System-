# -*- coding: utf-8 -*-
"""
Feature Pyramid Network (FPN)

多尺度特征融合，自顶向下路径 + 横向连接。
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import NECKS
from ..layers import ConvModule


@NECKS.register(name='FPN')
class FPN(nn.Module):
    """
    Feature Pyramid Network
    
    Args:
        in_channels: 输入各层通道数列表
        out_channels: 输出统一通道数
        num_outs: 输出特征层数量
        start_level: 起始输入层级
        end_level: 结束输入层级
        add_extra_convs: 是否添加额外卷积层
        relu_before_extra_convs: 额外卷积前是否加ReLU
        norm_cfg: 归一化层配置
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int = 5,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: bool = False,
        relu_before_extra_convs: bool = False,
        norm_cfg: Optional[Dict[str, Any]] = None,
        act_cfg: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        
        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level

        # 横向连接（1x1卷积）
        self.lateral_convs = nn.ModuleList()
        # 输出卷积（3x3卷积）
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i], out_channels, 1,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            )
            fpn_conv = ConvModule(
                out_channels, out_channels, 3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
        # 额外层
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels > 0:
            self.extra_convs = nn.ModuleList()
            for i in range(extra_levels):
                in_ch = out_channels if i > 0 else in_channels[self.backbone_end_level - 1]
                self.extra_convs.append(ConvModule(
                    in_ch, out_channels, 3, stride=2, padding=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg,
                ))
        else:
            self.extra_convs = None
            
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            inputs: 多尺度特征列表
            
        Returns:
            融合后的特征金字塔
        """
        # 横向连接
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest'
            )
            
        # 输出卷积
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(len(laterals))
        ]
        
        # 额外层
        if self.extra_convs is not None:
            for i, extra_conv in enumerate(self.extra_convs):
                if i == 0 and not self.add_extra_convs:
                    x = F.max_pool2d(outs[-1], 1, stride=2)
                else:
                    if self.relu_before_extra_convs:
                        x = F.relu(outs[-1] if i == 0 else x)
                    else:
                        x = outs[-1] if i == 0 else x
                    x = extra_conv(x)
                outs.append(x)
        elif self.num_outs > len(outs):
            # 使用max pooling生成额外层
            for _ in range(self.num_outs - len(outs)):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
                
        return outs
