# -*- coding: utf-8 -*-
"""
语义分割任务头

提供FCN、ASPP、PSP、UPer、SegFormer等分割头。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import HEADS
from ..layers import ConvModule, DepthwiseSeparableConv


@HEADS.register(name='FCNHead')
class FCNHead(nn.Module):
    """
    全卷积分割头
    
    Args:
        num_classes: 类别数
        in_channels: 输入通道数
        channels: 中间特征通道数
        num_convs: 卷积层数量
        dropout_ratio: Dropout比例
        norm_cfg: 归一化层配置
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        channels: int = 256,
        num_convs: int = 2,
        dropout_ratio: float = 0.1,
        in_index: int = -1,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='ReLU', inplace=True),
        align_corners: bool = False,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.in_index = in_index
        self.align_corners = align_corners
        
        convs = []
        for i in range(num_convs):
            convs.append(ConvModule(
                in_channels if i == 0 else channels,
                channels, 3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
        self.convs = nn.Sequential(*convs)
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = inputs[self.in_index] if isinstance(inputs, (list, tuple)) else inputs
        x = self.convs(x)
        x = self.dropout(x)
        return self.conv_seg(x)
        
    def loss(self, seg_logits: torch.Tensor, seg_label: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg_logits = F.interpolate(seg_logits, size=seg_label.shape[1:], mode='bilinear', align_corners=self.align_corners)
        return {'loss_seg': self.loss_fn(seg_logits, seg_label)}


@HEADS.register(name='ASPPHead')
class ASPPHead(nn.Module):
    """空洞空间金字塔池化头"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        channels: int = 256,
        dilations: List[int] = [1, 6, 12, 18],
        dropout_ratio: float = 0.1,
        in_index: int = -1,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='ReLU', inplace=True),
        align_corners: bool = False,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.in_index = in_index
        self.align_corners = align_corners
        
        # ASPP模块
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                self.aspp.append(ConvModule(in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
            else:
                self.aspp.append(ConvModule(in_channels, channels, 3, padding=dilation, dilation=dilation, norm_cfg=norm_cfg, act_cfg=act_cfg))
                
        # 全局池化分支
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        
        # 融合
        self.bottleneck = ConvModule((len(dilations) + 1) * channels, channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = inputs[self.in_index] if isinstance(inputs, (list, tuple)) else inputs
        
        aspp_outs = [aspp(x) for aspp in self.aspp]
        pool_out = self.image_pool(x)
        pool_out = F.interpolate(pool_out, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
        aspp_outs.append(pool_out)
        
        x = torch.cat(aspp_outs, dim=1)
        x = self.bottleneck(x)
        x = self.dropout(x)
        return self.conv_seg(x)
        
    def loss(self, seg_logits: torch.Tensor, seg_label: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg_logits = F.interpolate(seg_logits, size=seg_label.shape[1:], mode='bilinear', align_corners=self.align_corners)
        return {'loss_seg': self.loss_fn(seg_logits, seg_label)}


@HEADS.register(name='PSPHead')
class PSPHead(nn.Module):
    """金字塔池化头"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        channels: int = 512,
        pool_scales: List[int] = [1, 2, 3, 6],
        dropout_ratio: float = 0.1,
        in_index: int = -1,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='ReLU', inplace=True),
        align_corners: bool = False,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.in_index = in_index
        self.align_corners = align_corners
        
        # PPM模块
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(in_channels, channels // len(pool_scales), 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ) for scale in pool_scales
        ])
        
        self.bottleneck = ConvModule(in_channels + channels, channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = inputs[self.in_index] if isinstance(inputs, (list, tuple)) else inputs
        
        ppm_outs = [x]
        for ppm in self.ppm:
            ppm_out = ppm(x)
            ppm_out = F.interpolate(ppm_out, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(ppm_out)
            
        x = torch.cat(ppm_outs, dim=1)
        x = self.bottleneck(x)
        x = self.dropout(x)
        return self.conv_seg(x)
        
    def loss(self, seg_logits: torch.Tensor, seg_label: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg_logits = F.interpolate(seg_logits, size=seg_label.shape[1:], mode='bilinear', align_corners=self.align_corners)
        return {'loss_seg': self.loss_fn(seg_logits, seg_label)}


@HEADS.register(name='UPerHead')
class UPerHead(nn.Module):
    """统一感知解析头"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: List[int],
        channels: int = 512,
        pool_scales: List[int] = [1, 2, 3, 6],
        dropout_ratio: float = 0.1,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='ReLU', inplace=True),
        align_corners: bool = False,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.align_corners = align_corners
        
        # PPM on last feature
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(in_channels[-1], channels // len(pool_scales), 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ) for scale in pool_scales
        ])
        self.ppm_bottleneck = ConvModule(in_channels[-1] + channels, channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # FPN
        self.lateral_convs = nn.ModuleList([ConvModule(ch, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg) for ch in in_channels[:-1]])
        self.fpn_convs = nn.ModuleList([ConvModule(channels, channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg) for _ in in_channels[:-1]])
        
        # Fusion
        self.fpn_bottleneck = ConvModule(len(in_channels) * channels, channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # PPM
        x = inputs[-1]
        ppm_outs = [x]
        for ppm in self.ppm:
            ppm_out = ppm(x)
            ppm_out = F.interpolate(ppm_out, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(ppm_out)
        ppm_out = self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))
        
        # FPN
        fpn_outs = [ppm_out]
        for i in range(len(inputs) - 2, -1, -1):
            lateral = self.lateral_convs[i](inputs[i])
            fpn_outs.insert(0, self.fpn_convs[i](lateral))
            
        # Upsample and concat
        target_size = fpn_outs[0].shape[2:]
        fpn_outs = [F.interpolate(out, size=target_size, mode='bilinear', align_corners=self.align_corners) for out in fpn_outs]
        
        x = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        x = self.dropout(x)
        return self.conv_seg(x)
        
    def loss(self, seg_logits: torch.Tensor, seg_label: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg_logits = F.interpolate(seg_logits, size=seg_label.shape[1:], mode='bilinear', align_corners=self.align_corners)
        return {'loss_seg': self.loss_fn(seg_logits, seg_label)}


@HEADS.register(name='SegFormerHead')
class SegFormerHead(nn.Module):
    """SegFormer轻量头"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: List[int],
        channels: int = 256,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.align_corners = align_corners
        
        self.linear_fuse = nn.ModuleList([nn.Conv2d(ch, channels, 1) for ch in in_channels])
        self.linear_pred = nn.Sequential(
            nn.Conv2d(channels * len(in_channels), channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(channels, num_classes, 1),
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        target_size = inputs[0].shape[2:]
        outs = []
        for i, (x, linear) in enumerate(zip(inputs, self.linear_fuse)):
            x = linear(x)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
            outs.append(x)
        return self.linear_pred(torch.cat(outs, dim=1))
        
    def loss(self, seg_logits: torch.Tensor, seg_label: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg_logits = F.interpolate(seg_logits, size=seg_label.shape[1:], mode='bilinear', align_corners=self.align_corners)
        return {'loss_seg': self.loss_fn(seg_logits, seg_label)}
