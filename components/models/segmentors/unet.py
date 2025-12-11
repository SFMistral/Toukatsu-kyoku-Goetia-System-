# -*- coding: utf-8 -*-
"""
U-Net分割器
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import MODELS
from .base_segmentor import BaseSegmentor
from ..layers import ConvModule


class DoubleConv(nn.Module):
    """双卷积块"""
    
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN')):
        super().__init__()
        self.conv = nn.Sequential(
            ConvModule(in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg),
        )
        
    def forward(self, x):
        return self.conv(x)


@MODELS.register(name='UNet')
class UNet(BaseSegmentor):
    """
    U-Net分割器
    
    Args:
        in_channels: 输入通道数
        num_classes: 类别数
        base_channels: 基础通道数
        num_stages: 编码器/解码器阶段数
        norm_cfg: 归一化配置
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        base_channels: int = 64,
        num_stages: int = 5,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(init_cfg)
        
        self.num_classes = num_classes
        
        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        ch = in_channels
        for i in range(num_stages):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(DoubleConv(ch, out_ch, norm_cfg))
            if i < num_stages - 1:
                self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
            
        # 解码器
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(num_stages - 2, -1, -1):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.decoders.append(DoubleConv(in_ch, out_ch, norm_cfg))
            
        # 输出
        self.out_conv = nn.Conv2d(base_channels, num_classes, 1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = img
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            feats.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        return feats
        
    def encode_decode(self, img: torch.Tensor) -> torch.Tensor:
        # 编码
        enc_feats = []
        x = img
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            enc_feats.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
                
        # 解码
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = enc_feats[-(i + 2)]
            # 处理尺寸不匹配
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            
        return self.out_conv(x)
        
    def forward_train(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg_logits = self.encode_decode(inputs)
        seg_logits = F.interpolate(seg_logits, size=targets.shape[1:], mode='bilinear', align_corners=False)
        return {'loss_seg': self.loss_fn(seg_logits, targets)}
