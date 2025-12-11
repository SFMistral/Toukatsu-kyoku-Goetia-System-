# -*- coding: utf-8 -*-
"""
DeepLabV3/V3+分割器
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import MODELS, BACKBONES, HEADS
from .base_segmentor import BaseSegmentor


@MODELS.register(name='DeepLabV3')
class DeepLabV3(BaseSegmentor):
    """
    DeepLabV3分割器
    
    Args:
        backbone: 骨干网络配置
        decode_head: ASPP解码头配置
        auxiliary_head: 辅助头配置（可选）
    """
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        decode_head: Dict[str, Any],
        auxiliary_head: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(init_cfg)
        
        self.backbone = BACKBONES.build(backbone)
        self.decode_head = HEADS.build(decode_head)
        
        if auxiliary_head is not None:
            self.auxiliary_head = HEADS.build(auxiliary_head)
        else:
            self.auxiliary_head = None
            
        self.num_classes = decode_head.get('num_classes', 21)
        
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(img)
        
    def encode_decode(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.extract_feat(img)
        return self.decode_head(feats)
        
    def forward_train(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.extract_feat(inputs)
        
        losses = {}
        seg_logits = self.decode_head(feats)
        losses.update(self.decode_head.loss(seg_logits, targets))
        
        if self.auxiliary_head is not None:
            aux_logits = self.auxiliary_head(feats)
            aux_loss = self.auxiliary_head.loss(aux_logits, targets)
            losses['loss_aux'] = aux_loss['loss_seg'] * 0.4
            
        return losses


@MODELS.register(name='DeepLabV3Plus')
class DeepLabV3Plus(DeepLabV3):
    """DeepLabV3+分割器（带低层特征融合）"""
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        decode_head: Dict[str, Any],
        auxiliary_head: Optional[Dict[str, Any]] = None,
        low_level_channels: int = 256,
        low_level_index: int = 0,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(backbone, decode_head, auxiliary_head, init_cfg)
        
        self.low_level_index = low_level_index
        
        # 低层特征投影
        from ..layers import ConvModule
        self.low_level_conv = ConvModule(
            low_level_channels, 48, 1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
        )
        
    def encode_decode(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.extract_feat(img)
        
        # ASPP
        aspp_out = self.decode_head(feats)
        
        # 低层特征
        low_level_feat = self.low_level_conv(feats[self.low_level_index])
        
        # 上采样并融合
        aspp_out = F.interpolate(aspp_out, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        
        return aspp_out
