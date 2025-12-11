# -*- coding: utf-8 -*-
"""
SegFormer分割器
"""

from typing import Dict, Any, Optional, List
import torch

from registry.model_registry import MODELS, BACKBONES, HEADS
from .base_segmentor import BaseSegmentor


@MODELS.register(name='SegFormer')
class SegFormer(BaseSegmentor):
    """
    SegFormer分割器
    
    Args:
        backbone: Mix Transformer配置
        decode_head: SegFormer Head配置
    """
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        decode_head: Dict[str, Any],
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(init_cfg)
        
        self.backbone = BACKBONES.build(backbone)
        self.decode_head = HEADS.build(decode_head)
        self.num_classes = decode_head.get('num_classes', 21)
        
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(img)
        
    def encode_decode(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.extract_feat(img)
        return self.decode_head(feats)


@MODELS.register(name='PSPNet')
class PSPNet(BaseSegmentor):
    """
    PSPNet分割器
    
    Args:
        backbone: 骨干网络配置
        decode_head: PSP Head配置
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
