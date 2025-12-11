# -*- coding: utf-8 -*-
"""
Mask R-CNN实例分割器
"""

from typing import Dict, Any, Optional, List
import torch

from registry.model_registry import MODELS, BACKBONES, NECKS, HEADS
from .base_segmentor import BaseSegmentor


@MODELS.register(name='MaskRCNN')
class MaskRCNN(BaseSegmentor):
    """
    Mask R-CNN实例分割器
    
    Args:
        backbone: 骨干网络配置
        neck: FPN配置
        rpn_head: RPN头配置
        roi_head: RoI头配置（含mask head）
        train_cfg: 训练配置
        test_cfg: 测试配置
    """
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        rpn_head: Optional[Dict[str, Any]] = None,
        roi_head: Optional[Dict[str, Any]] = None,
        mask_head: Optional[Dict[str, Any]] = None,
        train_cfg: Optional[Dict[str, Any]] = None,
        test_cfg: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(init_cfg)
        
        self.backbone = BACKBONES.build(backbone)
        
        if neck is not None:
            self.neck = NECKS.build(neck)
        else:
            self.neck = None
            
        if rpn_head is not None:
            self.rpn_head = HEADS.build(rpn_head)
        else:
            self.rpn_head = None
            
        if roi_head is not None:
            self.roi_head = HEADS.build(roi_head)
        else:
            self.roi_head = None
            
        if mask_head is not None:
            self.mask_head = HEADS.build(mask_head)
        else:
            self.mask_head = None
            
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x
        
    def encode_decode(self, img: torch.Tensor) -> torch.Tensor:
        # Mask R-CNN不使用encode_decode
        raise NotImplementedError
        
    def forward_train(self, inputs: torch.Tensor, targets: Any) -> Dict[str, torch.Tensor]:
        feats = self.extract_feat(inputs)
        losses = {}
        
        # RPN
        if self.rpn_head is not None:
            # 简化实现
            pass
            
        # RoI Head + Mask Head
        if self.roi_head is not None:
            # 简化实现
            pass
            
        return losses
        
    def forward_test(self, inputs: torch.Tensor) -> List[Any]:
        feats = self.extract_feat(inputs)
        # 简化实现
        return []
