# -*- coding: utf-8 -*-
"""
FCOS无锚框检测器
"""

from typing import Dict, Any, Optional, List
import torch

from registry.model_registry import MODELS, BACKBONES, NECKS, HEADS
from .base_detector import BaseDetector


@MODELS.register(name='FCOS')
class FCOS(BaseDetector):
    """
    FCOS无锚框检测器
    
    Args:
        backbone: 骨干网络配置
        neck: FPN配置
        head: FCOS Head配置
        train_cfg: 训练配置
        test_cfg: 测试配置
    """
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        neck: Dict[str, Any],
        head: Dict[str, Any],
        train_cfg: Optional[Dict[str, Any]] = None,
        test_cfg: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(init_cfg)
        
        self.backbone = BACKBONES.build(backbone)
        self.neck = NECKS.build(neck)
        self.head = HEADS.build(head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone(img)
        x = self.neck(x)
        return x
        
    def forward_train(self, img: torch.Tensor, targets: Any) -> Dict[str, torch.Tensor]:
        feats = self.extract_feat(img)
        cls_scores, bbox_preds, centernesses = self.head(feats)
        return self.head.loss(cls_scores, bbox_preds, centernesses, targets)
        
    def forward_test(self, img: torch.Tensor) -> List[Any]:
        feats = self.extract_feat(img)
        cls_scores, bbox_preds, centernesses = self.head(feats)
        return cls_scores, bbox_preds, centernesses
