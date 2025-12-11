# -*- coding: utf-8 -*-
"""
Faster R-CNN检测器
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

from registry.model_registry import MODELS, BACKBONES, NECKS, HEADS
from .base_detector import BaseDetector


@MODELS.register(name='FasterRCNN')
class FasterRCNN(BaseDetector):
    """
    Faster R-CNN检测器
    
    Args:
        backbone: 骨干网络配置
        neck: Neck配置
        rpn_head: RPN头配置
        roi_head: RoI头配置
        train_cfg: 训练配置
        test_cfg: 测试配置
    """
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        rpn_head: Optional[Dict[str, Any]] = None,
        roi_head: Optional[Dict[str, Any]] = None,
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
            
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x
        
    def forward_train(self, img: torch.Tensor, targets: Any) -> Dict[str, torch.Tensor]:
        feats = self.extract_feat(img)
        losses = {}
        
        # RPN
        if self.rpn_head is not None:
            rpn_outs = self.rpn_head(feats)
            # 简化：实际需要计算RPN损失
            
        # RoI Head
        if self.roi_head is not None:
            # 简化：实际需要RoI pooling和损失计算
            pass
            
        return losses
        
    def forward_test(self, img: torch.Tensor) -> List[Any]:
        feats = self.extract_feat(img)
        # 简化：实际需要RPN提议和RoI预测
        return []
