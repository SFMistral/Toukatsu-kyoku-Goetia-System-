# -*- coding: utf-8 -*-
"""
通用图像分类器

组合Backbone + Neck + Head构建分类器。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import MODELS, BACKBONES, NECKS, HEADS
from .base_classifier import BaseClassifier


class GlobalAveragePooling(nn.Module):
    """全局平均池化"""
    
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[-1]
        if x.dim() == 4:
            return F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x


@MODELS.register(name='ImageClassifier')
class ImageClassifier(BaseClassifier):
    """
    通用图像分类器
    
    Args:
        backbone: 骨干网络配置
        neck: Neck配置（可选）
        head: 分类头配置
        pretrained: 预训练权重路径
        init_cfg: 初始化配置
    """
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        pretrained: Optional[str] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(init_cfg)
        
        # 构建backbone
        self.backbone = BACKBONES.build(backbone)
        
        # 构建neck
        if neck is not None:
            self.neck = NECKS.build(neck)
        else:
            self.neck = GlobalAveragePooling()

        # 构建head
        # 自动设置in_channels
        if 'in_channels' not in head:
            if hasattr(self.backbone, 'out_channels'):
                out_channels = self.backbone.out_channels
                head['in_channels'] = out_channels[-1] if isinstance(out_channels, (list, tuple)) else out_channels
        self.head = HEADS.build(head)
        
        # 加载预训练
        if pretrained:
            self.load_pretrained(pretrained)
            
    def extract_feat(self, img: torch.Tensor) -> torch.Tensor:
        """特征提取"""
        x = self.backbone(img)
        x = self.neck(x)
        return x
        
    def forward_features(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """返回中间特征"""
        return self.backbone(inputs)
