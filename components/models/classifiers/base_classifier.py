# -*- coding: utf-8 -*-
"""
分类器基类

定义分类器统一接口。
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List
import torch

from ..base_model import BaseModel


class BaseClassifier(BaseModel):
    """
    分类器基类
    
    定义分类器统一接口，组合Backbone + Head。
    """
    
    def __init__(self, init_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(init_cfg)
        
    @abstractmethod
    def extract_feat(self, img: torch.Tensor) -> torch.Tensor:
        """特征提取"""
        pass
        
    def forward_train(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """训练前向"""
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, targets)
        
    def forward_test(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理前向"""
        feats = self.extract_feat(inputs)
        return self.head.predict(feats)
        
    def simple_test(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """单图推理"""
        return self.forward_test(img)
