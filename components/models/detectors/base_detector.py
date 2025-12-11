# -*- coding: utf-8 -*-
"""
检测器基类

定义检测器统一接口。
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List
import torch

from ..base_model import BaseModel


class BaseDetector(BaseModel):
    """
    检测器基类
    
    定义检测器统一接口，组合Backbone + Neck + Head。
    """
    
    def __init__(self, init_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(init_cfg)
        
    @abstractmethod
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """特征提取（Backbone + Neck）"""
        pass
        
    @abstractmethod
    def forward_train(self, img: torch.Tensor, targets: Any) -> Dict[str, torch.Tensor]:
        """训练前向"""
        pass
        
    @abstractmethod
    def forward_test(self, img: torch.Tensor) -> List[Any]:
        """推理前向"""
        pass
        
    def simple_test(self, img: torch.Tensor) -> List[Any]:
        """单图推理"""
        return self.forward_test(img)
        
    def aug_test(self, imgs: List[torch.Tensor]) -> List[Any]:
        """增强测试（TTA）"""
        # 简化实现：取平均
        results = [self.simple_test(img) for img in imgs]
        return results[0]  # 实际应该做结果融合
