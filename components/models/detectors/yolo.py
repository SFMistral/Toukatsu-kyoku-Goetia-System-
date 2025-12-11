# -*- coding: utf-8 -*-
"""
YOLO系列检测器
"""

from typing import Dict, Any, Optional, List
import torch

from registry.model_registry import MODELS, BACKBONES, NECKS, HEADS
from .base_detector import BaseDetector


@MODELS.register(name='YOLO')
class YOLO(BaseDetector):
    """
    YOLO检测器
    
    Args:
        backbone: Darknet/CSPDarknet配置
        neck: YOLO Neck配置
        head: YOLO Head配置
        size: 模型尺寸 n/s/m/l/x
        train_cfg: 训练配置
        test_cfg: 测试配置
    """
    
    # 尺寸配置
    SIZE_CONFIGS = {
        'n': {'deepen_factor': 0.33, 'widen_factor': 0.25},
        's': {'deepen_factor': 0.33, 'widen_factor': 0.50},
        'm': {'deepen_factor': 0.67, 'widen_factor': 0.75},
        'l': {'deepen_factor': 1.0, 'widen_factor': 1.0},
        'x': {'deepen_factor': 1.33, 'widen_factor': 1.25},
    }
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        neck: Dict[str, Any],
        head: Dict[str, Any],
        size: str = 's',
        train_cfg: Optional[Dict[str, Any]] = None,
        test_cfg: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(init_cfg)
        
        # 应用尺寸配置
        if size in self.SIZE_CONFIGS:
            cfg = self.SIZE_CONFIGS[size]
            backbone.setdefault('deepen_factor', cfg['deepen_factor'])
            backbone.setdefault('widen_factor', cfg['widen_factor'])
            neck.setdefault('deepen_factor', cfg['deepen_factor'])
            neck.setdefault('widen_factor', cfg['widen_factor'])
            
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
        outputs = self.head(feats)
        return self.head.loss(outputs, targets)
        
    def forward_test(self, img: torch.Tensor) -> List[Any]:
        feats = self.extract_feat(img)
        outputs = self.head(feats)
        # 后处理：NMS等
        return outputs
