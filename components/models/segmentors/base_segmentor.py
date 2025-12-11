# -*- coding: utf-8 -*-
"""
分割器基类

定义分割器统一接口。
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F

from ..base_model import BaseModel


class BaseSegmentor(BaseModel):
    """
    分割器基类
    
    定义分割器统一接口，组合Backbone + Decode Head。
    """
    
    def __init__(self, init_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(init_cfg)
        self.align_corners = False
        
    @abstractmethod
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """特征提取"""
        pass
        
    @abstractmethod
    def encode_decode(self, img: torch.Tensor) -> torch.Tensor:
        """编码解码流程"""
        pass
        
    def forward_train(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """训练前向"""
        seg_logits = self.encode_decode(inputs)
        return self.decode_head.loss(seg_logits, targets)
        
    def forward_test(self, inputs: torch.Tensor) -> torch.Tensor:
        """推理前向"""
        return self.inference(inputs)
        
    def inference(self, img: torch.Tensor) -> torch.Tensor:
        """推理（支持滑窗/翻转增强）"""
        return self.whole_inference(img)
        
    def whole_inference(self, img: torch.Tensor) -> torch.Tensor:
        """整图推理"""
        seg_logits = self.encode_decode(img)
        return seg_logits.argmax(dim=1)
        
    def slide_inference(self, img: torch.Tensor, crop_size: tuple, stride: tuple) -> torch.Tensor:
        """滑窗推理（大图）"""
        B, C, H, W = img.shape
        h_crop, w_crop = crop_size
        h_stride, w_stride = stride
        
        h_grids = max(H - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(W - w_crop + w_stride - 1, 0) // w_stride + 1
        
        preds = img.new_zeros((B, self.num_classes, H, W))
        count = img.new_zeros((B, 1, H, W))
        
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, H)
                x2 = min(x1 + w_crop, W)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg = self.encode_decode(crop_img)
                
                preds[:, :, y1:y2, x1:x2] += crop_seg
                count[:, :, y1:y2, x1:x2] += 1
                
        return (preds / count).argmax(dim=1)
