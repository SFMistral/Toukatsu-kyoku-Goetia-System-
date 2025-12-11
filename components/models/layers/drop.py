# -*- coding: utf-8 -*-
"""
正则化Drop层模块

提供Dropout、DropPath、DropBlock等正则化层。
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Dropout):
    """标准Dropout层（包装）"""
    pass


class DropPath(nn.Module):
    """
    随机深度（Stochastic Depth）
    
    在训练时随机丢弃整个残差分支。
    
    Args:
        drop_prob: Drop概率
        scale_by_keep: 是否按保留率缩放
    """
    
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        
        if self.scale_by_keep:
            random_tensor.div_(keep_prob)
            
        return x * random_tensor
        
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class DropBlock2d(nn.Module):
    """
    DropBlock正则化
    
    随机丢弃连续的区域而不是单个元素。
    
    Args:
        drop_prob: Drop概率
        block_size: 丢弃块大小
    """
    
    def __init__(self, drop_prob: float = 0.1, block_size: int = 7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        B, C, H, W = x.shape
        
        # 计算gamma（采样概率）
        gamma = (
            self.drop_prob / (self.block_size ** 2) *
            (H * W) / ((H - self.block_size + 1) * (W - self.block_size + 1))
        )
        
        # 生成mask
        mask = (torch.rand(B, C, H, W, device=x.device) < gamma).float()
        
        # 扩展mask到block_size
        block_mask = F.max_pool2d(
            mask,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2,
        )
        
        # 裁剪到原始大小
        if block_mask.shape[2] != H or block_mask.shape[3] != W:
            block_mask = block_mask[:, :, :H, :W]
            
        block_mask = 1 - block_mask
        
        # 归一化
        normalize_scale = block_mask.numel() / (block_mask.sum() + 1e-6)
        
        return x * block_mask * normalize_scale
        
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}, block_size={self.block_size}'
