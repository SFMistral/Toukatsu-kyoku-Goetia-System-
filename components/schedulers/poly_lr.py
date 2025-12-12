# -*- coding: utf-8 -*-
"""
多项式学习率调度模块

提供 PolyLR、LinearLR 等多项式学习率调度器。
"""

from typing import List, Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler
from registry import SCHEDULERS, ComponentSource


class PolyLR(_LRScheduler):
    """
    多项式衰减学习率调度器
    
    学习率按多项式曲线衰减。
    
    学习率公式:
        lr = (base_lr - eta_min) * (1 - iter / total_iters) ^ power + eta_min
    
    Args:
        optimizer: 优化器实例
        total_iters: 总迭代数
        power: 多项式幂次 (默认 1.0)
        eta_min: 最小学习率 (默认 0)
        by_epoch: 按 epoch 更新 (默认 True)
        last_epoch: 上次 epoch (默认 -1)
        
    Note:
        - power=1.0: 线性衰减
        - power=0.9: 常用于语义分割
        - power=2.0: 平方衰减
        
    Example:
        >>> # 语义分割常用配置
        >>> scheduler = PolyLR(optimizer, total_iters=40000, power=0.9, by_epoch=False)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int,
        power: float = 1.0,
        eta_min: float = 0,
        by_epoch: bool = True,
        last_epoch: int = -1
    ):
        self.total_iters = total_iters
        self.power = power
        self.eta_min = eta_min
        self.by_epoch = by_epoch
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch >= self.total_iters:
            return [self.eta_min for _ in self.base_lrs]
            
        factor = (1 - self.last_epoch / self.total_iters) ** self.power
        return [
            (base_lr - self.eta_min) * factor + self.eta_min
            for base_lr in self.base_lrs
        ]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        if self.last_epoch >= self.total_iters:
            return [self.eta_min for _ in self.base_lrs]
            
        factor = (1 - self.last_epoch / self.total_iters) ** self.power
        return [
            (base_lr - self.eta_min) * factor + self.eta_min
            for base_lr in self.base_lrs
        ]


class LinearLR(_LRScheduler):
    """
    线性学习率调度器
    
    学习率从 base_lr * start_factor 线性变化到 base_lr * end_factor。
    这是 PolyLR 的特例 (power=1.0)。
    
    学习率公式:
        factor = start_factor + (end_factor - start_factor) * iter / total_iters
        lr = base_lr * factor
    
    Args:
        optimizer: 优化器实例
        start_factor: 起始因子 (默认 1.0)
        end_factor: 结束因子 (默认 0.0)
        total_iters: 总迭代数
        last_epoch: 上次 epoch (默认 -1)
        
    Example:
        >>> # 线性衰减到 0
        >>> scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=100)
        >>> # 线性预热
        >>> scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=5)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        start_factor: float = 1.0,
        end_factor: float = 0.0,
        total_iters: int = 5,
        last_epoch: int = -1
    ):
        if start_factor < 0 or start_factor > 1:
            raise ValueError(f"start_factor must be in [0, 1], got {start_factor}")
        if end_factor < 0 or end_factor > 1:
            raise ValueError(f"end_factor must be in [0, 1], got {end_factor}")
            
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
            
        if self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
            
        factor = self.start_factor + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
        return [base_lr * factor for base_lr in self.base_lrs]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        if self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
            
        factor = self.start_factor + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
        return [base_lr * factor for base_lr in self.base_lrs]


# 注册调度器
SCHEDULERS.register('PolyLR', PolyLR, category='polynomial', source=ComponentSource.BUILTIN)
SCHEDULERS.register('LinearLR', LinearLR, category='polynomial', source=ComponentSource.BUILTIN)
