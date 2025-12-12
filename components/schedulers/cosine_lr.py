# -*- coding: utf-8 -*-
"""
余弦退火学习率调度模块

提供 CosineAnnealingLR、CosineAnnealingWarmRestarts 等余弦学习率调度器。
"""

import math
from typing import List, Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler
from registry import SCHEDULERS, ComponentSource


class CosineAnnealingLR(_LRScheduler):
    """
    余弦退火学习率调度器
    
    学习率按余弦曲线从初始值衰减到最小值。
    
    学习率公式:
        lr = eta_min + (base_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
    
    Args:
        optimizer: 优化器实例
        T_max: 周期长度 (epoch)
        eta_min: 最小学习率 (默认 0)
        last_epoch: 上次 epoch (默认 -1)
        
    Example:
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch == 0:
            return self.base_lrs
            
        if self.last_epoch == self.T_max:
            return [self.eta_min for _ in self.base_lrs]
            
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    带热重启的余弦退火学习率调度器
    
    每个周期结束后重置学习率，周期可逐渐增长。
    
    Args:
        optimizer: 优化器实例
        T_0: 首次周期长度 (epoch)
        T_mult: 周期倍增因子 (默认 1)
        eta_min: 最小学习率 (默认 0)
        last_epoch: 上次 epoch (默认 -1)
        
    Example:
        >>> # 周期: 10, 20, 40, 80, ...
        >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, got {T_mult}")
            
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0  # 当前周期长度
        self.T_cur = 0  # 当前周期内的 epoch
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
        
    def step(self, epoch: Optional[int] = None):
        """
        更新学习率
        
        Args:
            epoch: 当前 epoch (可选)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            
            # 检查是否需要重启
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            # 根据 epoch 计算当前周期位置
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, got {epoch}")
                
            if self.T_mult == 1:
                self.T_cur = epoch % self.T_0
                self.T_i = self.T_0
            else:
                n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) // (self.T_mult - 1)
                self.T_i = self.T_0 * self.T_mult ** n
                
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineAnnealingWithMinLR(_LRScheduler):
    """
    带最小学习率比例的余弦退火调度器
    
    支持通过比例或绝对值设置最小学习率。
    
    Args:
        optimizer: 优化器实例
        T_max: 周期长度 (epoch)
        eta_min: 最小学习率绝对值 (默认 0)
        eta_min_ratio: 最小学习率比例 (相对于 base_lr)
        last_epoch: 上次 epoch (默认 -1)
        
    Note:
        eta_min_ratio 优先于 eta_min
        
    Example:
        >>> # 最小学习率为初始的 1%
        >>> scheduler = CosineAnnealingWithMinLR(optimizer, T_max=100, eta_min_ratio=0.01)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        eta_min_ratio: Optional[float] = None,
        last_epoch: int = -1
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, last_epoch)
        
        # 计算每个参数组的最小学习率
        if eta_min_ratio is not None:
            self.eta_mins = [base_lr * eta_min_ratio for base_lr in self.base_lrs]
        else:
            self.eta_mins = [eta_min for _ in self.base_lrs]
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch == 0:
            return self.base_lrs
            
        return [
            eta_min + (base_lr - eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr, eta_min in zip(self.base_lrs, self.eta_mins)
        ]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        return [
            eta_min + (base_lr - eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr, eta_min in zip(self.base_lrs, self.eta_mins)
        ]


# 注册调度器
SCHEDULERS.register('CosineAnnealingLR', CosineAnnealingLR, category='cosine', source=ComponentSource.BUILTIN, force=True)
SCHEDULERS.register('CosineAnnealingWarmRestarts', CosineAnnealingWarmRestarts, category='cosine', source=ComponentSource.BUILTIN, force=True)
SCHEDULERS.register('CosineAnnealingWithMinLR', CosineAnnealingWithMinLR, category='cosine', source=ComponentSource.BUILTIN)
