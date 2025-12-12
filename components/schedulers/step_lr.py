# -*- coding: utf-8 -*-
"""
步进式学习率调度模块

提供 StepLR、MultiStepLR、ExponentialLR 等步进式学习率调度器。
"""

from typing import Iterator, List, Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler
from registry import SCHEDULERS, ComponentSource


class StepLR(_LRScheduler):
    """
    固定步长衰减学习率调度器
    
    每隔 step_size 个 epoch，学习率乘以 gamma。
    
    学习率公式:
        lr = base_lr * gamma ^ (epoch // step_size)
    
    Args:
        optimizer: 优化器实例
        step_size: 衰减步长 (epoch)
        gamma: 衰减因子 (默认 0.1)
        last_epoch: 上次 epoch (默认 -1)
        
    Example:
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]


class MultiStepLR(_LRScheduler):
    """
    多里程碑衰减学习率调度器
    
    在指定的里程碑 epoch 处，学习率乘以 gamma。
    
    学习率公式:
        lr = base_lr * gamma ^ (已过里程碑数量)
    
    Args:
        optimizer: 优化器实例
        milestones: 衰减里程碑列表 (epoch)
        gamma: 衰减因子 (默认 0.1)
        last_epoch: 上次 epoch (默认 -1)
        
    Example:
        >>> scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        milestones_passed = sum(1 for m in self.milestones if m <= self.last_epoch)
        return [
            base_lr * self.gamma ** milestones_passed
            for base_lr in self.base_lrs
        ]


class ExponentialLR(_LRScheduler):
    """
    指数衰减学习率调度器
    
    每个 epoch，学习率乘以 gamma。
    
    学习率公式:
        lr = base_lr * gamma ^ epoch
    
    Args:
        optimizer: 优化器实例
        gamma: 每 epoch 衰减因子
        last_epoch: 上次 epoch (默认 -1)
        
    Example:
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        last_epoch: int = -1
    ):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        return [
            base_lr * self.gamma ** self.last_epoch
            for base_lr in self.base_lrs
        ]


# 注册调度器
SCHEDULERS.register('StepLR', StepLR, category='step', source=ComponentSource.BUILTIN, force=True)
SCHEDULERS.register('MultiStepLR', MultiStepLR, category='step', source=ComponentSource.BUILTIN, force=True)
SCHEDULERS.register('ExponentialLR', ExponentialLR, category='exponential', source=ComponentSource.BUILTIN, force=True)
