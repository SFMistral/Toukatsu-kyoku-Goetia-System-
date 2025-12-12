# -*- coding: utf-8 -*-
"""
学习率预热策略模块

提供 LinearWarmup、ExponentialWarmup、ConstantWarmup 等预热策略。
"""

import math
from typing import List, Optional, Any, Dict
import torch
from torch.optim.lr_scheduler import _LRScheduler
from registry import SCHEDULERS, ComponentSource


class LinearWarmup(_LRScheduler):
    """
    线性预热学习率调度器
    
    学习率从 base_lr * warmup_ratio 线性增长到 base_lr。
    
    学习率公式:
        lr = base_lr * (warmup_ratio + (1 - warmup_ratio) * iter / warmup_iters)
    
    Args:
        optimizer: 优化器实例
        warmup_iters: 预热迭代数
        warmup_ratio: 起始学习率比例 (默认 0.001)
        by_epoch: 按 epoch 预热 (默认 False)
        last_epoch: 上次 epoch (默认 -1)
        
    Example:
        >>> scheduler = LinearWarmup(optimizer, warmup_iters=500, warmup_ratio=0.001)
        >>> for iter in range(1000):
        ...     train_step(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        warmup_ratio: float = 0.001,
        by_epoch: bool = False,
        last_epoch: int = -1
    ):
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.by_epoch = by_epoch
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch >= self.warmup_iters:
            return self.base_lrs
            
        factor = self.warmup_ratio + (1 - self.warmup_ratio) * self.last_epoch / self.warmup_iters
        return [base_lr * factor for base_lr in self.base_lrs]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        if self.last_epoch >= self.warmup_iters:
            return self.base_lrs
            
        factor = self.warmup_ratio + (1 - self.warmup_ratio) * self.last_epoch / self.warmup_iters
        return [base_lr * factor for base_lr in self.base_lrs]


class ExponentialWarmup(_LRScheduler):
    """
    指数预热学习率调度器
    
    学习率从 base_lr * warmup_ratio 指数增长到 base_lr。
    
    学习率公式:
        lr = base_lr * warmup_ratio ^ (1 - iter / warmup_iters)
    
    Args:
        optimizer: 优化器实例
        warmup_iters: 预热迭代数
        warmup_ratio: 起始学习率比例 (默认 0.001)
        last_epoch: 上次 epoch (默认 -1)
        
    Example:
        >>> scheduler = ExponentialWarmup(optimizer, warmup_iters=500, warmup_ratio=0.001)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        warmup_ratio: float = 0.001,
        last_epoch: int = -1
    ):
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch >= self.warmup_iters:
            return self.base_lrs
            
        factor = self.warmup_ratio ** (1 - self.last_epoch / self.warmup_iters)
        return [base_lr * factor for base_lr in self.base_lrs]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        if self.last_epoch >= self.warmup_iters:
            return self.base_lrs
            
        factor = self.warmup_ratio ** (1 - self.last_epoch / self.warmup_iters)
        return [base_lr * factor for base_lr in self.base_lrs]


class ConstantWarmup(_LRScheduler):
    """
    常数预热学习率调度器
    
    预热期间保持常数学习率，预热结束后恢复到 base_lr。
    
    学习率:
        lr = base_lr * warmup_ratio  (iter < warmup_iters)
        lr = base_lr                 (iter >= warmup_iters)
    
    Args:
        optimizer: 优化器实例
        warmup_iters: 预热迭代数
        warmup_ratio: 常数学习率比例 (默认 0.001)
        last_epoch: 上次 epoch (默认 -1)
        
    Example:
        >>> scheduler = ConstantWarmup(optimizer, warmup_iters=500, warmup_ratio=0.1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        warmup_ratio: float = 0.001,
        last_epoch: int = -1
    ):
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch >= self.warmup_iters:
            return self.base_lrs
            
        return [base_lr * self.warmup_ratio for base_lr in self.base_lrs]
    
    def _get_closed_form_lr(self) -> List[float]:
        """闭式计算学习率"""
        if self.last_epoch >= self.warmup_iters:
            return self.base_lrs
            
        return [base_lr * self.warmup_ratio for base_lr in self.base_lrs]


class WarmupScheduler:
    """
    预热包装器
    
    包装任意调度器添加预热功能，预热结束后切换到主调度器。
    支持预热按 iter、主调度按 epoch 的混合模式。
    
    Args:
        optimizer: 优化器实例
        scheduler: 主调度器实例
        warmup_type: 预热类型 ('linear', 'exponential', 'constant')
        warmup_iters: 预热迭代数
        warmup_ratio: 起始学习率比例 (默认 0.001)
        warmup_by_epoch: 预热按 epoch (默认 False)
        
    Example:
        >>> main_scheduler = CosineAnnealingLR(optimizer, T_max=100)
        >>> scheduler = WarmupScheduler(
        ...     optimizer=optimizer,
        ...     scheduler=main_scheduler,
        ...     warmup_type='linear',
        ...     warmup_iters=500,
        ...     warmup_ratio=0.001
        ... )
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: _LRScheduler,
        warmup_type: str = 'linear',
        warmup_iters: int = 500,
        warmup_ratio: float = 0.001,
        warmup_by_epoch: bool = False
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_type = warmup_type
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_iter = 0
        self.warmup_finished = False
        
    def _get_warmup_factor(self) -> float:
        """计算预热因子"""
        progress = min(self.current_iter / self.warmup_iters, 1.0)
        
        if self.warmup_type == 'linear':
            return self.warmup_ratio + (1 - self.warmup_ratio) * progress
        elif self.warmup_type == 'exponential':
            return self.warmup_ratio ** (1 - progress)
        elif self.warmup_type == 'constant':
            return self.warmup_ratio if progress < 1.0 else 1.0
        else:
            raise ValueError(f"Unknown warmup type: {self.warmup_type}")
            
    def step(self, epoch: Optional[int] = None):
        """
        更新学习率
        
        Args:
            epoch: 当前 epoch (可选)
        """
        self.current_iter += 1
        
        if self.current_iter <= self.warmup_iters:
            # 预热阶段
            factor = self._get_warmup_factor()
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * factor
        else:
            # 主调度器阶段
            if not self.warmup_finished:
                # 恢复初始学习率
                for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    param_group['lr'] = base_lr
                self.warmup_finished = True
                
            self.scheduler.step(epoch)
            
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def get_last_lr(self) -> List[float]:
        """获取最后一次学习率"""
        return self.get_lr()
        
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'scheduler': self.scheduler.state_dict(),
            'current_iter': self.current_iter,
            'warmup_finished': self.warmup_finished,
            'base_lrs': self.base_lrs
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.current_iter = state_dict['current_iter']
        self.warmup_finished = state_dict['warmup_finished']
        self.base_lrs = state_dict['base_lrs']


# 注册调度器
SCHEDULERS.register('LinearWarmup', LinearWarmup, category='warmup', source=ComponentSource.BUILTIN)
SCHEDULERS.register('ExponentialWarmup', ExponentialWarmup, category='warmup', source=ComponentSource.BUILTIN)
SCHEDULERS.register('ConstantWarmup', ConstantWarmup, category='warmup', source=ComponentSource.BUILTIN)
SCHEDULERS.register('WarmupScheduler', WarmupScheduler, category='warmup', source=ComponentSource.BUILTIN, force=True)
