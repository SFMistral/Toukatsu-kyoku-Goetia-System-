# -*- coding: utf-8 -*-
"""
OneCycle 学习率调度模块

提供 OneCycleLR、CyclicLR 等周期性学习率调度器。
"""

import math
from typing import List, Optional, Callable, Union
import torch
from torch.optim.lr_scheduler import _LRScheduler
from registry import SCHEDULERS, ComponentSource


class OneCycleLR(_LRScheduler):
    """
    OneCycle 学习率调度器
    
    实现 1cycle 学习率策略，支持两阶段和三阶段模式。
    
    两阶段模式:
        1. 上升阶段: base_lr → max_lr
        2. 下降阶段: max_lr → min_lr
        
    三阶段模式:
        1. 上升阶段: base_lr → max_lr
        2. 下降阶段: max_lr → base_lr
        3. 退火阶段: base_lr → min_lr
    
    Args:
        optimizer: 优化器实例
        max_lr: 最大学习率
        total_steps: 总步数
        pct_start: 上升阶段比例 (默认 0.3)
        anneal_strategy: 退火策略 'cos' 或 'linear' (默认 'cos')
        div_factor: 初始学习率因子 (默认 25)
        final_div_factor: 最终学习率因子 (默认 10000)
        three_phase: 是否三阶段 (默认 False)
        last_epoch: 上次 epoch (默认 -1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        three_phase: bool = False,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase

        # 处理 max_lr
        if isinstance(max_lr, (list, tuple)):
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
            
        # 计算初始和最终学习率
        self.initial_lrs = [lr / div_factor for lr in self.max_lrs]
        self.final_lrs = [lr / (div_factor * final_div_factor) for lr in self.max_lrs]
        
        # 计算阶段边界
        if three_phase:
            self.step_up = int(total_steps * pct_start)
            self.step_down = int(total_steps * (1 - pct_start) * 0.5) + self.step_up
            self.step_anneal = total_steps
        else:
            self.step_up = int(total_steps * pct_start)
            self.step_down = total_steps
            
        super().__init__(optimizer, last_epoch)
        
        # 设置初始学习率
        for param_group, lr in zip(self.optimizer.param_groups, self.initial_lrs):
            param_group['lr'] = lr
            
    def _anneal(self, start: float, end: float, pct: float) -> float:
        """计算退火值"""
        if self.anneal_strategy == 'cos':
            return end + (start - end) * (1 + math.cos(math.pi * pct)) / 2
        elif self.anneal_strategy == 'linear':
            return start + (end - start) * pct
        else:
            raise ValueError(f"Unknown anneal strategy: {self.anneal_strategy}")
            
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        step = self.last_epoch
        
        if self.three_phase:
            if step < self.step_up:
                # 上升阶段
                pct = step / self.step_up
                return [self._anneal(init_lr, max_lr, pct) 
                        for init_lr, max_lr in zip(self.initial_lrs, self.max_lrs)]
            elif step < self.step_down:
                # 下降阶段
                pct = (step - self.step_up) / (self.step_down - self.step_up)
                return [self._anneal(max_lr, init_lr, pct)
                        for init_lr, max_lr in zip(self.initial_lrs, self.max_lrs)]
            else:
                # 退火阶段
                pct = (step - self.step_down) / (self.step_anneal - self.step_down)
                return [self._anneal(init_lr, final_lr, pct)
                        for init_lr, final_lr in zip(self.initial_lrs, self.final_lrs)]
        else:
            if step < self.step_up:
                # 上升阶段
                pct = step / self.step_up
                return [self._anneal(init_lr, max_lr, pct)
                        for init_lr, max_lr in zip(self.initial_lrs, self.max_lrs)]
            else:
                # 下降阶段
                pct = (step - self.step_up) / (self.step_down - self.step_up)
                return [self._anneal(max_lr, final_lr, pct)
                        for max_lr, final_lr in zip(self.max_lrs, self.final_lrs)]


class CyclicLR(_LRScheduler):
    """
    周期性学习率调度器
    
    学习率在 base_lr 和 max_lr 之间周期性变化。
    
    模式说明:
        - triangular: 三角波
        - triangular2: 每周期减半的三角波
        - exp_range: 指数衰减的三角波
    
    Args:
        optimizer: 优化器实例
        base_lr: 基础学习率
        max_lr: 最大学习率
        step_size_up: 上升步数 (默认 2000)
        step_size_down: 下降步数 (默认 None, 等于 step_size_up)
        mode: 模式 (默认 'triangular')
        gamma: exp_range 模式的衰减因子 (默认 1.0)
        scale_fn: 自定义缩放函数 (默认 None)
        cycle_momentum: 是否同步调整动量 (默认 True)
        base_momentum: 基础动量 (默认 0.8)
        max_momentum: 最大动量 (默认 0.9)
        last_epoch: 上次 epoch (默认 -1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: Union[float, List[float]],
        max_lr: Union[float, List[float]],
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable[[int], float]] = None,
        cycle_momentum: bool = True,
        base_momentum: float = 0.8,
        max_momentum: float = 0.9,
        last_epoch: int = -1
    ):
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down else step_size_up
        self.mode = mode
        self.gamma = gamma
        self._scale_fn = scale_fn
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        
        # 处理 base_lr 和 max_lr (在 super().__init__ 之前保存)
        if isinstance(base_lr, (list, tuple)):
            self._base_lrs = list(base_lr)
        else:
            self._base_lrs = [base_lr] * len(optimizer.param_groups)
            
        if isinstance(max_lr, (list, tuple)):
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
            
        # 设置缩放函数
        if scale_fn is None:
            if mode == 'triangular':
                self._scale_fn = lambda x: 1.0
            elif mode == 'triangular2':
                self._scale_fn = lambda x: 1 / (2.0 ** x)
            elif mode == 'exp_range':
                self._scale_fn = lambda x: gamma ** x
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
        super().__init__(optimizer, last_epoch)
        
        # 覆盖 base_lrs (super().__init__ 会从 optimizer 获取)
        self.base_lrs = self._base_lrs
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = 1 + self.last_epoch / (self.step_size_up + self.step_size_down) - cycle
        
        if x <= self.step_size_up / (self.step_size_up + self.step_size_down):
            # 上升阶段
            scale = x * (self.step_size_up + self.step_size_down) / self.step_size_up
        else:
            # 下降阶段
            scale = (1 - x) * (self.step_size_up + self.step_size_down) / self.step_size_down
            
        scale_factor = self._scale_fn(cycle - 1)
        
        return [
            base_lr + (max_lr - base_lr) * scale * scale_factor
            for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)
        ]
        
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        super().step(epoch)
        
        # 同步调整动量
        if self.cycle_momentum:
            cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
            x = 1 + self.last_epoch / (self.step_size_up + self.step_size_down) - cycle
            
            if x <= self.step_size_up / (self.step_size_up + self.step_size_down):
                scale = x * (self.step_size_up + self.step_size_down) / self.step_size_up
            else:
                scale = (1 - x) * (self.step_size_up + self.step_size_down) / self.step_size_down
                
            # 动量与学习率反向变化
            momentum = self.max_momentum - (self.max_momentum - self.base_momentum) * scale
            
            for param_group in self.optimizer.param_groups:
                if 'momentum' in param_group:
                    param_group['momentum'] = momentum
                elif 'betas' in param_group:
                    param_group['betas'] = (momentum, param_group['betas'][1])


# 注册调度器
SCHEDULERS.register('OneCycleLR', OneCycleLR, category='cycle', source=ComponentSource.BUILTIN, force=True)
SCHEDULERS.register('CyclicLR', CyclicLR, category='cycle', source=ComponentSource.BUILTIN, force=True)
