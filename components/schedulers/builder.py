# -*- coding: utf-8 -*-
"""
调度器构建器模块

配置驱动的调度器实例化，支持自动包装预热策略、组合多个调度器。
"""

from typing import Dict, Any, Optional, List, Union
import torch
from torch.optim.lr_scheduler import _LRScheduler
from registry import SCHEDULERS


def build_scheduler(
    config: Dict[str, Any],
    optimizer: torch.optim.Optimizer
) -> Union[_LRScheduler, 'SequentialScheduler']:
    """
    根据配置构建调度器
    
    Args:
        config: 调度器配置字典
            - type: 调度器类型 (必填)
            - by_epoch: 按 epoch 更新 (默认 True)
            - warmup: 预热配置 (可选)
            - warmup_iters: 预热迭代数 (默认 0)
            - warmup_ratio: 预热起始比例 (默认 0.001)
            - warmup_by_epoch: 预热按 epoch (默认 False)
            - begin: 开始 epoch/iter (默认 0)
            - end: 结束 epoch/iter (默认 -1)
            - 其他调度器特定参数
        optimizer: PyTorch 优化器
        
    Returns:
        构建的调度器实例
        
    Example:
        >>> config = {
        ...     'type': 'CosineAnnealingLR',
        ...     'T_max': 100,
        ...     'eta_min': 1e-6
        ... }
        >>> scheduler = build_scheduler(config, optimizer)
    """
    config = config.copy()
    
    # 获取调度器类型
    scheduler_type = config.pop('type', None)
    if scheduler_type is None:
        raise KeyError("Config must contain 'type' field")

    # 处理组合调度器
    if scheduler_type == 'SequentialScheduler':
        return _build_sequential_scheduler(config, optimizer)
    elif scheduler_type == 'ChainedScheduler':
        return _build_chained_scheduler(config, optimizer)
        
    # 提取预热相关配置
    warmup_config = config.pop('warmup', None)
    warmup_iters = config.pop('warmup_iters', 0)
    warmup_ratio = config.pop('warmup_ratio', 0.001)
    warmup_by_epoch = config.pop('warmup_by_epoch', False)
    
    # 提取通用配置
    config.pop('by_epoch', True)
    config.pop('begin', 0)
    config.pop('end', -1)
    
    # 获取调度器类
    scheduler_cls = SCHEDULERS.get(scheduler_type)
    
    # 构建主调度器
    scheduler = scheduler_cls(optimizer, **config)
    
    # 包装预热
    if warmup_config or warmup_iters > 0:
        from .warmup import WarmupScheduler
        
        warmup_type = warmup_config if isinstance(warmup_config, str) else 'linear'
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_type=warmup_type,
            warmup_iters=warmup_iters,
            warmup_ratio=warmup_ratio,
            warmup_by_epoch=warmup_by_epoch
        )
        
    return scheduler


def build_scheduler_with_warmup(
    config: Dict[str, Any],
    optimizer: torch.optim.Optimizer
) -> Union[_LRScheduler, 'WarmupScheduler']:
    """
    构建带预热的调度器
    
    Args:
        config: 调度器配置字典
            - type: 调度器类型 (必填)
            - warmup_type: 预热类型 ('linear', 'exponential', 'constant')
            - warmup_iters: 预热迭代数 (必填)
            - warmup_ratio: 预热起始比例 (默认 0.001)
            - 其他调度器特定参数
        optimizer: PyTorch 优化器
        
    Returns:
        带预热的调度器实例
        
    Example:
        >>> config = {
        ...     'type': 'CosineAnnealingLR',
        ...     'T_max': 100,
        ...     'warmup_type': 'linear',
        ...     'warmup_iters': 500,
        ...     'warmup_ratio': 0.001
        ... }
        >>> scheduler = build_scheduler_with_warmup(config, optimizer)
    """
    config = config.copy()
    
    # 提取预热配置
    warmup_type = config.pop('warmup_type', 'linear')
    warmup_iters = config.pop('warmup_iters', 500)
    warmup_ratio = config.pop('warmup_ratio', 0.001)
    warmup_by_epoch = config.pop('warmup_by_epoch', False)
    
    # 构建主调度器
    scheduler = build_scheduler(config, optimizer)
    
    # 如果已经是 WarmupScheduler，直接返回
    from .warmup import WarmupScheduler
    if isinstance(scheduler, WarmupScheduler):
        return scheduler
        
    # 包装预热
    return WarmupScheduler(
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_type=warmup_type,
        warmup_iters=warmup_iters,
        warmup_ratio=warmup_ratio,
        warmup_by_epoch=warmup_by_epoch
    )


def _build_sequential_scheduler(
    config: Dict[str, Any],
    optimizer: torch.optim.Optimizer
) -> 'SequentialScheduler':
    """构建顺序调度器"""
    schedulers_config = config.get('schedulers', [])
    
    schedulers = []
    for sched_config in schedulers_config:
        scheduler = build_scheduler(sched_config, optimizer)
        schedulers.append(scheduler)
        
    return SequentialScheduler(schedulers)


def _build_chained_scheduler(
    config: Dict[str, Any],
    optimizer: torch.optim.Optimizer
) -> 'ChainedScheduler':
    """构建链式调度器"""
    schedulers_config = config.get('schedulers', [])
    
    schedulers = []
    for sched_config in schedulers_config:
        scheduler = build_scheduler(sched_config, optimizer)
        schedulers.append(scheduler)
        
    return ChainedScheduler(schedulers)


class SequentialScheduler:
    """
    顺序执行多个调度器
    
    按配置的 begin/end 范围顺序执行多个调度器。
    
    Args:
        schedulers: 调度器列表
        
    Example:
        >>> scheduler = SequentialScheduler([
        ...     LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=5),
        ...     CosineAnnealingLR(optimizer, T_max=95)
        ... ])
    """
    
    def __init__(self, schedulers: List[_LRScheduler]):
        self.schedulers = schedulers
        self.current_epoch = 0
        self._current_scheduler_idx = 0
        
        # 从调度器配置中提取 begin/end
        self._scheduler_ranges = []
        current_begin = 0
        for scheduler in schedulers:
            begin = getattr(scheduler, 'begin', current_begin)
            end = getattr(scheduler, 'end', -1)
            self._scheduler_ranges.append((begin, end))
            if end > 0:
                current_begin = end
                
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
            
        # 找到当前应该使用的调度器
        for i, (begin, end) in enumerate(self._scheduler_ranges):
            if begin <= self.current_epoch and (end == -1 or self.current_epoch < end):
                self._current_scheduler_idx = i
                break
                
        # 调用当前调度器
        self.schedulers[self._current_scheduler_idx].step()
        
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return self.schedulers[self._current_scheduler_idx].get_lr()
        
    def get_last_lr(self) -> List[float]:
        """获取最后一次学习率"""
        scheduler = self.schedulers[self._current_scheduler_idx]
        if hasattr(scheduler, 'get_last_lr'):
            return scheduler.get_last_lr()
        return self.get_lr()
        
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'schedulers': [s.state_dict() for s in self.schedulers],
            'current_epoch': self.current_epoch,
            'current_scheduler_idx': self._current_scheduler_idx
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        for s, sd in zip(self.schedulers, state_dict['schedulers']):
            s.load_state_dict(sd)
        self.current_epoch = state_dict['current_epoch']
        self._current_scheduler_idx = state_dict['current_scheduler_idx']


class ChainedScheduler:
    """
    链式调度器
    
    同时应用多个调度器，学习率为所有调度器的乘积效果。
    
    Args:
        schedulers: 调度器列表
        
    Example:
        >>> scheduler = ChainedScheduler([
        ...     LinearWarmup(optimizer, warmup_iters=500),
        ...     CosineAnnealingLR(optimizer, T_max=100)
        ... ])
    """
    
    def __init__(self, schedulers: List[_LRScheduler]):
        self.schedulers = schedulers
        
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        for scheduler in self.schedulers:
            scheduler.step(epoch)
            
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        if self.schedulers:
            return self.schedulers[-1].get_lr()
        return []
        
    def get_last_lr(self) -> List[float]:
        """获取最后一次学习率"""
        if self.schedulers:
            scheduler = self.schedulers[-1]
            if hasattr(scheduler, 'get_last_lr'):
                return scheduler.get_last_lr()
            return self.get_lr()
        return []
        
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'schedulers': [s.state_dict() for s in self.schedulers]
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        for s, sd in zip(self.schedulers, state_dict['schedulers']):
            s.load_state_dict(sd)


def get_scheduler_info(scheduler) -> Dict[str, Any]:
    """
    获取调度器信息
    
    Args:
        scheduler: 调度器实例
        
    Returns:
        调度器信息字典
    """
    info = {
        'type': type(scheduler).__name__,
    }
    
    # 获取当前学习率
    if hasattr(scheduler, 'get_last_lr'):
        info['current_lrs'] = scheduler.get_last_lr()
    elif hasattr(scheduler, 'get_lr'):
        info['current_lrs'] = scheduler.get_lr()
        
    # 获取基础学习率
    if hasattr(scheduler, 'base_lrs'):
        info['base_lrs'] = scheduler.base_lrs
        
    # 获取特定参数
    if hasattr(scheduler, 'T_max'):
        info['T_max'] = scheduler.T_max
    if hasattr(scheduler, 'eta_min'):
        info['eta_min'] = scheduler.eta_min
    if hasattr(scheduler, 'warmup_iters'):
        info['warmup_iters'] = scheduler.warmup_iters
        
    return info


# 注册组合调度器
SCHEDULERS.register('SequentialScheduler', SequentialScheduler, category='combined', source='builtin')
SCHEDULERS.register('ChainedScheduler', ChainedScheduler, category='combined', source='builtin')
