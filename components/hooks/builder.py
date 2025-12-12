# -*- coding: utf-8 -*-
"""
钩子构建器模块

配置驱动的钩子实例化，支持批量构建和优先级排序。
"""

from typing import Dict, Any, List, Optional, Union
from .base_hook import BaseHook, HookPriority
from registry import HOOKS

# 优先级常量
PRIORITY_HIGHEST = HookPriority.HIGHEST
PRIORITY_VERY_HIGH = HookPriority.VERY_HIGH
PRIORITY_HIGH = HookPriority.HIGH
PRIORITY_ABOVE_NORMAL = HookPriority.ABOVE_NORMAL
PRIORITY_NORMAL = HookPriority.NORMAL
PRIORITY_BELOW_NORMAL = HookPriority.BELOW_NORMAL
PRIORITY_LOW = HookPriority.LOW
PRIORITY_VERY_LOW = HookPriority.VERY_LOW
PRIORITY_LOWEST = HookPriority.LOWEST

# 默认钩子优先级映射
DEFAULT_HOOK_PRIORITIES = {
    'EMAHook': PRIORITY_ABOVE_NORMAL,
    'LoggerHook': PRIORITY_NORMAL,
    'TensorBoardHook': PRIORITY_NORMAL,
    'WandbHook': PRIORITY_NORMAL,
    'MLflowHook': PRIORITY_NORMAL,
    'CheckpointHook': PRIORITY_BELOW_NORMAL,
    'EvalHook': PRIORITY_LOW,
    'DistEvalHook': PRIORITY_LOW,
    'EarlyStoppingHook': PRIORITY_VERY_LOW,
    'ProfilerHook': PRIORITY_VERY_HIGH,
    'MemoryProfilerHook': PRIORITY_VERY_HIGH,
    'IterTimerHook': PRIORITY_VERY_HIGH,
    'VisualizationHook': PRIORITY_LOW,
    'DetVisualizationHook': PRIORITY_LOW,
    'SegVisualizationHook': PRIORITY_LOW,
}


def build_hook(config: Dict[str, Any]) -> BaseHook:
    """
    根据配置构建单个钩子
    
    Args:
        config: 钩子配置字典
            - type: 钩子类型 (必填)
            - priority: 优先级 (可选)
            - 其他钩子特定参数
            
    Returns:
        构建的钩子实例
        
    Raises:
        KeyError: 配置缺少 type 字段
        
    Example:
        >>> config = {
        ...     'type': 'CheckpointHook',
        ...     'interval': 1,
        ...     'max_keep_ckpts': 3
        ... }
        >>> hook = build_hook(config)
    """
    config = config.copy()
    
    hook_type = config.pop('type', None)
    if hook_type is None:
        raise KeyError("Config must contain 'type' field")
        
    # 获取优先级
    priority = config.pop('priority', None)
    if priority is None:
        priority = DEFAULT_HOOK_PRIORITIES.get(hook_type, PRIORITY_NORMAL)
        
    # 获取钩子类
    hook_cls = HOOKS.get(hook_type)
    
    # 构建钩子实例
    hook = hook_cls(**config)
    
    # 设置优先级
    hook.priority = priority
    
    return hook


def build_hooks(configs: List[Dict[str, Any]], sort: bool = True) -> List[BaseHook]:
    """
    根据配置列表批量构建钩子
    
    Args:
        configs: 钩子配置列表
        sort: 是否按优先级排序
        
    Returns:
        钩子实例列表（可选排序）
        
    Example:
        >>> configs = [
        ...     {'type': 'LoggerHook', 'interval': 50},
        ...     {'type': 'CheckpointHook', 'interval': 1},
        ...     {'type': 'EMAHook', 'momentum': 0.9999}
        ... ]
        >>> hooks = build_hooks(configs)
    """
    hooks = [build_hook(cfg) for cfg in configs]
    
    if sort:
        hooks = sort_hooks(hooks)
        
    return hooks


def sort_hooks(hooks: List[BaseHook]) -> List[BaseHook]:
    """
    按优先级排序钩子列表
    
    优先级值越小越先执行。
    
    Args:
        hooks: 钩子列表
        
    Returns:
        排序后的钩子列表
    """
    return sorted(hooks, key=lambda h: h.priority)


def get_default_hooks(
    checkpoint: bool = True,
    logger: bool = True,
    timer: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    获取默认钩子配置集
    
    Args:
        checkpoint: 是否包含检查点钩子
        logger: 是否包含日志钩子
        timer: 是否包含计时钩子
        **kwargs: 额外的钩子配置覆盖
        
    Returns:
        默认钩子配置列表
        
    Example:
        >>> configs = get_default_hooks(checkpoint=True, logger=True)
        >>> hooks = build_hooks(configs)
    """
    default_configs = []
    
    if timer:
        timer_config = {
            'type': 'IterTimerHook',
            'interval': 50,
            'log_eta': True,
        }
        timer_config.update(kwargs.get('timer_config', {}))
        default_configs.append(timer_config)
        
    if logger:
        logger_config = {
            'type': 'LoggerHook',
            'interval': 50,
            'by_epoch': True,
        }
        logger_config.update(kwargs.get('logger_config', {}))
        default_configs.append(logger_config)
        
    if checkpoint:
        checkpoint_config = {
            'type': 'CheckpointHook',
            'interval': 1,
            'by_epoch': True,
            'save_best': True,
            'max_keep_ckpts': 3,
        }
        checkpoint_config.update(kwargs.get('checkpoint_config', {}))
        default_configs.append(checkpoint_config)
        
    return default_configs


def get_hook_info(hook: BaseHook) -> Dict[str, Any]:
    """
    获取钩子信息
    
    Args:
        hook: 钩子实例
        
    Returns:
        钩子信息字典
    """
    return {
        'type': type(hook).__name__,
        'priority': hook.priority,
        'triggered_stages': hook.get_triggered_stages(),
    }
