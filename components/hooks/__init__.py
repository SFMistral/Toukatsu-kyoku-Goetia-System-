# -*- coding: utf-8 -*-
"""
Hooks 模块 - 训练钩子组件

提供训练过程的扩展机制，通过钩子函数在训练生命周期的关键节点注入自定义逻辑。
"""

from .base_hook import BaseHook, HookPriority
from .builder import (
    build_hook,
    build_hooks,
    get_default_hooks,
    PRIORITY_HIGHEST,
    PRIORITY_VERY_HIGH,
    PRIORITY_HIGH,
    PRIORITY_ABOVE_NORMAL,
    PRIORITY_NORMAL,
    PRIORITY_BELOW_NORMAL,
    PRIORITY_LOW,
    PRIORITY_VERY_LOW,
    PRIORITY_LOWEST,
)
from .checkpoint_hook import CheckpointHook
from .logger_hook import LoggerHook, TensorBoardHook, WandbHook, MLflowHook
from .eval_hook import EvalHook, DistEvalHook
from .early_stopping_hook import EarlyStoppingHook
from .ema_hook import EMAHook
from .profiler_hook import ProfilerHook, MemoryProfilerHook, IterTimerHook
from .visualization_hook import VisualizationHook, DetVisualizationHook, SegVisualizationHook

from registry import HOOKS

__all__ = [
    # 基类
    'BaseHook',
    'HookPriority',
    # 构建器
    'build_hook',
    'build_hooks',
    'get_default_hooks',
    # 优先级常量
    'PRIORITY_HIGHEST',
    'PRIORITY_VERY_HIGH',
    'PRIORITY_HIGH',
    'PRIORITY_ABOVE_NORMAL',
    'PRIORITY_NORMAL',
    'PRIORITY_BELOW_NORMAL',
    'PRIORITY_LOW',
    'PRIORITY_VERY_LOW',
    'PRIORITY_LOWEST',
    # 钩子类
    'CheckpointHook',
    'LoggerHook',
    'TensorBoardHook',
    'WandbHook',
    'MLflowHook',
    'EvalHook',
    'DistEvalHook',
    'EarlyStoppingHook',
    'EMAHook',
    'ProfilerHook',
    'MemoryProfilerHook',
    'IterTimerHook',
    'VisualizationHook',
    'DetVisualizationHook',
    'SegVisualizationHook',
    # 注册器
    'HOOKS',
]
