# -*- coding: utf-8 -*-
"""
Schedulers 模块 - 学习率调度组件库

提供多种学习率调度策略，支持预热机制、周期性调度、自定义调度曲线等功能。
可按 epoch 或 iteration 更新学习率，通过注册器管理实现配置驱动的调度器构建。
"""

from .builder import (
    build_scheduler,
    build_scheduler_with_warmup,
    SequentialScheduler,
    ChainedScheduler
)
from .step_lr import StepLR, MultiStepLR, ExponentialLR
from .cosine_lr import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CosineAnnealingWithMinLR
)
from .warmup import (
    LinearWarmup,
    ExponentialWarmup,
    ConstantWarmup,
    WarmupScheduler
)
from .poly_lr import PolyLR, LinearLR
from .onecycle_lr import OneCycleLR, CyclicLR

# 从 registry 导入调度器注册器
from registry import SCHEDULERS

__version__ = "1.0.0"
__author__ = "AI Training Framework Team"

__all__ = [
    # 构建函数
    "build_scheduler",
    "build_scheduler_with_warmup",
    
    # 组合调度器
    "SequentialScheduler",
    "ChainedScheduler",
    
    # Step 系列
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    
    # Cosine 系列
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CosineAnnealingWithMinLR",
    
    # Warmup 系列
    "LinearWarmup",
    "ExponentialWarmup",
    "ConstantWarmup",
    "WarmupScheduler",
    
    # Poly 系列
    "PolyLR",
    "LinearLR",
    
    # OneCycle 系列
    "OneCycleLR",
    "CyclicLR",
    
    # 注册器
    "SCHEDULERS",
]
