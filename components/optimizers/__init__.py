# -*- coding: utf-8 -*-
"""
Optimizers 模块 - 优化器组件库

提供常用优化器的封装与扩展，支持参数分组、层级学习率衰减、权重衰减过滤等高级功能。
"""

from .builder import build_optimizer, build_param_groups
from .sgd import SGD, SGDP, NesterovSGD
from .adam import Adam, AdamP, NAdam, RAdam, Adagrad, Adadelta
from .adamw import AdamW, FusedAdamW, Adam8bit
from .lion import Lion
from .layer_decay import (
    get_layer_decay_params,
    get_num_layers,
    get_layer_id
)

# 从 registry 导入优化器注册器
from registry import OPTIMIZERS

__version__ = "1.0.0"
__author__ = "AI Training Framework Team"

__all__ = [
    # 构建函数
    "build_optimizer",
    "build_param_groups",
    
    # SGD 系列
    "SGD",
    "SGDP",
    "NesterovSGD",
    
    # Adam 系列
    "Adam",
    "AdamP",
    "NAdam",
    "RAdam",
    "Adagrad",
    "Adadelta",
    
    # AdamW 系列
    "AdamW",
    "FusedAdamW",
    "Adam8bit",
    
    # Lion
    "Lion",
    
    # 层级衰减工具
    "get_layer_decay_params",
    "get_num_layers",
    "get_layer_id",
    
    # 注册器
    "OPTIMIZERS",
]
