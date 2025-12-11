# -*- coding: utf-8 -*-
"""
权重初始化模块

提供各种权重初始化方法。
"""

from typing import Optional, Union, List, Dict, Any
import math
import torch
import torch.nn as nn


def constant_init(module: nn.Module, val: float, bias: float = 0.0):
    """常数初始化"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(
    module: nn.Module,
    gain: float = 1.0,
    bias: float = 0.0,
    distribution: str = 'normal'
):
    """Xavier初始化"""
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = 'fan_out',
    nonlinearity: str = 'relu',
    bias: float = 0.0,
    distribution: str = 'normal'
):
    """Kaiming初始化"""
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(
    module: nn.Module,
    mean: float = 0.0,
    std: float = 1.0,
    bias: float = 0.0
):
    """正态分布初始化"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(
    module: nn.Module,
    a: float = 0.0,
    b: float = 1.0,
    bias: float = 0.0
):
    """均匀分布初始化"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(
    module: nn.Module,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    bias: float = 0.0
):
    """截断正态分布初始化"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.trunc_normal_(module.weight, mean, std, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob: float) -> float:
    """基于先验概率计算偏置初始值"""
    return -math.log((1 - prior_prob) / prior_prob)


def init_weights(
    module: nn.Module,
    init_cfg: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
):
    """
    根据配置初始化模块权重
    
    Args:
        module: 要初始化的模块
        init_cfg: 初始化配置，支持以下格式：
            - dict(type='Kaiming', layer='Conv2d')
            - dict(type='Pretrained', checkpoint='path/to/weights.pth')
            - [dict(...), dict(...)]  # 多个配置
    """
    if init_cfg is None:
        # 默认初始化
        _default_init(module)
        return
        
    if isinstance(init_cfg, list):
        for cfg in init_cfg:
            init_weights(module, cfg)
        return
        
    init_type = init_cfg.get('type', 'Kaiming')
    
    if init_type == 'Pretrained':
        # 加载预训练权重
        checkpoint = init_cfg.get('checkpoint')
        if checkpoint:
            from .ckpt_convert import load_checkpoint
            load_checkpoint(module, checkpoint, strict=False)
        return
        
    # 获取要初始化的层类型
    layer_types = init_cfg.get('layer', None)
    if layer_types is not None:
        if isinstance(layer_types, str):
            layer_types = [layer_types]
        layer_types = tuple(_get_layer_class(t) for t in layer_types)
        
    # 初始化函数映射
    init_funcs = {
        'Constant': lambda m: constant_init(m, init_cfg.get('val', 0)),
        'Xavier': lambda m: xavier_init(
            m, gain=init_cfg.get('gain', 1),
            distribution=init_cfg.get('distribution', 'normal')
        ),
        'Kaiming': lambda m: kaiming_init(
            m, a=init_cfg.get('a', 0),
            mode=init_cfg.get('mode', 'fan_out'),
            nonlinearity=init_cfg.get('nonlinearity', 'relu'),
            distribution=init_cfg.get('distribution', 'normal')
        ),
        'Normal': lambda m: normal_init(
            m, mean=init_cfg.get('mean', 0),
            std=init_cfg.get('std', 0.01)
        ),
        'Uniform': lambda m: uniform_init(
            m, a=init_cfg.get('a', 0),
            b=init_cfg.get('b', 1)
        ),
        'TruncNormal': lambda m: trunc_normal_init(
            m, mean=init_cfg.get('mean', 0),
            std=init_cfg.get('std', 0.02)
        ),
    }
    
    init_func = init_funcs.get(init_type)
    if init_func is None:
        raise ValueError(f"Unknown init type: {init_type}")
        
    for m in module.modules():
        if layer_types is None or isinstance(m, layer_types):
            init_func(m)


def _default_init(module: nn.Module):
    """默认初始化策略"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            trunc_normal_init(m, std=0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            constant_init(m, val=1, bias=0)


def _get_layer_class(layer_name: str):
    """根据名称获取层类"""
    layer_map = {
        'Conv2d': nn.Conv2d,
        'Conv1d': nn.Conv1d,
        'Conv3d': nn.Conv3d,
        'Linear': nn.Linear,
        'BatchNorm2d': nn.BatchNorm2d,
        'BatchNorm1d': nn.BatchNorm1d,
        'GroupNorm': nn.GroupNorm,
        'LayerNorm': nn.LayerNorm,
    }
    if layer_name not in layer_map:
        raise ValueError(f"Unknown layer type: {layer_name}")
    return layer_map[layer_name]
