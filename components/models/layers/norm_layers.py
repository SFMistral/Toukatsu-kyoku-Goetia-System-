# -*- coding: utf-8 -*-
"""
归一化层模块

提供各种归一化层及构建工具。
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

# 归一化层注册表
NORM_LAYERS = {}


def register_norm(name: str):
    """归一化层注册装饰器"""
    def decorator(cls):
        NORM_LAYERS[name] = cls
        return cls
    return decorator


class LayerNorm2d(nn.LayerNorm):
    """适用于CNN的2D LayerNorm (channels_first格式)"""
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__(num_channels, eps=eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H, W, C) -> LayerNorm -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


# 注册内置归一化层
register_norm('BN')(nn.BatchNorm2d)
register_norm('BN1d')(nn.BatchNorm1d)
register_norm('BN2d')(nn.BatchNorm2d)
register_norm('BN3d')(nn.BatchNorm3d)
register_norm('SyncBN')(nn.SyncBatchNorm)
register_norm('GN')(nn.GroupNorm)
register_norm('LN')(nn.LayerNorm)
register_norm('LN2d')(LayerNorm2d)
register_norm('IN')(nn.InstanceNorm2d)
register_norm('IN1d')(nn.InstanceNorm1d)
register_norm('IN2d')(nn.InstanceNorm2d)
register_norm('IN3d')(nn.InstanceNorm3d)


def build_norm_layer(
    cfg: Dict[str, Any],
    num_features: int,
    postfix: str = ''
) -> Tuple[str, nn.Module]:
    """
    根据配置构建归一化层
    
    Args:
        cfg: 归一化层配置，如 dict(type='BN', requires_grad=True)
        num_features: 特征通道数
        postfix: 层名称后缀
        
    Returns:
        (层名称, 归一化层模块)
    """
    cfg = cfg.copy()
    norm_type = cfg.pop('type')
    requires_grad = cfg.pop('requires_grad', True)
    
    if norm_type not in NORM_LAYERS:
        raise KeyError(f"Norm layer '{norm_type}' not found. Available: {list(NORM_LAYERS.keys())}")
    
    norm_cls = NORM_LAYERS[norm_type]
    
    # 根据不同类型设置参数
    if norm_type in ('GN',):
        num_groups = cfg.pop('num_groups', 32)
        # 确保num_groups不超过num_features
        num_groups = min(num_groups, num_features)
        layer = norm_cls(num_groups, num_features, **cfg)
    elif norm_type in ('LN',):
        layer = norm_cls(num_features, **cfg)
    elif norm_type in ('LN2d',):
        layer = norm_cls(num_features, **cfg)
    else:
        layer = norm_cls(num_features, **cfg)
    
    # 设置requires_grad
    for param in layer.parameters():
        param.requires_grad = requires_grad
        
    # 生成层名称
    abbr_map = {
        'BN': 'bn', 'BN1d': 'bn', 'BN2d': 'bn', 'BN3d': 'bn',
        'SyncBN': 'bn', 'GN': 'gn', 'LN': 'ln', 'LN2d': 'ln',
        'IN': 'in', 'IN1d': 'in', 'IN2d': 'in', 'IN3d': 'in',
    }
    name = abbr_map.get(norm_type, 'norm') + str(postfix)
    
    return name, layer
