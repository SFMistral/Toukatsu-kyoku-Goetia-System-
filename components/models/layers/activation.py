# -*- coding: utf-8 -*-
"""
激活函数模块

提供各种激活函数及构建工具。
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 激活函数注册表
ACTIVATIONS = {}


def register_activation(name: str):
    """激活函数注册装饰器"""
    def decorator(cls):
        ACTIVATIONS[name] = cls
        return cls
    return decorator


class Swish(nn.Module):
    """Swish/SiLU激活函数: x * sigmoid(x)"""
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x, inplace=self.inplace)


class Mish(nn.Module):
    """Mish激活函数: x * tanh(softplus(x))"""
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x, inplace=self.inplace)


class HardSwish(nn.Module):
    """Hard Swish激活函数"""
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardswish(x, inplace=self.inplace)


class HardSigmoid(nn.Module):
    """Hard Sigmoid激活函数"""
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardsigmoid(x, inplace=self.inplace)


# 注册内置激活函数
register_activation('ReLU')(nn.ReLU)
register_activation('LeakyReLU')(nn.LeakyReLU)
register_activation('ReLU6')(nn.ReLU6)
register_activation('PReLU')(nn.PReLU)
register_activation('SiLU')(nn.SiLU)
register_activation('Swish')(Swish)
register_activation('GELU')(nn.GELU)
register_activation('Mish')(Mish)
register_activation('HardSwish')(HardSwish)
register_activation('HardSigmoid')(HardSigmoid)
register_activation('Sigmoid')(nn.Sigmoid)
register_activation('Tanh')(nn.Tanh)


def build_activation(cfg: Optional[Dict[str, Any]]) -> Optional[nn.Module]:
    """
    根据配置构建激活函数
    
    Args:
        cfg: 激活函数配置，如 dict(type='ReLU', inplace=True)
        
    Returns:
        激活函数模块，cfg为None时返回None
    """
    if cfg is None:
        return None
        
    cfg = cfg.copy()
    act_type = cfg.pop('type')
    
    if act_type not in ACTIVATIONS:
        raise KeyError(f"Activation '{act_type}' not found. Available: {list(ACTIVATIONS.keys())}")
        
    return ACTIVATIONS[act_type](**cfg)
