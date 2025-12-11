# -*- coding: utf-8 -*-
"""
检查点格式转换模块

提供不同框架预训练权重的转换能力。
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import re
import torch
import torch.nn as nn


def load_checkpoint(
    model: nn.Module,
    checkpoint: str,
    map_location: str = 'cpu',
    strict: bool = True,
) -> Dict[str, Any]:
    """
    加载检查点
    
    Args:
        model: 模型
        checkpoint: 检查点路径或URL
        map_location: 设备映射
        strict: 是否严格匹配
        
    Returns:
        加载信息
    """
    if checkpoint.startswith(('http://', 'https://')):
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location=map_location)
    else:
        state_dict = torch.load(checkpoint, map_location=map_location)

    # 处理嵌套结构
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
        
    # 自动转换
    state_dict = auto_convert(state_dict, model)
    
    # 加载权重
    result = model.load_state_dict(state_dict, strict=strict)
    
    return {
        'missing_keys': result.missing_keys if hasattr(result, 'missing_keys') else [],
        'unexpected_keys': result.unexpected_keys if hasattr(result, 'unexpected_keys') else [],
    }


def convert_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    mapping: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """
    通用检查点转换
    
    Args:
        state_dict: 原始状态字典
        mapping: 键名映射 {old_key: new_key}
        
    Returns:
        转换后的状态字典
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = mapping.get(key, key)
        new_state_dict[new_key] = value
    return new_state_dict


def convert_from_torchvision(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """从torchvision格式转换"""
    # torchvision ResNet键名映射
    mapping = {}
    for key in state_dict.keys():
        new_key = key
        # layer1 -> stages.0
        new_key = re.sub(r'^layer(\d+)', lambda m: f'stages.{int(m.group(1))-1}', new_key)
        # conv1 -> stem.conv
        if key.startswith('conv1'):
            new_key = key.replace('conv1', 'stem.conv')
        elif key.startswith('bn1'):
            new_key = key.replace('bn1', 'stem.bn')
        mapping[key] = new_key
    return convert_checkpoint(state_dict, mapping)


def convert_from_timm(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """从timm格式转换"""
    # timm通常使用类似的命名，可能需要处理一些特殊情况
    mapping = {}
    for key in state_dict.keys():
        new_key = key
        # 处理patch_embed
        if 'patch_embed' in key:
            new_key = key.replace('patch_embed.proj', 'patch_embed.projection')
        mapping[key] = new_key
    return convert_checkpoint(state_dict, mapping)


def auto_convert(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    自动检测并转换检查点格式
    
    Args:
        state_dict: 原始状态字典
        model: 目标模型
        
    Returns:
        转换后的状态字典
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    # 如果完全匹配，直接返回
    if model_keys == ckpt_keys:
        return state_dict
        
    # 尝试不同的转换策略
    converters = [
        convert_from_torchvision,
        convert_from_timm,
    ]
    
    best_match = 0
    best_state_dict = state_dict
    
    for converter in converters:
        try:
            converted = converter(state_dict, model)
            match_count = len(set(converted.keys()) & model_keys)
            if match_count > best_match:
                best_match = match_count
                best_state_dict = converted
        except Exception:
            continue
            
    return best_state_dict


def remap_keys(
    state_dict: Dict[str, torch.Tensor],
    mapping: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """键名重映射"""
    return convert_checkpoint(state_dict, mapping)


def filter_by_prefix(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    keep: bool = True,
) -> Dict[str, torch.Tensor]:
    """按前缀过滤"""
    if keep:
        return {k: v for k, v in state_dict.items() if k.startswith(prefix)}
    return {k: v for k, v in state_dict.items() if not k.startswith(prefix)}


def strip_prefix(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """移除前缀"""
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def add_prefix(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """添加前缀"""
    return {prefix + k: v for k, v in state_dict.items()}
