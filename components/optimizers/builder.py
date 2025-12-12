# -*- coding: utf-8 -*-
"""
优化器构建器模块

配置驱动的优化器实例化，支持参数分组、权重衰减过滤、层级学习率设置。
"""

import re
from typing import Dict, Any, Optional, List, Iterator, Union
import torch
import torch.nn as nn
from registry import OPTIMIZERS
from .layer_decay import get_layer_decay_params


# 默认不应用权重衰减的参数模式
DEFAULT_NO_WEIGHT_DECAY_PATTERNS = [
    'bias',
    'LayerNorm.weight',
    'LayerNorm.bias',
    'BatchNorm',
    'bn.weight',
    'bn.bias',
    'norm.weight',
    'norm.bias',
    'pos_embed',
    'cls_token',
    'relative_position_bias_table',
    'absolute_pos_embed',
]


def build_optimizer(
    config: Dict[str, Any],
    model: nn.Module
) -> torch.optim.Optimizer:
    """
    根据配置构建优化器
    
    Args:
        config: 优化器配置字典
            - type: 优化器类型 (必填)
            - lr: 基础学习率 (必填)
            - weight_decay: 权重衰减 (默认 0.0)
            - no_weight_decay_params: 不衰减的参数名模式 (默认 [])
            - param_groups: 自定义参数分组 (可选)
            - layer_decay: 层级衰减配置 (可选)
            - 其他优化器特定参数
        model: PyTorch 模型
        
    Returns:
        构建的优化器实例
        
    Example:
        >>> config = {
        ...     'type': 'AdamW',
        ...     'lr': 0.001,
        ...     'weight_decay': 0.05,
        ...     'betas': [0.9, 0.999]
        ... }
        >>> optimizer = build_optimizer(config, model)
    """
    config = config.copy()
    
    # 获取优化器类型
    optimizer_type = config.pop('type', None)
    if optimizer_type is None:
        raise KeyError("Config must contain 'type' field")
        
    # 获取优化器类
    optimizer_cls = OPTIMIZERS.get(optimizer_type)
    
    # 提取参数分组相关配置
    layer_decay_config = config.pop('layer_decay', None)
    param_groups_config = config.pop('param_groups', None)
    no_wd_params = config.pop('no_weight_decay_params', [])
    
    # 构建参数分组
    if layer_decay_config is not None:
        # 使用层级学习率衰减
        layer_decay_config['base_lr'] = config.get('lr')
        layer_decay_config['weight_decay'] = config.get('weight_decay', 0.0)
        layer_decay_config['no_weight_decay_params'] = no_wd_params
        param_groups = get_layer_decay_params(model, layer_decay_config)
    elif param_groups_config is not None:
        # 使用自定义参数分组
        param_groups = build_param_groups(
            model, 
            config, 
            param_groups_config,
            no_wd_params
        )
    else:
        # 默认分组（区分权重衰减）
        param_groups = _build_default_param_groups(
            model,
            config.get('lr'),
            config.get('weight_decay', 0.0),
            no_wd_params
        )
        
    # 移除已处理的参数，但保留 lr 用于某些优化器的默认值
    base_lr = config.pop('lr', None)
    config.pop('weight_decay', None)
    
    # 某些优化器（如 PyTorch 原生 SGD）需要 lr 参数
    # 检查优化器是否需要 lr 作为位置参数
    import inspect
    sig = inspect.signature(optimizer_cls.__init__)
    params = list(sig.parameters.keys())
    
    # 如果 lr 是必需参数且不在 config 中，添加回去
    if 'lr' in params and base_lr is not None:
        lr_param = sig.parameters.get('lr')
        if lr_param and lr_param.default == inspect.Parameter.empty:
            config['lr'] = base_lr
    
    # 构建优化器
    return optimizer_cls(param_groups, **config)


def build_param_groups(
    model: nn.Module,
    base_config: Dict[str, Any],
    param_groups_config: List[Dict[str, Any]],
    no_wd_params: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    构建自定义参数分组
    
    Args:
        model: PyTorch 模型
        base_config: 基础配置 (包含 lr, weight_decay 等)
        param_groups_config: 参数分组配置列表
            每组配置包含:
            - params: 参数名模式 (字符串或正则)
            - lr_mult: 学习率倍数 (默认 1.0)
            - weight_decay_mult: 权重衰减倍数 (默认 1.0)
        no_wd_params: 不应用权重衰减的参数名模式
        
    Returns:
        参数分组列表
        
    Example:
        >>> param_groups_config = [
        ...     {'params': 'backbone', 'lr_mult': 0.1},
        ...     {'params': 'head', 'lr_mult': 1.0}
        ... ]
        >>> groups = build_param_groups(model, base_config, param_groups_config)
    """
    base_lr = base_config.get('lr', 0.001)
    base_wd = base_config.get('weight_decay', 0.0)
    no_wd_params = no_wd_params or []
    
    # 合并默认不衰减模式
    all_no_wd_patterns = list(set(no_wd_params + DEFAULT_NO_WEIGHT_DECAY_PATTERNS))
    
    # 收集所有参数
    named_params = dict(model.named_parameters())
    used_params = set()
    param_groups = []
    
    # 按配置分组
    for group_config in param_groups_config:
        group_config = group_config.copy()
        pattern = group_config.pop('params', None)
        lr_mult = group_config.pop('lr_mult', 1.0)
        wd_mult = group_config.pop('weight_decay_mult', 1.0)
        
        if pattern is None:
            continue
            
        # 匹配参数
        decay_params = []
        no_decay_params = []
        
        for name, param in named_params.items():
            if name in used_params:
                continue
            if not param.requires_grad:
                continue
                
            # 检查是否匹配模式
            if not _match_pattern(name, pattern):
                continue
                
            used_params.add(name)
            
            # 检查是否应用权重衰减
            if _should_skip_weight_decay(name, all_no_wd_patterns):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        # 添加有权重衰减的组
        if decay_params:
            param_groups.append({
                'params': decay_params,
                'lr': base_lr * lr_mult,
                'weight_decay': base_wd * wd_mult,
                'group_name': f'{pattern}_decay',
                **group_config
            })
            
        # 添加无权重衰减的组
        if no_decay_params:
            param_groups.append({
                'params': no_decay_params,
                'lr': base_lr * lr_mult,
                'weight_decay': 0.0,
                'group_name': f'{pattern}_no_decay',
                **group_config
            })
            
    # 添加剩余参数到默认组
    remaining_decay = []
    remaining_no_decay = []
    
    for name, param in named_params.items():
        if name in used_params:
            continue
        if not param.requires_grad:
            continue
            
        if _should_skip_weight_decay(name, all_no_wd_patterns):
            remaining_no_decay.append(param)
        else:
            remaining_decay.append(param)
            
    if remaining_decay:
        param_groups.append({
            'params': remaining_decay,
            'lr': base_lr,
            'weight_decay': base_wd,
            'group_name': 'default_decay'
        })
        
    if remaining_no_decay:
        param_groups.append({
            'params': remaining_no_decay,
            'lr': base_lr,
            'weight_decay': 0.0,
            'group_name': 'default_no_decay'
        })
        
    return param_groups


def _build_default_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    no_wd_params: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    构建默认参数分组（区分权重衰减）
    
    Args:
        model: PyTorch 模型
        lr: 学习率
        weight_decay: 权重衰减
        no_wd_params: 额外的不衰减参数模式
        
    Returns:
        参数分组列表
    """
    no_wd_params = no_wd_params or []
    all_no_wd_patterns = list(set(no_wd_params + DEFAULT_NO_WEIGHT_DECAY_PATTERNS))
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if _should_skip_weight_decay(name, all_no_wd_patterns):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    param_groups = []
    
    if decay_params:
        param_groups.append({
            'params': decay_params,
            'lr': lr,
            'weight_decay': weight_decay
        })
        
    if no_decay_params:
        param_groups.append({
            'params': no_decay_params,
            'lr': lr,
            'weight_decay': 0.0
        })
        
    return param_groups


def _match_pattern(name: str, pattern: str) -> bool:
    """
    检查参数名是否匹配模式
    
    支持简单字符串匹配和正则表达式。
    
    Args:
        name: 参数名
        pattern: 匹配模式
        
    Returns:
        是否匹配
    """
    # 简单包含匹配
    if pattern in name:
        return True
        
    # 正则匹配
    try:
        if re.search(pattern, name):
            return True
    except re.error:
        pass
        
    return False


def _should_skip_weight_decay(name: str, patterns: List[str]) -> bool:
    """
    检查参数是否应跳过权重衰减
    
    Args:
        name: 参数名
        patterns: 跳过模式列表
        
    Returns:
        是否跳过权重衰减
    """
    for pattern in patterns:
        if pattern in name:
            return True
    return False


def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    获取优化器信息
    
    Args:
        optimizer: 优化器实例
        
    Returns:
        优化器信息字典
    """
    info = {
        'type': type(optimizer).__name__,
        'num_param_groups': len(optimizer.param_groups),
        'param_groups': []
    }
    
    for i, group in enumerate(optimizer.param_groups):
        group_info = {
            'index': i,
            'lr': group.get('lr'),
            'weight_decay': group.get('weight_decay', 0.0),
            'num_params': len(group['params']),
            'total_params': sum(p.numel() for p in group['params']),
        }
        
        # 添加组名（如果有）
        if 'group_name' in group:
            group_info['name'] = group['group_name']
        if 'layer_id' in group:
            group_info['layer_id'] = group['layer_id']
            
        info['param_groups'].append(group_info)
        
    # 总参数量
    info['total_params'] = sum(g['total_params'] for g in info['param_groups'])
    
    return info
