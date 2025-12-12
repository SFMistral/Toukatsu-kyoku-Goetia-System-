# -*- coding: utf-8 -*-
"""
层级学习率衰减工具模块

为不同层设置不同学习率，支持多种衰减模式，自动识别模型层级结构。
"""

import re
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn


def get_num_layers(model: nn.Module) -> int:
    """
    获取模型层数
    
    自动识别常见模型架构的层数：
    - ViT: patch_embed + blocks + head
    - Swin: patch_embed + stages + head
    - ResNet: conv1 + layer1-4 + fc
    
    Args:
        model: PyTorch 模型
        
    Returns:
        模型层数
    """
    # 尝试从模型属性获取
    if hasattr(model, 'num_layers'):
        return model.num_layers
    if hasattr(model, 'depth'):
        return model.depth
        
    # ViT 类模型
    if hasattr(model, 'blocks'):
        return len(model.blocks) + 2  # patch_embed + blocks + head
        
    # Swin 类模型
    if hasattr(model, 'layers') and hasattr(model, 'num_layers'):
        return sum(model.depths) + 2 if hasattr(model, 'depths') else model.num_layers + 2
        
    # ResNet 类模型
    if hasattr(model, 'layer4'):
        num_layers = 2  # conv1 + fc
        for i in range(1, 5):
            layer = getattr(model, f'layer{i}', None)
            if layer is not None:
                num_layers += len(layer)
        return num_layers
        
    # 通用：按模块深度计算
    max_depth = 0
    for name, _ in model.named_parameters():
        depth = name.count('.')
        max_depth = max(max_depth, depth)
    return max_depth + 1


def get_layer_id(name: str, num_layers: int, model: Optional[nn.Module] = None) -> int:
    """
    获取参数所属层 ID
    
    Args:
        name: 参数名称
        num_layers: 总层数
        model: 可选的模型引用，用于更精确的层级识别
        
    Returns:
        层 ID (0 为最底层，num_layers-1 为最顶层)
    """
    # ViT 类模型
    if 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
        return 0
    if 'blocks' in name:
        match = re.search(r'blocks\.(\d+)', name)
        if match:
            block_id = int(match.group(1))
            return block_id + 1
    if 'head' in name or 'fc_norm' in name or 'norm.' in name:
        return num_layers - 1
        
    # Swin 类模型
    if 'layers' in name:
        match = re.search(r'layers\.(\d+)\.blocks\.(\d+)', name)
        if match:
            stage_id = int(match.group(1))
            block_id = int(match.group(2))
            # 需要累计之前 stage 的 block 数
            return stage_id * 2 + block_id + 1
            
    # ResNet 类模型
    if 'conv1' in name or 'bn1' in name:
        return 0
    for i in range(1, 5):
        if f'layer{i}' in name:
            match = re.search(rf'layer{i}\.(\d+)', name)
            if match:
                block_id = int(match.group(1))
                base = sum(range(i)) * 2  # 简化计算
                return base + block_id + 1
    if 'fc' in name:
        return num_layers - 1
        
    # 通用：按名称深度
    depth = name.count('.')
    return min(depth, num_layers - 1)


def get_layer_decay_params(
    model: nn.Module,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    获取层级衰减参数分组
    
    Args:
        model: PyTorch 模型
        config: 层级衰减配置
            - decay_rate: 衰减率 (默认 0.75)
            - decay_type: 衰减类型 ('layer_wise', 'stage_wise', 'uniform')
            - num_layers: 层数 (可选，自动检测)
            - base_lr: 基础学习率
            - weight_decay: 权重衰减
            - no_weight_decay_params: 不应用权重衰减的参数名模式
            
    Returns:
        参数分组列表，每组包含:
        - params: 参数列表
        - lr: 该组学习率
        - weight_decay: 权重衰减
        - layer_id: 层 ID
    """
    decay_rate = config.get('decay_rate', 0.75)
    decay_type = config.get('decay_type', 'layer_wise')
    num_layers = config.get('num_layers') or get_num_layers(model)
    base_lr = config.get('base_lr', config.get('lr', 0.001))
    weight_decay = config.get('weight_decay', 0.0)
    no_wd_patterns = config.get('no_weight_decay_params', [])
    
    # 默认不应用权重衰减的参数
    default_no_wd = ['bias', 'norm', 'LayerNorm', 'BatchNorm', 
                    'pos_embed', 'cls_token', 'relative_position']
    no_wd_patterns = list(set(no_wd_patterns + default_no_wd))
    
    # 按层分组参数
    layer_params: Dict[int, Dict[str, List]] = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 获取层 ID
        if decay_type == 'uniform':
            layer_id = 0
        elif decay_type == 'stage_wise':
            layer_id = _get_stage_id(name, num_layers)
        else:  # layer_wise
            layer_id = get_layer_id(name, num_layers, model)
            
        # 判断是否应用权重衰减
        apply_wd = True
        for pattern in no_wd_patterns:
            if pattern in name:
                apply_wd = False
                break
                
        # 初始化层分组
        if layer_id not in layer_params:
            layer_params[layer_id] = {'decay': [], 'no_decay': []}
            
        if apply_wd:
            layer_params[layer_id]['decay'].append(param)
        else:
            layer_params[layer_id]['no_decay'].append(param)
            
    # 构建参数组
    param_groups = []
    
    for layer_id in sorted(layer_params.keys()):
        # 计算该层学习率: base_lr * (decay_rate ^ (num_layers - layer_id - 1))
        layer_lr = base_lr * (decay_rate ** (num_layers - layer_id - 1))
        
        # 有权重衰减的参数组
        if layer_params[layer_id]['decay']:
            param_groups.append({
                'params': layer_params[layer_id]['decay'],
                'lr': layer_lr,
                'weight_decay': weight_decay,
                'layer_id': layer_id
            })
            
        # 无权重衰减的参数组
        if layer_params[layer_id]['no_decay']:
            param_groups.append({
                'params': layer_params[layer_id]['no_decay'],
                'lr': layer_lr,
                'weight_decay': 0.0,
                'layer_id': layer_id
            })
            
    return param_groups


def _get_stage_id(name: str, num_stages: int = 4) -> int:
    """获取参数所属阶段 ID (用于 stage_wise 衰减)"""
    # Swin/ConvNeXt 类模型
    if 'stages' in name or 'layers' in name:
        match = re.search(r'(?:stages|layers)\.(\d+)', name)
        if match:
            return int(match.group(1))
            
    # ResNet 类模型
    for i in range(1, 5):
        if f'layer{i}' in name:
            return i - 1
            
    # 头部
    if 'head' in name or 'fc' in name or 'classifier' in name:
        return num_stages - 1
        
    # 默认为第一阶段
    return 0
