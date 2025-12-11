# -*- coding: utf-8 -*-
"""
模型构建器模块

提供配置驱动的模型实例化能力。
"""

from typing import Dict, Any, Optional
import torch.nn as nn

from registry.model_registry import MODELS, BACKBONES, NECKS, HEADS


def build_model(config: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    构建完整模型（检测器/分割器/分类器）
    
    Args:
        config: 模型配置，必须包含'type'字段
        default_args: 默认参数
        
    Returns:
        模型实例
    """
    return MODELS.build(config, default_args)


def build_backbone(config: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    构建骨干网络
    
    Args:
        config: 骨干网络配置
        default_args: 默认参数
        
    Returns:
        骨干网络实例
    """
    return BACKBONES.build(config, default_args)


def build_neck(config: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    构建特征融合层
    
    Args:
        config: Neck配置
        default_args: 默认参数
        
    Returns:
        Neck实例
    """
    return NECKS.build(config, default_args)


def build_head(config: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    构建任务头
    
    Args:
        config: Head配置
        default_args: 默认参数
        
    Returns:
        Head实例
    """
    return HEADS.build(config, default_args)
