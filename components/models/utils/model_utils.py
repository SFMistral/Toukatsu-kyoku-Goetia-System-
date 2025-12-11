# -*- coding: utf-8 -*-
"""
模型通用工具模块

提供模型复杂度分析、参数冻结、模块操作等工具函数。
"""

from typing import Optional, Tuple, Union, List, Type
import torch
import torch.nn as nn


def get_model_complexity(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    计算模型FLOPs和参数量
    
    Args:
        model: 模型
        input_shape: 输入形状 (C, H, W)
        device: 设备
        
    Returns:
        (flops, params) - FLOPs数量和参数量
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        
        if device is None:
            device = next(model.parameters()).device
            
        dummy_input = torch.randn(1, *input_shape, device=device)
        
        flops = FlopCountAnalysis(model, dummy_input)
        params = parameter_count(model)
        
        return flops.total(), sum(params.values())
    except ImportError:
        # 回退到简单计算
        params = count_parameters(model)
        flops = _estimate_flops(model, input_shape)
        return flops, params


def _estimate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> float:
    """简单估算FLOPs"""
    total_flops = 0
    
    def hook_fn(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Conv2d):
            # FLOPs = 2 * Cin * Cout * K^2 * Hout * Wout
            out_h, out_w = output.shape[2:]
            flops = 2 * module.in_channels * module.out_channels * \
                    module.kernel_size[0] * module.kernel_size[1] * \
                    out_h * out_w / module.groups
            total_flops += flops
        elif isinstance(module, nn.Linear):
            # FLOPs = 2 * in_features * out_features
            flops = 2 * module.in_features * module.out_features
            total_flops += flops
            
    hooks = []
    for m in model.modules():
        hooks.append(m.register_forward_hook(hook_fn))
        
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape, device=device)
    
    with torch.no_grad():
        model(dummy_input)
        
    for hook in hooks:
        hook.remove()
        
    return total_flops


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    统计模型参数数量
    
    Args:
        model: 模型
        trainable_only: 是否只统计可训练参数
        
    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_module(module: nn.Module):
    """冻结模块参数"""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module):
    """解冻模块参数"""
    for param in module.parameters():
        param.requires_grad = True


def freeze_bn(module: nn.Module):
    """冻结BatchNorm层"""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            freeze_module(m)


def set_bn_eval(module: nn.Module):
    """设置BatchNorm为eval模式"""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def fuse_conv_bn(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    融合Conv + BN层
    
    Args:
        model: 模型
        inplace: 是否原地修改
        
    Returns:
        融合后的模型
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
        
    def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        # 计算融合后的权重和偏置
        w_conv = conv.weight.clone()
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
        
        bn_mean = bn.running_mean
        bn_var = bn.running_var
        bn_gamma = bn.weight
        bn_beta = bn.bias
        bn_eps = bn.eps
        
        std = torch.sqrt(bn_var + bn_eps)
        
        w_fused = w_conv * (bn_gamma / std).view(-1, 1, 1, 1)
        b_fused = (b_conv - bn_mean) * bn_gamma / std + bn_beta
        
        # 创建新的卷积层
        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
            groups=conv.groups, bias=True,
        )
        fused_conv.weight.data = w_fused
        fused_conv.bias.data = b_fused
        
        return fused_conv
        
    # 遍历并融合
    prev_name = None
    prev_module = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and isinstance(prev_module, nn.Conv2d):
            # 找到Conv-BN对
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model if not parent_name else get_module_by_name(model, parent_name)
            
            conv_name = prev_name.split('.')[-1]
            bn_name = name.split('.')[-1]
            
            fused = _fuse(prev_module, module)
            setattr(parent, conv_name, fused)
            setattr(parent, bn_name, nn.Identity())
            
        prev_name = name
        prev_module = module
        
    return model


def replace_module(
    model: nn.Module,
    old_type: Type[nn.Module],
    new_module_fn,
    inplace: bool = False,
) -> nn.Module:
    """
    替换模型中的模块
    
    Args:
        model: 模型
        old_type: 要替换的模块类型
        new_module_fn: 创建新模块的函数，接收旧模块作为参数
        inplace: 是否原地修改
        
    Returns:
        替换后的模型
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
        
    for name, module in model.named_modules():
        if isinstance(module, old_type):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model if not parent_name else get_module_by_name(model, parent_name)
            child_name = name.split('.')[-1]
            
            new_module = new_module_fn(module)
            setattr(parent, child_name, new_module)
            
    return model


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """按名称获取模块"""
    names = name.split('.')
    module = model
    for n in names:
        module = getattr(module, n)
    return module


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    将数值调整为divisor的倍数
    
    Args:
        v: 原始值
        divisor: 除数
        min_value: 最小值
        
    Returns:
        调整后的值
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保向下取整不会减少超过10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def auto_pad(
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1
) -> Union[int, Tuple[int, int]]:
    """自动计算same padding"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
        
    pad_h = (kernel_size[0] - 1) * dilation[0] // 2
    pad_w = (kernel_size[1] - 1) * dilation[1] // 2
    
    if pad_h == pad_w:
        return pad_h
    return (pad_h, pad_w)
