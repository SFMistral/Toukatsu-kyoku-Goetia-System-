# -*- coding: utf-8 -*-
"""
AdamW 优化器模块

提供 AdamW 及其变体的封装，包括高效实现和低显存版本。
"""

from typing import Iterator, Optional, Tuple
import torch
from torch.optim import Optimizer
from registry import OPTIMIZERS, ComponentSource


class AdamW(torch.optim.AdamW):
    """
    AdamW 优化器 (解耦权重衰减)
    
    与 Adam 的区别：
    - Adam: grad = grad + weight_decay * param
    - AdamW: param = param - lr * weight_decay * param
    
    AdamW 权重衰减不依赖梯度缩放，正则化效果更稳定。
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 0.001)
        betas: 动量参数 (默认 (0.9, 0.999))
        eps: 数值稳定性 (默认 1e-8)
        weight_decay: 权重衰减 (默认 0.01)
        amsgrad: 是否使用 AMSGrad (默认 False)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )


class FusedAdamW(Optimizer):
    """
    融合实现的 AdamW 优化器
    
    特性：
    - CUDA 内核融合 (需要 apex 或 DeepSpeed)
    - 减少内存访问
    - 更高的计算效率
    
    如果 apex/DeepSpeed 不可用，回退到标准 AdamW 实现。
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 0.001)
        betas: 动量参数 (默认 (0.9, 0.999))
        eps: 数值稳定性 (默认 1e-8)
        weight_decay: 权重衰减 (默认 0.01)
        amsgrad: 是否使用 AMSGrad (默认 False)
        use_apex: 是否尝试使用 apex (默认 True)
        use_deepspeed: 是否尝试使用 DeepSpeed (默认 True)
    """
    
    _fused_impl = None
    _impl_source = 'native'
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        use_apex: bool = True,
        use_deepspeed: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        
        # 尝试加载融合实现
        self._try_load_fused_impl(use_apex, use_deepspeed)
        
        super().__init__(params, defaults)
        
    def _try_load_fused_impl(self, use_apex: bool, use_deepspeed: bool):
        """尝试加载融合实现"""
        if FusedAdamW._fused_impl is not None:
            return
            
        # 尝试 apex
        if use_apex:
            try:
                from apex.optimizers import FusedAdam
                FusedAdamW._fused_impl = FusedAdam
                FusedAdamW._impl_source = 'apex'
                return
            except ImportError:
                pass
                
        # 尝试 DeepSpeed
        if use_deepspeed:
            try:
                from deepspeed.ops.adam import FusedAdam
                FusedAdamW._fused_impl = FusedAdam
                FusedAdamW._impl_source = 'deepspeed'
                return
            except ImportError:
                pass
                
        # 回退到原生实现
        FusedAdamW._impl_source = 'native'
        
    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError('FusedAdamW does not support sparse gradients')
                    
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                        
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    
                state['step'] += 1
                
                # 解耦权重衰减
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    
                # 更新一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['amsgrad']:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                    
                step_size = group['lr'] / bias_correction1
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss
        
    @property
    def impl_source(self) -> str:
        """返回当前使用的实现来源"""
        return FusedAdamW._impl_source


class Adam8bit(Optimizer):
    """
    8-bit Adam 优化器 (省显存)
    
    特性：
    - 优化器状态 8-bit 量化
    - 显存占用减少约 75%
    - 需要 bitsandbytes 库
    
    如果 bitsandbytes 不可用，回退到标准 AdamW 实现。
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 0.001)
        betas: 动量参数 (默认 (0.9, 0.999))
        eps: 数值稳定性 (默认 1e-8)
        weight_decay: 权重衰减 (默认 0.01)
        min_8bit_size: 最小 8-bit 量化大小 (默认 4096)
        percentile_clipping: 百分位裁剪 (默认 100)
        block_wise: 是否使用块级量化 (默认 True)
    """
    
    _bnb_available = None
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        min_8bit_size: int = 4096,
        percentile_clipping: int = 100,
        block_wise: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            min_8bit_size=min_8bit_size,
            percentile_clipping=percentile_clipping,
            block_wise=block_wise
        )
        
        # 检查 bitsandbytes 可用性
        self._check_bnb_available()
        
        super().__init__(params, defaults)
        
    def _check_bnb_available(self):
        """检查 bitsandbytes 是否可用"""
        if Adam8bit._bnb_available is None:
            try:
                import bitsandbytes as bnb
                Adam8bit._bnb_available = True
            except ImportError:
                Adam8bit._bnb_available = False
                import warnings
                warnings.warn(
                    "bitsandbytes not available, Adam8bit will use standard AdamW implementation. "
                    "Install with: pip install bitsandbytes"
                )
                
    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    
                    # 根据参数大小决定是否使用 8-bit
                    use_8bit = (
                        Adam8bit._bnb_available and 
                        p.numel() >= group['min_8bit_size'] and
                        p.is_cuda
                    )
                    state['use_8bit'] = use_8bit
                    
                    if use_8bit:
                        import bitsandbytes.functional as F
                        # 8-bit 状态
                        state['exp_avg'] = torch.zeros_like(p, dtype=torch.uint8)
                        state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.uint8)
                        state['exp_avg_qmap'] = F.create_dynamic_map(signed=True)
                        state['exp_avg_sq_qmap'] = F.create_dynamic_map(signed=False)
                        state['exp_avg_absmax'] = torch.zeros((1,), device=p.device)
                        state['exp_avg_sq_absmax'] = torch.zeros((1,), device=p.device)
                    else:
                        # 标准 32-bit 状态
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        
                state['step'] += 1
                
                # 解耦权重衰减
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    
                if state['use_8bit']:
                    self._step_8bit(p, grad, state, group)
                else:
                    self._step_32bit(p, grad, state, group)
                    
        return loss
        
    def _step_32bit(self, p, grad, state, group):
        """标准 32-bit 更新"""
        beta1, beta2 = group['betas']
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        
        # 更新一阶和二阶矩估计
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # 偏差校正
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
        step_size = group['lr'] / bias_correction1
        
        p.addcdiv_(exp_avg, denom, value=-step_size)
        
    def _step_8bit(self, p, grad, state, group):
        """8-bit 量化更新"""
        import bitsandbytes.functional as F
        
        beta1, beta2 = group['betas']
        
        # 反量化
        exp_avg = F.dequantize_blockwise(
            state['exp_avg'],
            state['exp_avg_qmap'],
            state['exp_avg_absmax']
        )
        exp_avg_sq = F.dequantize_blockwise(
            state['exp_avg_sq'],
            state['exp_avg_sq_qmap'],
            state['exp_avg_sq_absmax']
        )
        
        # 更新一阶和二阶矩估计
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # 偏差校正
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
        step_size = group['lr'] / bias_correction1
        
        p.addcdiv_(exp_avg, denom, value=-step_size)
        
        # 重新量化
        state['exp_avg'], state['exp_avg_qmap'], state['exp_avg_absmax'] = \
            F.quantize_blockwise(exp_avg, code=state['exp_avg_qmap'])
        state['exp_avg_sq'], state['exp_avg_sq_qmap'], state['exp_avg_sq_absmax'] = \
            F.quantize_blockwise(exp_avg_sq, code=state['exp_avg_sq_qmap'])
            
    @property
    def is_8bit_enabled(self) -> bool:
        """返回 8-bit 是否可用"""
        return Adam8bit._bnb_available


# 注册优化器
OPTIMIZERS.register('AdamW', AdamW, category='adaptive', source=ComponentSource.BUILTIN, force=True)
OPTIMIZERS.register('FusedAdamW', FusedAdamW, category='adaptive', source=ComponentSource.BUILTIN)
OPTIMIZERS.register('Adam8bit', Adam8bit, category='adaptive', source=ComponentSource.BUILTIN)
