# -*- coding: utf-8 -*-
"""
Adam 优化器模块

提供 Adam 及其变体的封装。
"""

from typing import Iterator, Optional, Tuple
import torch
from torch.optim import Optimizer
from registry import OPTIMIZERS, ComponentSource


class Adam(torch.optim.Adam):
    """
    标准 Adam 优化器 (PyTorch 封装)
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 0.001)
        betas: 动量参数 (默认 (0.9, 0.999))
        eps: 数值稳定性 (默认 1e-8)
        weight_decay: 权重衰减 L2 (默认 0.0)
        amsgrad: 是否使用 AMSGrad (默认 False)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
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


class AdamP(Optimizer):
    """
    Adam with Decoupled Weight Decay and Projection
    
    结合解耦权重衰减和梯度投影，提升训练稳定性。
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 0.001)
        betas: 动量参数 (默认 (0.9, 0.999))
        eps: 数值稳定性 (默认 1e-8)
        weight_decay: 权重衰减 (默认 0.0)
        delta: 投影阈值 (默认 0.1)
        wd_ratio: 权重衰减比率 (默认 0.1)
        nesterov: 是否使用 Nesterov 动量 (默认 False)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov
        )
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # 更新一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Nesterov 动量
                if nesterov:
                    perturb = (beta1 * exp_avg_corrected + (1 - beta1) * grad / bias_correction1) / \
                              (exp_avg_sq_corrected.sqrt() + group['eps'])
                else:
                    perturb = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + group['eps'])
                    
                # 解耦权重衰减
                if group['weight_decay'] != 0:
                    if self._should_project(p, perturb, group['delta'], group['wd_ratio']):
                        wd = group['weight_decay']
                    else:
                        wd = group['weight_decay'] * group['wd_ratio']
                    p.mul_(1 - group['lr'] * wd)
                    
                # 参数更新
                p.add_(perturb, alpha=-group['lr'])
                
        return loss
        
    def _should_project(
        self,
        p: torch.Tensor,
        perturb: torch.Tensor,
        delta: float,
        wd_ratio: float
    ) -> bool:
        """判断是否应用完整权重衰减"""
        if p.dim() <= 1:
            return False
            
        p_norm = p.norm()
        perturb_norm = perturb.norm()
        
        if p_norm == 0 or perturb_norm == 0:
            return False
            
        cos_sim = (p * perturb).sum() / (p_norm * perturb_norm)
        return cos_sim.abs() < delta


class NAdam(Optimizer):
    """
    Adam with Nesterov momentum
    
    结合 Nesterov 动量，通常收敛更快。
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 0.001)
        betas: 动量参数 (默认 (0.9, 0.999))
        eps: 数值稳定性 (默认 1e-8)
        weight_decay: 权重衰减 (默认 0.0)
        momentum_decay: 动量衰减 (默认 0.004)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004
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
            momentum_decay=momentum_decay
        )
        super().__init__(params, defaults)
        
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
                
                # L2 正则化
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                    
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['mu_product'] = 1.0
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # 动量衰减
                mu = beta1 * (1 - 0.5 * (0.96 ** (state['step'] * group['momentum_decay'])))
                mu_next = beta1 * (1 - 0.5 * (0.96 ** ((state['step'] + 1) * group['momentum_decay'])))
                
                state['mu_product'] *= mu
                mu_product_next = state['mu_product'] * mu_next
                
                # 更新一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Nesterov 更新
                exp_avg_corrected = mu_next * exp_avg / (1 - mu_product_next) + \
                                   (1 - mu) * grad / (1 - state['mu_product'])
                                   
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                
                p.addcdiv_(exp_avg_corrected, denom, value=-group['lr'])
                
        return loss


class RAdam(Optimizer):
    """
    Rectified Adam
    
    自适应学习率校正，无需学习率预热，训练更稳定。
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 0.001)
        betas: 动量参数 (默认 (0.9, 0.999))
        eps: 数值稳定性 (默认 1e-8)
        weight_decay: 权重衰减 (默认 0.0)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
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
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        
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
                
                # L2 正则化
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                    
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # 更新一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg_corrected = exp_avg / bias_correction1
                
                # 计算最大长度 SMA
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * state['step'] * (beta2 ** state['step']) / bias_correction2
                
                # 方差校正
                if rho_t > 5:
                    # 使用自适应学习率
                    rect = ((rho_t - 4) * (rho_t - 2) * rho_inf / 
                           ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                    exp_avg_sq_corrected = (exp_avg_sq / bias_correction2).sqrt()
                    p.addcdiv_(exp_avg_corrected, exp_avg_sq_corrected.add_(group['eps']), 
                              value=-group['lr'] * rect)
                else:
                    # 使用 SGD 更新
                    p.add_(exp_avg_corrected, alpha=-group['lr'])
                    
        return loss


class Adagrad(torch.optim.Adagrad):
    """
    Adagrad 优化器 (PyTorch 封装)
    
    自适应学习率，适合稀疏梯度。
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 0.01)
        lr_decay: 学习率衰减 (默认 0.0)
        weight_decay: 权重衰减 (默认 0.0)
        eps: 数值稳定性 (默认 1e-10)
        initial_accumulator_value: 初始累加器值 (默认 0.0)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-10,
        initial_accumulator_value: float = 0.0
    ):
        super().__init__(
            params,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            eps=eps,
            initial_accumulator_value=initial_accumulator_value
        )


class Adadelta(torch.optim.Adadelta):
    """
    Adadelta 优化器 (PyTorch 封装)
    
    Adagrad 的改进版本，解决学习率单调递减问题。
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 1.0)
        rho: 衰减率 (默认 0.9)
        eps: 数值稳定性 (默认 1e-6)
        weight_decay: 权重衰减 (默认 0.0)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0
    ):
        super().__init__(
            params,
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay
        )


# 注册优化器
OPTIMIZERS.register('Adam', Adam, category='adaptive', source=ComponentSource.BUILTIN, force=True)
OPTIMIZERS.register('AdamP', AdamP, category='adaptive', source=ComponentSource.BUILTIN)
OPTIMIZERS.register('NAdam', NAdam, category='adaptive', source=ComponentSource.BUILTIN, force=True)
OPTIMIZERS.register('RAdam', RAdam, category='adaptive', source=ComponentSource.BUILTIN, force=True)
OPTIMIZERS.register('Adagrad', Adagrad, category='adaptive', source=ComponentSource.BUILTIN, force=True)
OPTIMIZERS.register('Adadelta', Adadelta, category='adaptive', source=ComponentSource.BUILTIN, force=True)
