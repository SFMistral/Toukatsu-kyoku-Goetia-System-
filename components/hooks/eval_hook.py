# -*- coding: utf-8 -*-
"""
评估钩子模块

提供模型评估功能，支持定期评估和分布式评估。
"""

from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

import torch

from .base_hook import BaseHook, HookPriority
from registry import HOOKS

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing import Any as RunnerType


class EvalHook(BaseHook):
    """
    评估钩子
    
    定期触发模型评估，计算评估指标。
    
    Args:
        dataloader: 验证数据加载器
        interval: 评估间隔（epoch 或 iter）
        by_epoch: 按 epoch 还是 iter 评估
        start: 开始评估的 epoch/iter
        save_best: 是否触发保存最佳
        metric: 评估指标名
        rule: 比较规则 ('greater' 或 'less')
        efficient_test: 是否节省内存
        test_fn: 自定义测试函数
        
    Example:
        >>> hook = EvalHook(
        ...     dataloader=val_loader,
        ...     interval=1,
        ...     metric='accuracy'
        ... )
    """
    
    priority = HookPriority.LOW
    
    def __init__(
        self,
        dataloader: Optional['DataLoader'] = None,
        interval: int = 1,
        by_epoch: bool = True,
        start: int = 0,
        save_best: bool = True,
        metric: str = 'accuracy',
        rule: str = 'greater',
        efficient_test: bool = False,
        test_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.start = start
        self.save_best = save_best
        self.metric = metric
        self.rule = rule
        self.efficient_test = efficient_test
        self.test_fn = test_fn
        
        # 状态追踪
        self._best_score: Optional[float] = None
        
        # 验证规则
        if rule not in ('greater', 'less'):
            raise ValueError(f"rule must be 'greater' or 'less', got {rule}")
            
    def before_run(self, runner: 'RunnerType') -> None:
        """训练开始前初始化"""
        # 如果没有提供 dataloader，尝试从 runner 获取
        if self.dataloader is None and hasattr(runner, 'val_dataloader'):
            self.dataloader = runner.val_dataloader
            
    def after_train_epoch(self, runner: 'RunnerType') -> None:
        """训练 epoch 结束后评估"""
        if not self.by_epoch:
            return
            
        if runner.epoch < self.start:
            return
            
        if self.every_n_epochs(runner, self.interval) or self.is_last_epoch(runner):
            self._do_evaluate(runner)
            
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """训练迭代结束后评估（按 iter 模式）"""
        if self.by_epoch:
            return
            
        if runner.iter < self.start:
            return
            
        if self.every_n_iters(runner, self.interval) or self.is_last_iter(runner):
            self._do_evaluate(runner)
            
    def _do_evaluate(self, runner: 'RunnerType') -> Dict[str, float]:
        """执行评估"""
        if self.dataloader is None:
            print("Warning: No validation dataloader provided")
            return {}
            
        # 使用自定义测试函数或默认评估
        if self.test_fn is not None:
            results = self.test_fn(runner.model, self.dataloader)
        else:
            results = self._default_evaluate(runner)
            
        # 更新 runner 的验证输出
        runner.val_outputs = results
        
        # 检查是否是最佳
        if self.save_best and self.metric in results:
            current_score = results[self.metric]
            is_best = self._is_better(current_score)
            if is_best:
                self._best_score = current_score
                runner.is_best = True
            else:
                runner.is_best = False
                
        return results
        
    def _default_evaluate(self, runner: 'RunnerType') -> Dict[str, float]:
        """默认评估逻辑"""
        model = runner.model
        model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                # 获取输入和标签
                if isinstance(batch, dict):
                    inputs = batch.get('inputs', batch.get('input'))
                    targets = batch.get('targets', batch.get('label'))
                elif isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    continue
                    
                # 移动到设备
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                if hasattr(runner, 'criterion'):
                    loss = runner.criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    
                # 计算准确率（分类任务）
                if outputs.dim() > 1:
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(targets).sum().item()
                    
                total_samples += inputs.size(0)
                
                # 节省内存模式
                if self.efficient_test:
                    del outputs
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
        model.train()
        
        results = {}
        if total_samples > 0:
            results['val_loss'] = total_loss / total_samples
            results['accuracy'] = total_correct / total_samples
            
        return results
        
    def _is_better(self, current: float) -> bool:
        """判断当前值是否更好"""
        if self._best_score is None:
            return True
            
        if self.rule == 'greater':
            return current > self._best_score
        else:
            return current < self._best_score


class DistEvalHook(EvalHook):
    """
    分布式评估钩子
    
    支持多 GPU 数据并行评估，结果自动聚合。
    
    Args:
        dataloader: 验证数据加载器
        interval: 评估间隔
        by_epoch: 按 epoch 还是 iter
        broadcast_bn_buffer: 是否广播 BN buffer
        tmpdir: 临时目录
        gpu_collect: 是否使用 GPU 收集结果
        **kwargs: 其他 EvalHook 参数
        
    Example:
        >>> hook = DistEvalHook(
        ...     dataloader=val_loader,
        ...     interval=1,
        ...     gpu_collect=True
        ... )
    """
    
    def __init__(
        self,
        dataloader: Optional['DataLoader'] = None,
        interval: int = 1,
        by_epoch: bool = True,
        broadcast_bn_buffer: bool = True,
        tmpdir: Optional[str] = None,
        gpu_collect: bool = False,
        **kwargs
    ):
        super().__init__(
            dataloader=dataloader,
            interval=interval,
            by_epoch=by_epoch,
            **kwargs
        )
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect
        
    def _do_evaluate(self, runner: 'RunnerType') -> Dict[str, float]:
        """执行分布式评估"""
        if self.dataloader is None:
            print("Warning: No validation dataloader provided")
            return {}
            
        # 检查是否是分布式环境
        if not self._is_distributed():
            return super()._do_evaluate(runner)
            
        import torch.distributed as dist
        
        # 广播 BN buffer
        if self.broadcast_bn_buffer:
            self._broadcast_bn_buffers(runner.model)
            
        # 执行评估
        results = self._distributed_evaluate(runner)
        
        # 仅主进程更新结果
        if self._is_main_process():
            runner.val_outputs = results
            
            if self.save_best and self.metric in results:
                current_score = results[self.metric]
                is_best = self._is_better(current_score)
                if is_best:
                    self._best_score = current_score
                    runner.is_best = True
                else:
                    runner.is_best = False
                    
        return results
        
    def _distributed_evaluate(self, runner: 'RunnerType') -> Dict[str, float]:
        """分布式评估逻辑"""
        import torch.distributed as dist
        
        model = runner.model
        model.eval()
        
        # 本地统计
        local_loss = torch.tensor(0.0).cuda()
        local_correct = torch.tensor(0).cuda()
        local_samples = torch.tensor(0).cuda()
        
        with torch.no_grad():
            for batch in self.dataloader:
                if isinstance(batch, dict):
                    inputs = batch.get('inputs', batch.get('input'))
                    targets = batch.get('targets', batch.get('label'))
                elif isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    continue
                    
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                if hasattr(runner, 'criterion'):
                    loss = runner.criterion(outputs, targets)
                    local_loss += loss * inputs.size(0)
                    
                if outputs.dim() > 1:
                    _, predicted = outputs.max(1)
                    local_correct += predicted.eq(targets).sum()
                    
                local_samples += inputs.size(0)
                
        model.train()
        
        # 聚合结果
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_samples, op=dist.ReduceOp.SUM)
        
        results = {}
        total_samples = local_samples.item()
        if total_samples > 0:
            results['val_loss'] = local_loss.item() / total_samples
            results['accuracy'] = local_correct.item() / total_samples
            
        return results
        
    def _broadcast_bn_buffers(self, model: torch.nn.Module) -> None:
        """广播 BatchNorm buffers"""
        import torch.distributed as dist
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, 
                                   torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
                for buf in module.buffers():
                    dist.broadcast(buf, src=0)
                    
    def _is_distributed(self) -> bool:
        """检查是否是分布式环境"""
        try:
            import torch.distributed as dist
            return dist.is_available() and dist.is_initialized()
        except:
            return False
            
    def _is_main_process(self) -> bool:
        """检查是否是主进程"""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except:
            pass
        return True


# 注册钩子（如果尚未注册）
def _safe_register(name, cls, **kwargs):
    if not HOOKS.contains(name):
        HOOKS.register(name, cls, **kwargs)
    else:
        HOOKS._components[name].cls = cls

_safe_register('EvalHook', EvalHook, 
               priority=HookPriority.LOW, category='evaluation')
_safe_register('DistEvalHook', DistEvalHook, 
               priority=HookPriority.LOW, category='evaluation')
