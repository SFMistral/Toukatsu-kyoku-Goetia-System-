# -*- coding: utf-8 -*-
"""
训练钩子注册器模块

管理训练过程中的各类钩子：日志、检查点、评估、优化等。
支持钩子优先级排序和依赖管理。
"""

from typing import Dict, Any, Optional, List, Type, Set
from abc import ABC, abstractmethod
from enum import IntEnum
from .registry import Registry, ComponentSource


class HookPriority(IntEnum):
    """钩子优先级（数值越小优先级越高）"""
    HIGHEST = 0
    VERY_HIGH = 10
    HIGH = 20
    ABOVE_NORMAL = 30
    NORMAL = 50
    BELOW_NORMAL = 60
    LOW = 70
    VERY_LOW = 80
    LOWEST = 100


class HookRegistry(Registry):
    """训练钩子注册器"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._hook_priorities: Dict[str, int] = {}
        self._hook_dependencies: Dict[str, Set[str]] = {}
        
    def register(
        self,
        name: Optional[str] = None,
        cls: Optional[Type] = None,
        priority: int = HookPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ):
        """
        注册钩子组件
        
        Args:
            name: 组件名称
            cls: 组件类
            priority: 钩子优先级
            dependencies: 依赖的其他钩子
            **kwargs: 其他注册参数
        """
        result = super().register(name=name, cls=cls, **kwargs)
        
        actual_name = name or (cls.__name__ if cls else None)
        if actual_name:
            self._hook_priorities[actual_name] = priority
            self._hook_dependencies[actual_name] = set(dependencies or [])
            
        return result
        
    def get_priority(self, name: str) -> int:
        """获取钩子优先级"""
        return self._hook_priorities.get(name, HookPriority.NORMAL)
        
    def get_dependencies(self, name: str) -> Set[str]:
        """获取钩子依赖"""
        return self._hook_dependencies.get(name, set())
        
    def build_sorted(self, configs: List[Dict[str, Any]]) -> List:
        """
        构建并按优先级排序的钩子列表
        
        Args:
            configs: 钩子配置列表
            
        Returns:
            排序后的钩子实例列表
        """
        hooks = []
        for cfg in configs:
            hook = self.build(cfg)
            hook_type = cfg.get('type')
            priority = self.get_priority(hook_type)
            hooks.append((priority, hook))
            
        # 按优先级排序
        hooks.sort(key=lambda x: x[0])
        return [hook for _, hook in hooks]
        
    def validate_dependencies(self, hook_names: List[str]) -> bool:
        """
        验证钩子依赖是否满足
        
        Args:
            hook_names: 要使用的钩子名称列表
            
        Returns:
            依赖是否满足
        """
        hook_set = set(hook_names)
        for name in hook_names:
            deps = self.get_dependencies(name)
            if not deps.issubset(hook_set):
                missing = deps - hook_set
                raise ValueError(
                    f"Hook '{name}' depends on {missing}, but they are not configured"
                )
        return True


# 创建钩子注册器单例
HOOKS = HookRegistry('hooks', base_class=None)


class BaseHook(ABC):
    """钩子基类"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        
    def before_run(self, runner):
        """训练开始前"""
        pass
        
    def after_run(self, runner):
        """训练结束后"""
        pass
        
    def before_epoch(self, runner):
        """每个epoch开始前"""
        pass
        
    def after_epoch(self, runner):
        """每个epoch结束后"""
        pass
        
    def before_iter(self, runner):
        """每个iteration开始前"""
        pass
        
    def after_iter(self, runner):
        """每个iteration结束后"""
        pass
        
    def before_train_epoch(self, runner):
        """训练epoch开始前"""
        pass
        
    def after_train_epoch(self, runner):
        """训练epoch结束后"""
        pass
        
    def before_val_epoch(self, runner):
        """验证epoch开始前"""
        pass
        
    def after_val_epoch(self, runner):
        """验证epoch结束后"""
        pass


# 日志钩子
class LoggerHook(BaseHook):
    """日志记录钩子"""
    
    def __init__(
        self,
        interval: int = 10,
        log_metric_by_epoch: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interval = interval
        self.log_metric_by_epoch = log_metric_by_epoch
        
    def after_iter(self, runner):
        if not self.enabled:
            return
            
        if runner.iter % self.interval == 0:
            log_str = f"Epoch [{runner.epoch}][{runner.inner_iter}/{runner.max_iters}] "
            
            # 添加损失信息
            if hasattr(runner, 'outputs') and 'loss' in runner.outputs:
                log_str += f"loss: {runner.outputs['loss']:.4f} "
                
            # 添加学习率
            if hasattr(runner, 'optimizer'):
                lr = runner.optimizer.param_groups[0]['lr']
                log_str += f"lr: {lr:.6f}"
                
            print(log_str)


class ProgressBarHook(BaseHook):
    """进度条钩子"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pbar = None
        
    def before_epoch(self, runner):
        if not self.enabled:
            return
            
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=runner.max_iters, desc=f"Epoch {runner.epoch}")
        except ImportError:
            self.pbar = None
            
    def after_iter(self, runner):
        if self.pbar:
            self.pbar.update(1)
            if hasattr(runner, 'outputs') and 'loss' in runner.outputs:
                self.pbar.set_postfix(loss=f"{runner.outputs['loss']:.4f}")
                
    def after_epoch(self, runner):
        if self.pbar:
            self.pbar.close()
            self.pbar = None


# 检查点钩子
class CheckpointHook(BaseHook):
    """检查点保存钩子"""
    
    def __init__(
        self,
        interval: int = 1,
        save_dir: str = 'checkpoints',
        save_optimizer: bool = True,
        max_keep: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interval = interval
        self.save_dir = save_dir
        self.save_optimizer = save_optimizer
        self.max_keep = max_keep
        self.saved_checkpoints = []
        
    def after_epoch(self, runner):
        if not self.enabled:
            return
            
        if (runner.epoch + 1) % self.interval == 0:
            self._save_checkpoint(runner)
            
    def _save_checkpoint(self, runner):
        import os
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': runner.epoch,
            'model_state_dict': runner.model.state_dict(),
        }
        
        if self.save_optimizer and hasattr(runner, 'optimizer'):
            checkpoint['optimizer_state_dict'] = runner.optimizer.state_dict()
            
        filepath = os.path.join(self.save_dir, f'epoch_{runner.epoch}.pth')
        
        try:
            import torch
            torch.save(checkpoint, filepath)
            self.saved_checkpoints.append(filepath)
            
            # 清理旧检查点
            while len(self.saved_checkpoints) > self.max_keep:
                old_ckpt = self.saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")


class BestCheckpointHook(BaseHook):
    """最佳模型检查点钩子"""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_dir: str = 'checkpoints',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.monitor = monitor
        self.mode = mode
        self.save_dir = save_dir
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def after_val_epoch(self, runner):
        if not self.enabled:
            return
            
        current_value = getattr(runner, self.monitor, None)
        if current_value is None:
            return
            
        is_best = (
            (self.mode == 'min' and current_value < self.best_value) or
            (self.mode == 'max' and current_value > self.best_value)
        )
        
        if is_best:
            self.best_value = current_value
            self._save_best(runner)
            
    def _save_best(self, runner):
        import os
        
        os.makedirs(self.save_dir, exist_ok=True)
        filepath = os.path.join(self.save_dir, 'best_model.pth')
        
        try:
            import torch
            torch.save({
                'epoch': runner.epoch,
                'model_state_dict': runner.model.state_dict(),
                f'best_{self.monitor}': self.best_value
            }, filepath)
            print(f"Saved best model with {self.monitor}={self.best_value:.4f}")
        except Exception as e:
            print(f"Failed to save best checkpoint: {e}")


# 评估钩子
class EvalHook(BaseHook):
    """评估钩子"""
    
    def __init__(
        self,
        interval: int = 1,
        start_epoch: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interval = interval
        self.start_epoch = start_epoch
        
    def after_train_epoch(self, runner):
        if not self.enabled:
            return
            
        if runner.epoch >= self.start_epoch and (runner.epoch + 1) % self.interval == 0:
            runner.val()


# 优化钩子
class GradientClipHook(BaseHook):
    """梯度裁剪钩子"""
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    def after_iter(self, runner):
        if not self.enabled:
            return
            
        try:
            import torch.nn.utils as utils
            utils.clip_grad_norm_(
                runner.model.parameters(),
                max_norm=self.max_norm,
                norm_type=self.norm_type
            )
        except Exception:
            pass


class EMAHook(BaseHook):
    """指数移动平均钩子"""
    
    def __init__(
        self,
        decay: float = 0.9999,
        update_interval: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.decay = decay
        self.update_interval = update_interval
        self.shadow = {}
        
    def before_run(self, runner):
        if not self.enabled:
            return
            
        # 初始化shadow参数
        for name, param in runner.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def after_iter(self, runner):
        if not self.enabled:
            return
            
        if runner.iter % self.update_interval == 0:
            for name, param in runner.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] + 
                        (1 - self.decay) * param.data
                    )


# 早停钩子
class EarlyStoppingHook(BaseHook):
    """早停钩子"""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def after_val_epoch(self, runner):
        if not self.enabled:
            return
            
        current_value = getattr(runner, self.monitor, None)
        if current_value is None:
            return
            
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
            
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs without improvement")
            runner.should_stop = True


# 注册内置钩子
HOOKS.register('LoggerHook', LoggerHook, 
              priority=HookPriority.VERY_LOW, category='logging', source=ComponentSource.BUILTIN)
HOOKS.register('ProgressBarHook', ProgressBarHook,
              priority=HookPriority.VERY_LOW, category='logging', source=ComponentSource.BUILTIN)
HOOKS.register('CheckpointHook', CheckpointHook,
              priority=HookPriority.LOW, category='checkpoint', source=ComponentSource.BUILTIN)
HOOKS.register('BestCheckpointHook', BestCheckpointHook,
              priority=HookPriority.LOW, category='checkpoint', 
              dependencies=['EvalHook'], source=ComponentSource.BUILTIN)
HOOKS.register('EvalHook', EvalHook,
              priority=HookPriority.NORMAL, category='evaluation', source=ComponentSource.BUILTIN)
HOOKS.register('GradientClipHook', GradientClipHook,
              priority=HookPriority.HIGH, category='optimization', source=ComponentSource.BUILTIN)
HOOKS.register('EMAHook', EMAHook,
              priority=HookPriority.ABOVE_NORMAL, category='optimization', source=ComponentSource.BUILTIN)
HOOKS.register('EarlyStoppingHook', EarlyStoppingHook,
              priority=HookPriority.BELOW_NORMAL, category='control',
              dependencies=['EvalHook'], source=ComponentSource.BUILTIN)
