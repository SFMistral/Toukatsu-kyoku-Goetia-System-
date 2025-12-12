# -*- coding: utf-8 -*-
"""
日志钩子模块

提供训练日志记录功能，支持多种日志后端。
"""

import os
import time
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING

from .base_hook import BaseHook, HookPriority
from registry import HOOKS

if TYPE_CHECKING:
    from typing import Any as RunnerType


class LoggerHook(BaseHook):
    """
    通用日志钩子
    
    记录训练指标到控制台和文件。
    
    Args:
        interval: 日志间隔（iter）
        by_epoch: 按 epoch 统计
        log_metric_by_epoch: 指标按 epoch 记录
        ignore_last: 忽略最后不完整间隔
        
    Example:
        >>> hook = LoggerHook(interval=50, by_epoch=True)
    """
    
    priority = HookPriority.NORMAL
    
    def __init__(
        self,
        interval: int = 50,
        by_epoch: bool = True,
        log_metric_by_epoch: bool = True,
        ignore_last: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interval = interval
        self.by_epoch = by_epoch
        self.log_metric_by_epoch = log_metric_by_epoch
        self.ignore_last = ignore_last
        
        # 统计缓存
        self._log_buffer: Dict[str, List[float]] = {}
        self._iter_count = 0
        
    def before_run(self, runner: 'RunnerType') -> None:
        """训练开始前初始化"""
        self._log_buffer.clear()
        self._iter_count = 0
        
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """训练迭代后记录日志"""
        self._iter_count += 1
        
        # 收集指标
        if hasattr(runner, 'outputs'):
            for key, value in runner.outputs.items():
                if isinstance(value, (int, float)):
                    if key not in self._log_buffer:
                        self._log_buffer[key] = []
                    self._log_buffer[key].append(value)
                    
        # 按间隔输出
        if self.every_n_iters(runner, self.interval):
            self._log_train_info(runner)
            
    def after_train_epoch(self, runner: 'RunnerType') -> None:
        """训练 epoch 结束后输出汇总"""
        if self.by_epoch:
            self._log_epoch_summary(runner)
            self._log_buffer.clear()
            
    def after_val_epoch(self, runner: 'RunnerType') -> None:
        """验证结束后记录指标"""
        self._log_val_info(runner)
        
    def _log_train_info(self, runner: 'RunnerType') -> None:
        """输出训练信息"""
        # 计算平均值
        avg_metrics = {}
        for key, values in self._log_buffer.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
                
        # 构建日志字符串
        log_str = f"Epoch [{runner.epoch + 1}][{runner.inner_iter + 1}/{runner.max_iters}] "
        
        # 添加指标
        for key, value in avg_metrics.items():
            log_str += f"{key}: {value:.4f} "
            
        # 添加学习率
        if hasattr(runner, 'optimizer'):
            lr = runner.optimizer.param_groups[0]['lr']
            log_str += f"lr: {lr:.6f}"
            
        print(log_str)
        
        # 清空缓存（如果不按 epoch 统计）
        if not self.log_metric_by_epoch:
            self._log_buffer.clear()
            
    def _log_epoch_summary(self, runner: 'RunnerType') -> None:
        """输出 epoch 汇总"""
        avg_metrics = {}
        for key, values in self._log_buffer.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
                
        log_str = f"Epoch [{runner.epoch + 1}] Summary: "
        for key, value in avg_metrics.items():
            log_str += f"{key}: {value:.4f} "
            
        print(log_str)
        
    def _log_val_info(self, runner: 'RunnerType') -> None:
        """输出验证信息"""
        if hasattr(runner, 'val_outputs'):
            log_str = f"Validation: "
            for key, value in runner.val_outputs.items():
                if isinstance(value, (int, float)):
                    log_str += f"{key}: {value:.4f} "
            print(log_str)


class TensorBoardHook(BaseHook):
    """
    TensorBoard 日志钩子
    
    Args:
        log_dir: 日志目录
        interval: 记录间隔
        log_graph: 是否记录模型图
        log_images: 是否记录图像
        image_interval: 图像记录间隔
        
    Example:
        >>> hook = TensorBoardHook(log_dir='runs', interval=100)
    """
    
    priority = HookPriority.NORMAL
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        interval: int = 50,
        log_graph: bool = False,
        log_images: bool = False,
        image_interval: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.interval = interval
        self.log_graph = log_graph
        self.log_images = log_images
        self.image_interval = image_interval
        
        self._writer = None
        self._graph_logged = False
        
    def before_run(self, runner: 'RunnerType') -> None:
        """初始化 TensorBoard writer"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            if self.log_dir is None:
                self.log_dir = os.path.join(
                    getattr(runner, 'work_dir', '.'),
                    'tensorboard'
                )
            self._writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("Warning: tensorboard not installed, TensorBoardHook disabled")
            
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """记录训练指标"""
        if self._writer is None:
            return
            
        if not self.every_n_iters(runner, self.interval):
            return
            
        global_step = runner.iter + 1
        
        # 记录标量
        if hasattr(runner, 'outputs'):
            for key, value in runner.outputs.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f'train/{key}', value, global_step)
                    
        # 记录学习率
        if hasattr(runner, 'optimizer'):
            lr = runner.optimizer.param_groups[0]['lr']
            self._writer.add_scalar('train/lr', lr, global_step)
            
        # 记录模型图
        if self.log_graph and not self._graph_logged:
            self._log_model_graph(runner)
            
    def after_val_epoch(self, runner: 'RunnerType') -> None:
        """记录验证指标"""
        if self._writer is None:
            return
            
        global_step = runner.epoch + 1
        
        if hasattr(runner, 'val_outputs'):
            for key, value in runner.val_outputs.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f'val/{key}', value, global_step)
                    
    def after_run(self, runner: 'RunnerType') -> None:
        """关闭 writer"""
        if self._writer is not None:
            self._writer.close()
            
    def _log_model_graph(self, runner: 'RunnerType') -> None:
        """记录模型图"""
        try:
            import torch
            if hasattr(runner, 'model') and hasattr(runner, 'data_batch'):
                inputs = runner.data_batch.get('inputs')
                if inputs is not None:
                    self._writer.add_graph(runner.model, inputs)
                    self._graph_logged = True
        except Exception as e:
            print(f"Failed to log model graph: {e}")


class WandbHook(BaseHook):
    """
    Weights & Biases 日志钩子
    
    Args:
        project: W&B 项目名
        name: 运行名称
        config: 配置字典
        log_model: 是否上传模型
        interval: 记录间隔
        log_code: 是否记录代码
        
    Example:
        >>> hook = WandbHook(project='my-project', name='exp-001')
    """
    
    priority = HookPriority.NORMAL
    
    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_model: bool = False,
        interval: int = 50,
        log_code: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.project = project
        self.name = name
        self.config = config
        self.log_model = log_model
        self.interval = interval
        self.log_code = log_code
        
        self._run = None
        
    def before_run(self, runner: 'RunnerType') -> None:
        """初始化 W&B"""
        try:
            import wandb
            
            self._run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                reinit=True
            )
            
            if self.log_code:
                wandb.run.log_code(".")
        except ImportError:
            print("Warning: wandb not installed, WandbHook disabled")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """记录训练指标"""
        if self._run is None:
            return
            
        if not self.every_n_iters(runner, self.interval):
            return
            
        import wandb
        
        log_dict = {'iter': runner.iter + 1, 'epoch': runner.epoch + 1}
        
        # 记录指标
        if hasattr(runner, 'outputs'):
            for key, value in runner.outputs.items():
                if isinstance(value, (int, float)):
                    log_dict[f'train/{key}'] = value
                    
        # 记录学习率
        if hasattr(runner, 'optimizer'):
            lr = runner.optimizer.param_groups[0]['lr']
            log_dict['train/lr'] = lr
            
        wandb.log(log_dict)
        
    def after_val_epoch(self, runner: 'RunnerType') -> None:
        """记录验证指标"""
        if self._run is None:
            return
            
        import wandb
        
        log_dict = {'epoch': runner.epoch + 1}
        
        if hasattr(runner, 'val_outputs'):
            for key, value in runner.val_outputs.items():
                if isinstance(value, (int, float)):
                    log_dict[f'val/{key}'] = value
                    
        wandb.log(log_dict)
        
    def after_run(self, runner: 'RunnerType') -> None:
        """结束 W&B 运行"""
        if self._run is not None:
            import wandb
            
            # 上传模型
            if self.log_model and hasattr(runner, 'work_dir'):
                best_path = os.path.join(runner.work_dir, 'best.pth')
                if os.path.exists(best_path):
                    artifact = wandb.Artifact('model', type='model')
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact)
                    
            wandb.finish()


class MLflowHook(BaseHook):
    """
    MLflow 日志钩子
    
    Args:
        experiment_name: 实验名称
        tracking_uri: MLflow 服务器 URI
        run_name: 运行名称
        interval: 记录间隔
        log_model: 是否记录模型
        
    Example:
        >>> hook = MLflowHook(experiment_name='my-experiment')
    """
    
    priority = HookPriority.NORMAL
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        interval: int = 50,
        log_model: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.interval = interval
        self.log_model = log_model
        
        self._run = None
        
    def before_run(self, runner: 'RunnerType') -> None:
        """初始化 MLflow"""
        try:
            import mlflow
            
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
                
            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)
                
            self._run = mlflow.start_run(run_name=self.run_name)
        except ImportError:
            print("Warning: mlflow not installed, MLflowHook disabled")
        except Exception as e:
            print(f"Warning: Failed to initialize mlflow: {e}")
            
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """记录训练指标"""
        if self._run is None:
            return
            
        if not self.every_n_iters(runner, self.interval):
            return
            
        import mlflow
        
        step = runner.iter + 1
        
        # 记录指标
        if hasattr(runner, 'outputs'):
            for key, value in runner.outputs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f'train_{key}', value, step=step)
                    
        # 记录学习率
        if hasattr(runner, 'optimizer'):
            lr = runner.optimizer.param_groups[0]['lr']
            mlflow.log_metric('train_lr', lr, step=step)
            
    def after_val_epoch(self, runner: 'RunnerType') -> None:
        """记录验证指标"""
        if self._run is None:
            return
            
        import mlflow
        
        step = runner.epoch + 1
        
        if hasattr(runner, 'val_outputs'):
            for key, value in runner.val_outputs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f'val_{key}', value, step=step)
                    
    def after_run(self, runner: 'RunnerType') -> None:
        """结束 MLflow 运行"""
        if self._run is not None:
            import mlflow
            
            # 记录模型
            if self.log_model and hasattr(runner, 'model'):
                try:
                    mlflow.pytorch.log_model(runner.model, 'model')
                except Exception as e:
                    print(f"Failed to log model: {e}")
                    
            mlflow.end_run()


# 注册钩子（如果尚未注册）
def _safe_register(name, cls, **kwargs):
    if not HOOKS.contains(name):
        HOOKS.register(name, cls, **kwargs)
    else:
        HOOKS._components[name].cls = cls

_safe_register('LoggerHook', LoggerHook, 
               priority=HookPriority.NORMAL, category='logging')
_safe_register('TensorBoardHook', TensorBoardHook, 
               priority=HookPriority.NORMAL, category='logging')
_safe_register('WandbHook', WandbHook, 
               priority=HookPriority.NORMAL, category='logging')
_safe_register('MLflowHook', MLflowHook, 
               priority=HookPriority.NORMAL, category='logging')
