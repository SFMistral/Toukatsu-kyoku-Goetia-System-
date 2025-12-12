# -*- coding: utf-8 -*-
"""
性能分析钩子模块

提供 GPU/CPU 性能分析、内存监控、迭代计时等功能。
"""

import os
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING

import torch

from .base_hook import BaseHook, HookPriority
from registry import HOOKS

if TYPE_CHECKING:
    from typing import Any as RunnerType


class ProfilerHook(BaseHook):
    """
    PyTorch Profiler 钩子
    
    使用 PyTorch Profiler 进行性能分析。
    
    Args:
        by_epoch: 按 epoch 分析
        profile_iters: 分析的迭代数
        schedule: 调度配置 (wait, warmup, active, repeat)
        on_trace_ready: 追踪回调
        record_shapes: 记录张量形状
        profile_memory: 分析内存
        with_stack: 记录调用栈
        with_flops: 计算 FLOPs
        export_chrome_trace: 导出 Chrome 追踪
        export_tensorboard: 导出到 TensorBoard
        
    Example:
        >>> hook = ProfilerHook(
        ...     profile_iters=100,
        ...     profile_memory=True,
        ...     export_chrome_trace=True
        ... )
    """
    
    priority = HookPriority.VERY_HIGH
    
    def __init__(
        self,
        by_epoch: bool = False,
        profile_iters: int = 100,
        schedule: Optional[Dict[str, int]] = None,
        on_trace_ready: Optional[callable] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        export_chrome_trace: bool = True,
        export_tensorboard: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.by_epoch = by_epoch
        self.profile_iters = profile_iters
        self.schedule = schedule or {'wait': 1, 'warmup': 1, 'active': 3, 'repeat': 1}
        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.export_chrome_trace = export_chrome_trace
        self.export_tensorboard = export_tensorboard
        
        self._profiler = None
        self._out_dir = None
        
    def before_run(self, runner: 'RunnerType') -> None:
        """初始化 profiler"""
        self._out_dir = os.path.join(
            getattr(runner, 'work_dir', '.'),
            'profiler'
        )
        os.makedirs(self._out_dir, exist_ok=True)
        
        # 创建调度
        schedule = torch.profiler.schedule(
            wait=self.schedule['wait'],
            warmup=self.schedule['warmup'],
            active=self.schedule['active'],
            repeat=self.schedule.get('repeat', 1)
        )
        
        # 创建追踪回调
        if self.on_trace_ready is None:
            self.on_trace_ready = self._default_trace_handler
            
        # 创建 profiler
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self.on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops
        )
        self._profiler.__enter__()
        
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """每次迭代后步进 profiler"""
        if self._profiler is not None and runner.iter < self.profile_iters:
            self._profiler.step()
            
    def after_run(self, runner: 'RunnerType') -> None:
        """训练结束后关闭 profiler"""
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            self._generate_report()
            
    def _default_trace_handler(self, prof) -> None:
        """默认追踪处理器"""
        if self.export_chrome_trace:
            trace_path = os.path.join(
                self._out_dir, 
                f'trace_{prof.step_num}.json'
            )
            prof.export_chrome_trace(trace_path)
            
        if self.export_tensorboard:
            tb_path = os.path.join(self._out_dir, 'tensorboard')
            prof.export_stacks(tb_path, 'self_cuda_time_total')
            
    def _generate_report(self) -> None:
        """生成性能报告"""
        if self._profiler is None:
            return
            
        report_path = os.path.join(self._out_dir, 'profiler_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Profiler Report\n")
            f.write("=" * 80 + "\n\n")
            
            # CPU 时间排序
            f.write("Top 20 operations by CPU time:\n")
            f.write("-" * 40 + "\n")
            f.write(self._profiler.key_averages().table(
                sort_by="cpu_time_total", row_limit=20
            ))
            f.write("\n\n")
            
            # CUDA 时间排序
            if torch.cuda.is_available():
                f.write("Top 20 operations by CUDA time:\n")
                f.write("-" * 40 + "\n")
                f.write(self._profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20
                ))
                
        print(f"Profiler report saved to {report_path}")


class MemoryProfilerHook(BaseHook):
    """
    内存分析钩子
    
    监控 GPU/CPU 内存使用情况。
    
    Args:
        interval: 记录间隔
        log_gpu: 记录 GPU 内存
        log_cpu: 记录 CPU 内存
        warn_threshold: 警告阈值（GB）
        
    Example:
        >>> hook = MemoryProfilerHook(interval=100, warn_threshold=10.0)
    """
    
    priority = HookPriority.VERY_HIGH
    
    def __init__(
        self,
        interval: int = 50,
        log_gpu: bool = True,
        log_cpu: bool = True,
        warn_threshold: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interval = interval
        self.log_gpu = log_gpu
        self.log_cpu = log_cpu
        self.warn_threshold = warn_threshold
        
        self._memory_history: List[Dict[str, float]] = []
        
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """记录内存使用"""
        if not self.every_n_iters(runner, self.interval):
            return
            
        memory_info = self._get_memory_info()
        memory_info['iter'] = runner.iter + 1
        self._memory_history.append(memory_info)
        
        # 输出日志
        log_str = f"[Memory] Iter {runner.iter + 1}: "
        
        if self.log_gpu and 'gpu_allocated' in memory_info:
            log_str += f"GPU: {memory_info['gpu_allocated']:.2f}GB "
            log_str += f"(peak: {memory_info['gpu_peak']:.2f}GB) "
            
        if self.log_cpu and 'cpu_used' in memory_info:
            log_str += f"CPU: {memory_info['cpu_used']:.2f}GB"
            
        print(log_str)
        
        # 检查警告阈值
        if self.warn_threshold is not None:
            if memory_info.get('gpu_allocated', 0) > self.warn_threshold:
                print(f"Warning: GPU memory usage exceeds {self.warn_threshold}GB!")
                
    def after_run(self, runner: 'RunnerType') -> None:
        """输出内存统计摘要"""
        if not self._memory_history:
            return
            
        print("\n" + "=" * 50)
        print("Memory Usage Summary")
        print("=" * 50)
        
        if self.log_gpu:
            gpu_peaks = [m.get('gpu_peak', 0) for m in self._memory_history]
            if gpu_peaks:
                print(f"GPU Peak Memory: {max(gpu_peaks):.2f}GB")
                
        if self.log_cpu:
            cpu_peaks = [m.get('cpu_used', 0) for m in self._memory_history]
            if cpu_peaks:
                print(f"CPU Peak Memory: {max(cpu_peaks):.2f}GB")
                
    def _get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        info = {}
        
        # GPU 内存
        if self.log_gpu and torch.cuda.is_available():
            info['gpu_allocated'] = torch.cuda.memory_allocated() / 1e9
            info['gpu_reserved'] = torch.cuda.memory_reserved() / 1e9
            info['gpu_peak'] = torch.cuda.max_memory_allocated() / 1e9
            
        # CPU 内存
        if self.log_cpu:
            try:
                import psutil
                process = psutil.Process()
                info['cpu_used'] = process.memory_info().rss / 1e9
            except ImportError:
                pass
                
        return info


class IterTimerHook(BaseHook):
    """
    迭代计时钩子
    
    记录每次迭代的耗时，计算吞吐量和 ETA。
    
    Args:
        interval: 统计间隔
        log_eta: 记录预计剩余时间
        
    Example:
        >>> hook = IterTimerHook(interval=50, log_eta=True)
    """
    
    priority = HookPriority.VERY_HIGH
    
    def __init__(
        self,
        interval: int = 50,
        log_eta: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interval = interval
        self.log_eta = log_eta
        
        self._start_time: float = 0
        self._iter_start_time: float = 0
        self._data_time: float = 0
        self._iter_times: List[float] = []
        
    def before_run(self, runner: 'RunnerType') -> None:
        """记录训练开始时间"""
        self._start_time = time.time()
        
    def before_train_iter(self, runner: 'RunnerType') -> None:
        """记录迭代开始时间"""
        self._iter_start_time = time.time()
        
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """记录迭代耗时"""
        iter_time = time.time() - self._iter_start_time
        self._iter_times.append(iter_time)
        
        if not self.every_n_iters(runner, self.interval):
            return
            
        # 计算统计信息
        recent_times = self._iter_times[-self.interval:]
        avg_time = sum(recent_times) / len(recent_times)
        
        # 计算吞吐量
        batch_size = getattr(runner, 'batch_size', 1)
        throughput = batch_size / avg_time
        
        log_str = f"[Timer] Iter {runner.iter + 1}: "
        log_str += f"avg_time: {avg_time:.3f}s "
        log_str += f"throughput: {throughput:.1f} samples/s"
        
        # 计算 ETA
        if self.log_eta:
            remaining_iters = runner.max_iters - runner.iter - 1
            eta_seconds = remaining_iters * avg_time
            eta_str = self._format_time(eta_seconds)
            log_str += f" ETA: {eta_str}"
            
        print(log_str)
        
    def after_run(self, runner: 'RunnerType') -> None:
        """输出总耗时"""
        total_time = time.time() - self._start_time
        print(f"\nTotal training time: {self._format_time(total_time)}")
        
        if self._iter_times:
            avg_time = sum(self._iter_times) / len(self._iter_times)
            print(f"Average iteration time: {avg_time:.3f}s")
            
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


# 注册钩子（如果尚未注册）
def _safe_register(name, cls, **kwargs):
    if not HOOKS.contains(name):
        HOOKS.register(name, cls, **kwargs)
    else:
        HOOKS._components[name].cls = cls

_safe_register('ProfilerHook', ProfilerHook, 
               priority=HookPriority.VERY_HIGH, category='profiling')
_safe_register('MemoryProfilerHook', MemoryProfilerHook, 
               priority=HookPriority.VERY_HIGH, category='profiling')
_safe_register('IterTimerHook', IterTimerHook, 
               priority=HookPriority.VERY_HIGH, category='profiling')
