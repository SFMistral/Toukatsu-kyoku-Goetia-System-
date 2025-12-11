# -*- coding: utf-8 -*-
"""
控制台日志写入器

负责将日志输出到终端，支持彩色输出、进度条集成、表格显示等功能。
"""

import sys
import threading
from typing import Dict, Any, List, Optional, TextIO
from datetime import datetime

from .base import WriterBase
from ..formatters import ColorFormatter, TableFormatter, FormatterBase


class ConsoleWriter(WriterBase):
    """控制台日志写入器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 配置参数
        self.level = config.get('level', 'INFO')
        self.show_progress = config.get('show_progress', True)
        self.use_colors = config.get('use_colors', True)
        self.output_stream = config.get('output_stream', 'stdout')  # stdout, stderr
        self.table_mode = config.get('table_mode', False)
        
        # 输出流
        self.stream: TextIO = sys.stdout if self.output_stream == 'stdout' else sys.stderr
        
        # 格式化器
        formatter_config = config.get('formatter', {})
        self.formatter = self._create_formatter(formatter_config)
        
        # 表格格式化器（用于指标汇总）
        if self.table_mode:
            table_config = config.get('table', {})
            self.table_formatter = TableFormatter(**table_config)
            self.metrics_buffer = {}
            self.table_header_shown = False
        
        # 进度条集成
        self.progress_bar = None
        self.tqdm_available = self._check_tqdm()
        
        # 线程安全
        self._lock = threading.Lock()
        
    def _create_formatter(self, formatter_config: Dict[str, Any]) -> FormatterBase:
        """创建格式化器"""
        from ..formatters import create_formatter
        
        formatter_type = formatter_config.get('type', 'color' if self.use_colors else 'plain')
        formatter_options = formatter_config.get('options', {})
        
        if formatter_type == 'color':
            formatter_options['use_colors'] = self.use_colors
            
        return create_formatter(formatter_type, **formatter_options)
        
    def _check_tqdm(self) -> bool:
        """检查是否可用tqdm"""
        try:
            import tqdm
            return True
        except ImportError:
            return False
            
    def open(self):
        """打开控制台输出"""
        self.is_open = True
        
    def close(self):
        """关闭控制台输出"""
        if self.table_mode and self.metrics_buffer:
            self._flush_table()
        self.is_open = False
        
    def write_text(self, record: Dict[str, Any]):
        """写入文本日志"""
        if not self.is_open:
            return
            
        try:
            message = self.formatter.format(record)
            
            with self._lock:
                if self.show_progress and self.tqdm_available:
                    # 使用tqdm的写入方法，避免与进度条冲突
                    try:
                        import tqdm
                        tqdm.tqdm.write(message, file=self.stream)
                    except:
                        self._write_direct(message)
                else:
                    self._write_direct(message)
                    
        except Exception as e:
            print(f"ConsoleWriter error: {e}", file=sys.stderr)
            
    def _write_direct(self, message: str):
        """直接写入流"""
        self.stream.write(message + '\n')
        self.stream.flush()
        
    def write_scalar(self, tag: str, value: float, step: int):
        """写入标量指标"""
        if self.table_mode:
            # 缓存指标用于表格显示
            self.metrics_buffer[tag] = value
            if 'step' not in self.metrics_buffer:
                self.metrics_buffer['step'] = step
        else:
            # 直接输出
            record = {
                'timestamp': datetime.now().timestamp(),
                'level': 'INFO',
                'message': f"{tag}: {value:.6f}",
                'logger_name': 'metrics'
            }
            self.write_text(record)
            
    def write_scalars(self, tag_value_dict: Dict[str, float], step: int):
        """写入多个标量指标"""
        if self.table_mode:
            # 更新缓存
            self.metrics_buffer.update(tag_value_dict)
            self.metrics_buffer['step'] = step
        else:
            # 逐个输出
            for tag, value in tag_value_dict.items():
                self.write_scalar(tag, value, step)
                
    def write_histogram(self, tag: str, values: List[float], step: int):
        """写入直方图数据"""
        if not values:
            return
            
        import statistics
        
        # 计算统计信息
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        min_val = min(values)
        max_val = max(values)
        
        if self.table_mode:
            # 添加到缓存
            self.metrics_buffer[f"{tag}_mean"] = mean_val
            self.metrics_buffer[f"{tag}_std"] = std_val
            self.metrics_buffer['step'] = step
        else:
            # 直接输出
            record = {
                'timestamp': datetime.now().timestamp(),
                'level': 'INFO',
                'message': f"{tag}: mean={mean_val:.4f}±{std_val:.4f}, "
                          f"range=[{min_val:.4f}, {max_val:.4f}], count={len(values)}",
                'logger_name': 'metrics'
            }
            self.write_text(record)
            
    def write_hyperparams(self, params: Dict[str, Any]):
        """写入超参数"""
        record = {
            'timestamp': datetime.now().timestamp(),
            'level': 'INFO',
            'message': f"Hyperparameters: {params}",
            'logger_name': 'hyperparams'
        }
        self.write_text(record)
        
    def flush(self):
        """刷新输出"""
        if self.table_mode and self.metrics_buffer:
            self._flush_table()
            
        with self._lock:
            self.stream.flush()
            
    def _flush_table(self):
        """刷新表格缓存"""
        if not self.metrics_buffer:
            return
            
        with self._lock:
            # 显示表格头（仅第一次）
            if not self.table_header_shown:
                header = self.table_formatter.format_header()
                self._write_direct(header)
                self.table_header_shown = True
                
            # 格式化并输出表格行
            table_row = self.table_formatter.format(self.metrics_buffer)
            self._write_direct(table_row)
            
            # 清空缓存
            self.metrics_buffer.clear()
            
    def set_progress_bar(self, pbar):
        """设置进度条对象"""
        self.progress_bar = pbar
        
    def update_progress(self, **kwargs):
        """更新进度条"""
        if self.progress_bar and hasattr(self.progress_bar, 'set_postfix'):
            self.progress_bar.set_postfix(**kwargs)
            
    def write_progress_metrics(self, metrics: Dict[str, float]):
        """写入进度条指标"""
        if self.show_progress and self.progress_bar:
            # 格式化指标用于进度条显示
            formatted_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.001:
                        formatted_metrics[key] = f"{value:.2e}"
                    else:
                        formatted_metrics[key] = f"{value:.4f}"
                else:
                    formatted_metrics[key] = str(value)
                    
            self.update_progress(**formatted_metrics)
        else:
            # 回退到普通输出
            self.write_scalars(metrics, 0)
            
    def create_progress_bar(self, total: int, desc: str = "Progress"):
        """创建进度条"""
        if not self.tqdm_available or not self.show_progress:
            return None
            
        try:
            import tqdm
            pbar = tqdm.tqdm(total=total, desc=desc, file=self.stream)
            self.set_progress_bar(pbar)
            return pbar
        except Exception:
            return None
            
    def write_separator(self, char: str = "=", length: int = 80):
        """写入分隔线"""
        separator = char * length
        record = {
            'timestamp': datetime.now().timestamp(),
            'level': 'INFO',
            'message': separator,
            'logger_name': 'separator'
        }
        self.write_text(record)
        
    def write_banner(self, text: str, char: str = "=", width: int = 80):
        """写入横幅文本"""
        # 计算填充
        text_len = len(text)
        if text_len >= width - 4:
            banner = f"{char}{char} {text} {char}{char}"
        else:
            padding = (width - text_len - 4) // 2
            banner = f"{char}{char}{' ' * padding} {text} {' ' * padding}{char}{char}"
            
        record = {
            'timestamp': datetime.now().timestamp(),
            'level': 'INFO',
            'message': banner,
            'logger_name': 'banner'
        }
        self.write_text(record)
        
    def write_section(self, title: str):
        """写入章节标题"""
        self.write_separator("-", 60)
        record = {
            'timestamp': datetime.now().timestamp(),
            'level': 'INFO',
            'message': f"  {title}",
            'logger_name': 'section'
        }
        self.write_text(record)
        self.write_separator("-", 60)