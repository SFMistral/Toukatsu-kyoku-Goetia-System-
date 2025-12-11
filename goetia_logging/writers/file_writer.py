# -*- coding: utf-8 -*-
"""
文件日志写入器

负责将日志写入到文件，支持日志滚动、压缩归档等功能。
"""

import os
import gzip
import shutil
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import WriterBase
from ..formatters import PlainFormatter, FormatterBase


class FileWriter(WriterBase):
    """文件日志写入器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 配置参数
        self.log_dir = config.get('log_dir', 'logs')
        self.filename = config.get('filename', 'train.log')
        self.max_size = config.get('max_size', 10 * 1024 * 1024)  # 10MB
        self.max_files = config.get('max_files', 5)
        self.rotation = config.get('rotation', 'size')  # size, time, both, none
        self.compression = config.get('compression', 'none')  # gz, zip, none
        self.encoding = config.get('encoding', 'utf-8')
        
        # 格式化器
        formatter_config = config.get('formatter', {})
        self.formatter = self._create_formatter(formatter_config)
        
        # 文件路径
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, self.filename)
        
        # 状态
        self.file_handle = None
        self.current_size = 0
        self.creation_time = None
        self._lock = threading.Lock()
        
    def _create_formatter(self, formatter_config: Dict[str, Any]) -> FormatterBase:
        """创建格式化器"""
        from ..formatters import create_formatter
        
        formatter_type = formatter_config.get('type', 'plain')
        return create_formatter(formatter_type, **formatter_config.get('options', {}))
        
    def open(self):
        """打开文件"""
        if self.is_open:
            return
            
        with self._lock:
            # 如果文件已存在，获取当前大小和创建时间
            if os.path.exists(self.log_path):
                self.current_size = os.path.getsize(self.log_path)
                self.creation_time = os.path.getctime(self.log_path)
            else:
                self.current_size = 0
                self.creation_time = None
                
            self.file_handle = open(self.log_path, 'a', encoding=self.encoding)
            self.is_open = True
            
            # 如果是新文件，记录创建时间
            if self.creation_time is None:
                self.creation_time = os.path.getctime(self.log_path)
                
    def close(self):
        """关闭文件"""
        if not self.is_open:
            return
            
        with self._lock:
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
            self.is_open = False
            
    def write_text(self, record: Dict[str, Any]):
        """写入文本日志"""
        if not self.is_open:
            self.open()
            
        try:
            message = self.formatter.format(record)
            message_bytes = (message + '\n').encode(self.encoding)
            
            with self._lock:
                # 检查是否需要滚动
                if self._should_rotate():
                    self._rotate_file()
                    
                self.file_handle.write(message + '\n')
                self.file_handle.flush()
                self.current_size += len(message_bytes)
                
        except Exception as e:
            print(f"FileWriter error: {e}")
            
    def write_scalar(self, tag: str, value: float, step: int):
        """写入标量指标"""
        record = {
            'timestamp': datetime.now().timestamp(),
            'level': 'INFO',
            'message': f"METRIC | {tag}: {value:.6f} (step: {step})",
            'tag': tag,
            'value': value,
            'step': step,
            'type': 'scalar'
        }
        self.write_text(record)
        
    def write_scalars(self, tag_value_dict: Dict[str, float], step: int):
        """写入多个标量指标"""
        for tag, value in tag_value_dict.items():
            self.write_scalar(tag, value, step)
            
    def write_histogram(self, tag: str, values: List[float], step: int):
        """写入直方图数据"""
        import statistics
        
        if not values:
            return
            
        # 计算统计信息
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        min_val = min(values)
        max_val = max(values)
        
        record = {
            'timestamp': datetime.now().timestamp(),
            'level': 'INFO',
            'message': f"HISTOGRAM | {tag}: mean={mean_val:.6f}, std={std_val:.6f}, "
                      f"min={min_val:.6f}, max={max_val:.6f}, count={len(values)} (step: {step})",
            'tag': tag,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'count': len(values),
            'step': step,
            'type': 'histogram'
        }
        self.write_text(record)
        
    def write_hyperparams(self, params: Dict[str, Any]):
        """写入超参数"""
        record = {
            'timestamp': datetime.now().timestamp(),
            'level': 'INFO',
            'message': f"HYPERPARAMS | {params}",
            'params': params,
            'type': 'hyperparams'
        }
        self.write_text(record)
        
    def flush(self):
        """刷新文件缓冲区"""
        if self.is_open and self.file_handle:
            with self._lock:
                self.file_handle.flush()
                os.fsync(self.file_handle.fileno())
                
    def _should_rotate(self) -> bool:
        """判断是否需要滚动文件"""
        if self.rotation == 'none':
            return False
            
        # 按大小滚动
        if self.rotation in ('size', 'both'):
            if self.current_size >= self.max_size:
                return True
                
        # 按时间滚动（每日）
        if self.rotation in ('time', 'both'):
            if self.creation_time:
                current_day = datetime.now().strftime('%Y%m%d')
                creation_day = datetime.fromtimestamp(self.creation_time).strftime('%Y%m%d')
                if current_day != creation_day:
                    return True
                    
        return False
        
    def _rotate_file(self):
        """执行文件滚动"""
        # 关闭当前文件
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            
        # 生成备份文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{self.filename}.{timestamp}"
        backup_path = os.path.join(self.log_dir, backup_name)
        
        # 移动当前文件
        if os.path.exists(self.log_path):
            shutil.move(self.log_path, backup_path)
            
            # 压缩文件
            if self.compression == 'gz':
                self._compress_file(backup_path, backup_path + '.gz')
                os.remove(backup_path)
                backup_path += '.gz'
            elif self.compression == 'zip':
                import zipfile
                zip_path = backup_path + '.zip'
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(backup_path, os.path.basename(backup_path))
                os.remove(backup_path)
                backup_path = zip_path
                
        # 清理旧文件
        self._cleanup_old_files()
        
        # 重新打开文件
        self.current_size = 0
        self.creation_time = None
        self.file_handle = open(self.log_path, 'w', encoding=self.encoding)
        self.creation_time = os.path.getctime(self.log_path)
        
    def _compress_file(self, source_path: str, target_path: str):
        """压缩文件"""
        with open(source_path, 'rb') as f_in:
            with gzip.open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    def _cleanup_old_files(self):
        """清理旧的日志文件"""
        if self.max_files <= 0:
            return
            
        # 获取所有相关的备份文件
        backup_files = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith(self.filename + '.'):
                file_path = os.path.join(self.log_dir, filename)
                if os.path.isfile(file_path):
                    backup_files.append((file_path, os.path.getctime(file_path)))
                    
        # 按创建时间排序
        backup_files.sort(key=lambda x: x[1], reverse=True)
        
        # 删除超出数量限制的文件
        for file_path, _ in backup_files[self.max_files:]:
            try:
                os.remove(file_path)
            except OSError:
                pass