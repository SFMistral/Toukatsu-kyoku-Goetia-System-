# -*- coding: utf-8 -*-
"""
JSON格式日志写入器

负责将日志以JSON格式写入文件，便于日志采集系统解析和分析。
"""

import os
import json
import threading
from typing import Dict, Any, List, Optional, TextIO
from datetime import datetime

from .base import WriterBase


class JsonWriter(WriterBase):
    """JSON格式日志写入器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 配置参数
        self.log_dir = config.get('log_dir', 'logs')
        self.filename = config.get('filename', 'logs.jsonl')
        self.metrics_filename = config.get('metrics_filename', 'metrics.jsonl')
        self.indent = config.get('indent', None)  # None为紧凑格式
        self.ensure_ascii = config.get('ensure_ascii', False)
        self.timestamp_format = config.get('timestamp_format', 'iso')  # iso, unix, unix_ms
        self.encoding = config.get('encoding', 'utf-8')
        
        # 字段过滤
        self.include_fields = config.get('include_fields', [])
        self.exclude_fields = config.get('exclude_fields', [])
        
        # 分离指标和普通日志
        self.separate_metrics = config.get('separate_metrics', True)
        
        # 文件路径
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, self.filename)
        self.metrics_path = os.path.join(self.log_dir, self.metrics_filename)
        
        # 文件句柄
        self.log_file: Optional[TextIO] = None
        self.metrics_file: Optional[TextIO] = None
        
        # 线程安全
        self._lock = threading.Lock()
        
    def open(self):
        """打开JSON文件"""
        if self.is_open:
            return
            
        with self._lock:
            self.log_file = open(self.log_path, 'a', encoding=self.encoding)
            
            if self.separate_metrics:
                self.metrics_file = open(self.metrics_path, 'a', encoding=self.encoding)
                
            self.is_open = True
            
    def close(self):
        """关闭JSON文件"""
        if not self.is_open:
            return
            
        with self._lock:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
                
            if self.metrics_file:
                self.metrics_file.close()
                self.metrics_file = None
                
            self.is_open = False
            
    def write_text(self, record: Dict[str, Any]):
        """写入文本日志"""
        if not self.is_open:
            self.open()
            
        try:
            # 处理记录
            json_record = self._prepare_record(record)
            json_line = self._serialize_record(json_record)
            
            with self._lock:
                self.log_file.write(json_line + '\n')
                self.log_file.flush()
                
        except Exception as e:
            print(f"JsonWriter error: {e}")
            
    def write_scalar(self, tag: str, value: float, step: int):
        """写入标量指标"""
        record = {
            'timestamp': datetime.now().timestamp(),
            'type': 'scalar',
            'tag': tag,
            'value': value,
            'step': step
        }
        
        if self.separate_metrics:
            self._write_metric_record(record)
        else:
            # 作为普通日志记录
            text_record = {
                'timestamp': record['timestamp'],
                'level': 'INFO',
                'message': f"METRIC | {tag}: {value}",
                'logger_name': 'metrics',
                'metric_type': 'scalar',
                'metric_tag': tag,
                'metric_value': value,
                'metric_step': step
            }
            self.write_text(text_record)
            
    def write_scalars(self, tag_value_dict: Dict[str, float], step: int):
        """写入多个标量指标"""
        if self.separate_metrics:
            # 批量写入指标文件
            records = []
            timestamp = datetime.now().timestamp()
            
            for tag, value in tag_value_dict.items():
                record = {
                    'timestamp': timestamp,
                    'type': 'scalar',
                    'tag': tag,
                    'value': value,
                    'step': step
                }
                records.append(record)
                
            self._write_metric_records(records)
        else:
            # 逐个写入
            for tag, value in tag_value_dict.items():
                self.write_scalar(tag, value, step)
                
    def write_histogram(self, tag: str, values: List[float], step: int):
        """写入直方图数据"""
        if not values:
            return
            
        import statistics
        
        # 计算统计信息
        stats = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
        
        record = {
            'timestamp': datetime.now().timestamp(),
            'type': 'histogram',
            'tag': tag,
            'step': step,
            'statistics': stats
        }
        
        # 可选：保存原始数据（注意文件大小）
        if len(values) <= 1000:  # 限制数据量
            record['values'] = values
            
        if self.separate_metrics:
            self._write_metric_record(record)
        else:
            # 作为普通日志记录
            text_record = {
                'timestamp': record['timestamp'],
                'level': 'INFO',
                'message': f"HISTOGRAM | {tag}: {stats}",
                'logger_name': 'metrics',
                'metric_type': 'histogram',
                'metric_tag': tag,
                'metric_step': step,
                'metric_stats': stats
            }
            self.write_text(text_record)
            
    def write_hyperparams(self, params: Dict[str, Any]):
        """写入超参数"""
        record = {
            'timestamp': datetime.now().timestamp(),
            'type': 'hyperparams',
            'params': params
        }
        
        if self.separate_metrics:
            self._write_metric_record(record)
        else:
            # 作为普通日志记录
            text_record = {
                'timestamp': record['timestamp'],
                'level': 'INFO',
                'message': f"HYPERPARAMS | {params}",
                'logger_name': 'hyperparams',
                'hyperparams': params
            }
            self.write_text(text_record)
            
    def flush(self):
        """刷新文件缓冲区"""
        with self._lock:
            if self.log_file:
                self.log_file.flush()
                os.fsync(self.log_file.fileno())
                
            if self.metrics_file:
                self.metrics_file.flush()
                os.fsync(self.metrics_file.fileno())
                
    def _prepare_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """准备记录用于JSON序列化"""
        # 复制记录
        json_record = record.copy()
        
        # 格式化时间戳
        if 'timestamp' in json_record:
            timestamp = json_record['timestamp']
            if self.timestamp_format == 'iso':
                json_record['timestamp'] = datetime.fromtimestamp(timestamp).isoformat()
            elif self.timestamp_format == 'unix':
                json_record['timestamp'] = int(timestamp)
            elif self.timestamp_format == 'unix_ms':
                json_record['timestamp'] = int(timestamp * 1000)
            # 否则保持原格式
            
        # 应用字段过滤
        if self.include_fields:
            json_record = {k: v for k, v in json_record.items() if k in self.include_fields}
        if self.exclude_fields:
            json_record = {k: v for k, v in json_record.items() if k not in self.exclude_fields}
            
        # 确保JSON可序列化
        json_record = self._make_json_serializable(json_record)
        
        return json_record
        
    def _serialize_record(self, record: Dict[str, Any]) -> str:
        """序列化记录为JSON字符串"""
        try:
            return json.dumps(
                record,
                indent=self.indent,
                ensure_ascii=self.ensure_ascii,
                separators=(',', ':') if self.indent is None else None,
                default=str  # 处理无法序列化的对象
            )
        except (TypeError, ValueError) as e:
            # 序列化失败时的回退处理
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'level': 'ERROR',
                'message': f'JSON serialization failed: {e}',
                'original_record': str(record)
            }
            return json.dumps(error_record)
            
    def _make_json_serializable(self, obj):
        """确保对象可以JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif hasattr(obj, 'isoformat'):  # datetime对象
            return obj.isoformat()
        else:
            return str(obj)
            
    def _write_metric_record(self, record: Dict[str, Any]):
        """写入单个指标记录"""
        if not self.is_open:
            self.open()
            
        try:
            json_record = self._prepare_record(record)
            json_line = self._serialize_record(json_record)
            
            with self._lock:
                if self.metrics_file:
                    self.metrics_file.write(json_line + '\n')
                    self.metrics_file.flush()
                else:
                    # 回退到普通日志文件
                    self.log_file.write(json_line + '\n')
                    self.log_file.flush()
                    
        except Exception as e:
            print(f"JsonWriter metric error: {e}")
            
    def _write_metric_records(self, records: List[Dict[str, Any]]):
        """批量写入指标记录"""
        if not self.is_open:
            self.open()
            
        try:
            with self._lock:
                for record in records:
                    json_record = self._prepare_record(record)
                    json_line = self._serialize_record(json_record)
                    
                    if self.metrics_file:
                        self.metrics_file.write(json_line + '\n')
                    else:
                        self.log_file.write(json_line + '\n')
                        
                # 批量刷新
                if self.metrics_file:
                    self.metrics_file.flush()
                else:
                    self.log_file.flush()
                    
        except Exception as e:
            print(f"JsonWriter batch metric error: {e}")
            
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志文件统计信息"""
        stats = {}
        
        if os.path.exists(self.log_path):
            stats['log_file'] = {
                'path': self.log_path,
                'size': os.path.getsize(self.log_path),
                'lines': self._count_lines(self.log_path)
            }
            
        if os.path.exists(self.metrics_path):
            stats['metrics_file'] = {
                'path': self.metrics_path,
                'size': os.path.getsize(self.metrics_path),
                'lines': self._count_lines(self.metrics_path)
            }
            
        return stats
        
    def _count_lines(self, file_path: str) -> int:
        """计算文件行数"""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                return sum(1 for _ in f)
        except Exception:
            return 0