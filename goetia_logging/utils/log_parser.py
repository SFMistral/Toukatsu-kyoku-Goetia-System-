# -*- coding: utf-8 -*-
"""
日志解析器

解析各种格式的日志文件，提取结构化信息和指标数据。
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Union, Iterator, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class LogRecord:
    """日志记录数据类"""
    timestamp: float
    level: str
    logger_name: str
    message: str
    raw_line: str
    line_number: int
    extra_fields: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_fields is None:
            self.extra_fields = {}


@dataclass
class MetricRecord:
    """指标记录数据类"""
    timestamp: float
    step: int
    tag: str
    value: float
    metric_type: str = 'scalar'
    extra_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_data is None:
            self.extra_data = {}


class LogParser:
    """日志解析器"""
    
    def __init__(self):
        # 预编译的正则表达式
        self.patterns = {
            # 标准日志格式: 2023-12-11 10:30:45 [INFO] logger_name: message
            'standard': re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'
                r'\[(?P<level>\w+)\]\s+'
                r'(?P<logger_name>\w+):\s+'
                r'(?P<message>.*)'
            ),
            
            # 指标格式: METRIC | tag: value (step: N)
            'metric': re.compile(
                r'METRIC\s*\|\s*(?P<tag>[\w/._-]+):\s*(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*'
                r'(?:\(step:\s*(?P<step>\d+)\))?'
            ),
            
            # 直方图格式: HISTOGRAM | tag: mean=X, std=Y, ...
            'histogram': re.compile(
                r'HISTOGRAM\s*\|\s*(?P<tag>[\w/._-]+):\s*'
                r'mean=(?P<mean>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?),?\s*'
                r'std=(?P<std>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?),?\s*'
                r'min=(?P<min>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?),?\s*'
                r'max=(?P<max>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?),?\s*'
                r'count=(?P<count>\d+)\s*'
                r'(?:\(step:\s*(?P<step>\d+)\))?'
            ),
            
            # 超参数格式: HYPERPARAMS | {...}
            'hyperparams': re.compile(
                r'HYPERPARAMS\s*\|\s*(?P<params>.*)'
            ),
            
            # 进度格式: Epoch N | Step M | ...
            'progress': re.compile(
                r'(?:Epoch\s+(?P<epoch>\d+))?\s*'
                r'(?:\|\s*)?(?:Step\s+(?P<step>\d+))?\s*'
                r'(?:\|\s*)?(?P<message>.*)'
            ),
            
            # 错误格式
            'error': re.compile(
                r'(?P<error_type>\w+Error|Exception):\s*(?P<error_message>.*)'
            ),
        }
        
        # 时间戳格式
        self.timestamp_formats = [
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d_%H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
        ]
        
    def parse_file(self, 
                   file_path: str,
                   encoding: str = 'utf-8',
                   start_line: int = 1,
                   end_line: Optional[int] = None,
                   level_filter: Optional[str] = None,
                   keyword_filter: Optional[str] = None) -> Dict[str, Any]:
        """解析日志文件"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Log file not found: {file_path}")
            
        results = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'records': [],
            'metrics': [],
            'errors': [],
            'statistics': {},
            'parsing_errors': []
        }
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line_num, line in enumerate(f, 1):
                    # 跳过行号范围外的行
                    if line_num < start_line:
                        continue
                    if end_line and line_num > end_line:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        # 尝试解析JSON格式
                        if line.startswith('{'):
                            record = self._parse_json_line(line, line_num)
                        else:
                            # 解析文本格式
                            record = self._parse_text_line(line, line_num)
                            
                        if record:
                            # 应用过滤器
                            if self._should_include_record(record, level_filter, keyword_filter):
                                if isinstance(record, LogRecord):
                                    results['records'].append(record)
                                elif isinstance(record, MetricRecord):
                                    results['metrics'].append(record)
                                    
                    except Exception as e:
                        results['parsing_errors'].append({
                            'line_number': line_num,
                            'line': line,
                            'error': str(e)
                        })
                        
        except Exception as e:
            raise RuntimeError(f"Failed to parse file {file_path}: {e}")
            
        # 计算统计信息
        results['statistics'] = self._compute_statistics(results)
        
        return results
        
    def parse_json_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """解析JSON Lines格式的日志文件"""
        return self.parse_file(file_path, **kwargs)
        
    def extract_metrics(self, 
                       file_path: str,
                       tags: Optional[List[str]] = None,
                       step_range: Optional[Tuple[int, int]] = None) -> Dict[str, List[MetricRecord]]:
        """提取指标数据"""
        
        results = self.parse_file(file_path)
        metrics_by_tag = {}
        
        for metric in results['metrics']:
            # 标签过滤
            if tags and metric.tag not in tags:
                continue
                
            # 步数范围过滤
            if step_range:
                start_step, end_step = step_range
                if not (start_step <= metric.step <= end_step):
                    continue
                    
            if metric.tag not in metrics_by_tag:
                metrics_by_tag[metric.tag] = []
            metrics_by_tag[metric.tag].append(metric)
            
        # 按步数排序
        for tag in metrics_by_tag:
            metrics_by_tag[tag].sort(key=lambda x: x.step)
            
        return metrics_by_tag
        
    def extract_errors(self, file_path: str) -> List[Dict[str, Any]]:
        """提取错误信息"""
        results = self.parse_file(file_path)
        
        errors = []
        for record in results['records']:
            if record.level in ('ERROR', 'CRITICAL'):
                error_info = {
                    'timestamp': record.timestamp,
                    'level': record.level,
                    'message': record.message,
                    'logger_name': record.logger_name,
                    'line_number': record.line_number
                }
                
                # 尝试解析错误类型
                error_match = self.patterns['error'].search(record.message)
                if error_match:
                    error_info.update(error_match.groupdict())
                    
                errors.append(error_info)
                
        return errors
        
    def get_log_summary(self, file_path: str) -> Dict[str, Any]:
        """获取日志摘要"""
        results = self.parse_file(file_path)
        
        summary = {
            'file_info': {
                'path': file_path,
                'size': results['file_size'],
                'total_lines': len(results['records']) + len(results['parsing_errors'])
            },
            'record_counts': {
                'total_records': len(results['records']),
                'metrics': len(results['metrics']),
                'errors': len([r for r in results['records'] if r.level in ('ERROR', 'CRITICAL')]),
                'warnings': len([r for r in results['records'] if r.level == 'WARNING']),
                'parsing_errors': len(results['parsing_errors'])
            },
            'time_range': self._get_time_range(results['records']),
            'metric_tags': list(set(m.tag for m in results['metrics'])),
            'logger_names': list(set(r.logger_name for r in results['records'])),
            'statistics': results['statistics']
        }
        
        return summary
        
    def _parse_json_line(self, line: str, line_num: int) -> Optional[Union[LogRecord, MetricRecord]]:
        """解析JSON格式的日志行"""
        try:
            data = json.loads(line)
            
            # 检查是否为指标记录
            if data.get('type') in ('scalar', 'histogram'):
                return MetricRecord(
                    timestamp=self._parse_timestamp(data.get('timestamp')),
                    step=data.get('step', 0),
                    tag=data.get('tag', ''),
                    value=data.get('value', 0.0),
                    metric_type=data.get('type', 'scalar'),
                    extra_data={k: v for k, v in data.items() 
                              if k not in ('timestamp', 'step', 'tag', 'value', 'type')}
                )
            else:
                # 普通日志记录
                return LogRecord(
                    timestamp=self._parse_timestamp(data.get('timestamp')),
                    level=data.get('level', 'INFO'),
                    logger_name=data.get('logger_name', 'root'),
                    message=data.get('message', ''),
                    raw_line=line,
                    line_number=line_num,
                    extra_fields={k: v for k, v in data.items() 
                                if k not in ('timestamp', 'level', 'logger_name', 'message')}
                )
                
        except json.JSONDecodeError:
            return None
            
    def _parse_text_line(self, line: str, line_num: int) -> Optional[Union[LogRecord, MetricRecord]]:
        """解析文本格式的日志行"""
        
        # 尝试匹配标准日志格式
        match = self.patterns['standard'].match(line)
        if match:
            groups = match.groupdict()
            
            # 检查消息中是否包含指标信息
            message = groups['message']
            
            # 尝试解析指标
            metric_match = self.patterns['metric'].search(message)
            if metric_match:
                metric_groups = metric_match.groupdict()
                return MetricRecord(
                    timestamp=self._parse_timestamp(groups['timestamp']),
                    step=int(metric_groups.get('step', 0)),
                    tag=metric_groups['tag'],
                    value=float(metric_groups['value']),
                    metric_type='scalar'
                )
                
            # 尝试解析直方图
            hist_match = self.patterns['histogram'].search(message)
            if hist_match:
                hist_groups = hist_match.groupdict()
                return MetricRecord(
                    timestamp=self._parse_timestamp(groups['timestamp']),
                    step=int(hist_groups.get('step', 0)),
                    tag=hist_groups['tag'],
                    value=float(hist_groups['mean']),
                    metric_type='histogram',
                    extra_data={
                        'mean': float(hist_groups['mean']),
                        'std': float(hist_groups['std']),
                        'min': float(hist_groups['min']),
                        'max': float(hist_groups['max']),
                        'count': int(hist_groups['count'])
                    }
                )
                
            # 普通日志记录
            return LogRecord(
                timestamp=self._parse_timestamp(groups['timestamp']),
                level=groups['level'],
                logger_name=groups['logger_name'],
                message=message,
                raw_line=line,
                line_number=line_num
            )
            
        return None
        
    def _parse_timestamp(self, timestamp_str: Union[str, float, int]) -> float:
        """解析时间戳"""
        if isinstance(timestamp_str, (int, float)):
            return float(timestamp_str)
            
        if isinstance(timestamp_str, str):
            # 尝试ISO格式
            try:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                return dt.timestamp()
            except ValueError:
                pass
                
            # 尝试各种格式
            for fmt in self.timestamp_formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return dt.timestamp()
                except ValueError:
                    continue
                    
        # 如果都失败了，返回当前时间
        return datetime.now().timestamp()
        
    def _should_include_record(self, 
                              record: Union[LogRecord, MetricRecord],
                              level_filter: Optional[str],
                              keyword_filter: Optional[str]) -> bool:
        """判断是否应该包含记录"""
        
        # 级别过滤
        if level_filter and isinstance(record, LogRecord):
            if record.level != level_filter.upper():
                return False
                
        # 关键词过滤
        if keyword_filter:
            if isinstance(record, LogRecord):
                if keyword_filter.lower() not in record.message.lower():
                    return False
            elif isinstance(record, MetricRecord):
                if keyword_filter.lower() not in record.tag.lower():
                    return False
                    
        return True
        
    def _compute_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计信息"""
        stats = {}
        
        # 记录级别统计
        level_counts = {}
        for record in results['records']:
            level = record.level
            level_counts[level] = level_counts.get(level, 0) + 1
        stats['level_counts'] = level_counts
        
        # 指标统计
        if results['metrics']:
            metric_stats = {}
            for metric in results['metrics']:
                tag = metric.tag
                if tag not in metric_stats:
                    metric_stats[tag] = {'count': 0, 'values': []}
                metric_stats[tag]['count'] += 1
                metric_stats[tag]['values'].append(metric.value)
                
            # 计算每个指标的统计量
            for tag, data in metric_stats.items():
                values = data['values']
                if values:
                    import statistics
                    data['mean'] = statistics.mean(values)
                    data['std'] = statistics.stdev(values) if len(values) > 1 else 0.0
                    data['min'] = min(values)
                    data['max'] = max(values)
                    del data['values']  # 删除原始数据节省内存
                    
            stats['metric_stats'] = metric_stats
            
        return stats
        
    def _get_time_range(self, records: List[LogRecord]) -> Optional[Dict[str, str]]:
        """获取时间范围"""
        if not records:
            return None
            
        timestamps = [r.timestamp for r in records]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        return {
            'start': datetime.fromtimestamp(start_time).isoformat(),
            'end': datetime.fromtimestamp(end_time).isoformat(),
            'duration_seconds': end_time - start_time
        }