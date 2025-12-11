# -*- coding: utf-8 -*-
"""
分布式日志聚合器

聚合多节点/多进程的日志，进行时间线对齐、指标统计等操作。
"""

import os
import glob
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass
import statistics

from .log_parser import LogParser, LogRecord, MetricRecord


@dataclass
class AggregatedMetric:
    """聚合后的指标"""
    tag: str
    step: int
    timestamp: float
    values: List[float]
    mean: float
    std: float
    min_val: float
    max_val: float
    count: int
    sources: List[str]  # 来源文件/节点


class LogAggregator:
    """分布式日志聚合器"""
    
    def __init__(self):
        self.parser = LogParser()
        
    def aggregate_directory(self, 
                           log_dir: str,
                           pattern: str = "*.log",
                           aggregation_method: str = "mean") -> Dict[str, Any]:
        """聚合目录中的所有日志文件"""
        
        log_files = glob.glob(os.path.join(log_dir, pattern))
        if not log_files:
            raise ValueError(f"No log files found in {log_dir} with pattern {pattern}")
            
        return self.aggregate_files(log_files, aggregation_method)
        
    def aggregate_files(self, 
                       file_paths: List[str],
                       aggregation_method: str = "mean") -> Dict[str, Any]:
        """聚合多个日志文件"""
        
        all_records = []
        all_metrics = []
        file_info = {}
        
        # 解析所有文件
        for file_path in file_paths:
            try:
                results = self.parser.parse_file(file_path)
                all_records.extend(results['records'])
                all_metrics.extend(results['metrics'])
                
                file_info[file_path] = {
                    'record_count': len(results['records']),
                    'metric_count': len(results['metrics']),
                    'file_size': results['file_size']
                }
                
            except Exception as e:
                print(f"Failed to parse {file_path}: {e}")
                file_info[file_path] = {'error': str(e)}
                
        # 聚合指标
        aggregated_metrics = self._aggregate_metrics(all_metrics, aggregation_method)
        
        # 合并日志记录
        merged_records = self._merge_records(all_records)
        
        return {
            'aggregated_metrics': aggregated_metrics,
            'merged_records': merged_records,
            'file_info': file_info,
            'summary': self._compute_aggregation_summary(aggregated_metrics, merged_records)
        }
        
    def aggregate_distributed_logs(self, 
                                  log_dirs: List[str],
                                  rank_pattern: str = "rank_{rank}",
                                  aggregation_method: str = "mean") -> Dict[str, Any]:
        """聚合分布式训练的日志"""
        
        rank_files = {}
        
        # 收集各个rank的日志文件
        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                continue
                
            # 查找rank文件
            for file_name in os.listdir(log_dir):
                if 'rank' in file_name.lower():
                    # 尝试提取rank号
                    import re
                    rank_match = re.search(r'rank[_\-]?(\d+)', file_name.lower())
                    if rank_match:
                        rank = int(rank_match.group(1))
                        if rank not in rank_files:
                            rank_files[rank] = []
                        rank_files[rank].append(os.path.join(log_dir, file_name))
                        
        if not rank_files:
            # 如果没有找到rank文件，尝试聚合所有文件
            all_files = []
            for log_dir in log_dirs:
                all_files.extend(glob.glob(os.path.join(log_dir, "*.log")))
            return self.aggregate_files(all_files, aggregation_method)
            
        # 按rank聚合
        rank_results = {}
        for rank, files in rank_files.items():
            rank_results[rank] = self.aggregate_files(files, aggregation_method)
            
        # 跨rank聚合
        return self._aggregate_across_ranks(rank_results, aggregation_method)
        
    def align_by_step(self, 
                     metrics_by_source: Dict[str, List[MetricRecord]],
                     alignment_method: str = "interpolate") -> Dict[str, List[AggregatedMetric]]:
        """按步数对齐指标"""
        
        # 收集所有步数
        all_steps = set()
        for metrics in metrics_by_source.values():
            all_steps.update(m.step for m in metrics)
        all_steps = sorted(all_steps)
        
        # 按标签分组
        metrics_by_tag = defaultdict(lambda: defaultdict(list))
        for source, metrics in metrics_by_source.items():
            for metric in metrics:
                metrics_by_tag[metric.tag][source].append(metric)
                
        # 对齐每个标签的指标
        aligned_metrics = {}
        for tag, source_metrics in metrics_by_tag.items():
            aligned_metrics[tag] = self._align_tag_metrics(
                source_metrics, all_steps, alignment_method
            )
            
        return aligned_metrics
        
    def align_by_time(self, 
                     metrics_by_source: Dict[str, List[MetricRecord]],
                     time_window: float = 1.0) -> Dict[str, List[AggregatedMetric]]:
        """按时间对齐指标"""
        
        # 找到时间范围
        all_timestamps = []
        for metrics in metrics_by_source.values():
            all_timestamps.extend(m.timestamp for m in metrics)
            
        if not all_timestamps:
            return {}
            
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        
        # 创建时间窗口
        time_windows = []
        current_time = start_time
        while current_time < end_time:
            time_windows.append((current_time, current_time + time_window))
            current_time += time_window
            
        # 按标签和时间窗口聚合
        metrics_by_tag = defaultdict(lambda: defaultdict(list))
        for source, metrics in metrics_by_source.items():
            for metric in metrics:
                # 找到对应的时间窗口
                for i, (window_start, window_end) in enumerate(time_windows):
                    if window_start <= metric.timestamp < window_end:
                        metrics_by_tag[metric.tag][i].append((metric, source))
                        break
                        
        # 聚合每个时间窗口的指标
        aligned_metrics = {}
        for tag, window_metrics in metrics_by_tag.items():
            aligned_metrics[tag] = []
            for window_idx in sorted(window_metrics.keys()):
                window_data = window_metrics[window_idx]
                if window_data:
                    aggregated = self._aggregate_window_metrics(
                        window_data, time_windows[window_idx]
                    )
                    aligned_metrics[tag].append(aggregated)
                    
        return aligned_metrics
        
    def compare_runs(self, 
                    run_dirs: List[str],
                    metric_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """比较多次运行的结果"""
        
        run_results = {}
        
        # 聚合每次运行
        for i, run_dir in enumerate(run_dirs):
            run_name = f"run_{i}"
            try:
                run_results[run_name] = self.aggregate_directory(run_dir)
            except Exception as e:
                print(f"Failed to aggregate run {run_dir}: {e}")
                continue
                
        if not run_results:
            return {}
            
        # 提取指标进行比较
        comparison = self._compare_run_metrics(run_results, metric_tags)
        
        return {
            'run_results': run_results,
            'comparison': comparison
        }
        
    def _aggregate_metrics(self, 
                          metrics: List[MetricRecord],
                          method: str) -> Dict[str, List[AggregatedMetric]]:
        """聚合指标数据"""
        
        # 按标签和步数分组
        grouped_metrics = defaultdict(lambda: defaultdict(list))
        for metric in metrics:
            grouped_metrics[metric.tag][metric.step].append(metric)
            
        aggregated = {}
        for tag, step_metrics in grouped_metrics.items():
            aggregated[tag] = []
            
            for step in sorted(step_metrics.keys()):
                step_data = step_metrics[step]
                values = [m.value for m in step_data]
                
                if not values:
                    continue
                    
                # 计算聚合值
                if method == "mean":
                    agg_value = statistics.mean(values)
                elif method == "sum":
                    agg_value = sum(values)
                elif method == "max":
                    agg_value = max(values)
                elif method == "min":
                    agg_value = min(values)
                elif method == "first":
                    agg_value = values[0]
                elif method == "last":
                    agg_value = values[-1]
                else:
                    agg_value = statistics.mean(values)
                    
                # 创建聚合指标
                aggregated_metric = AggregatedMetric(
                    tag=tag,
                    step=step,
                    timestamp=statistics.mean([m.timestamp for m in step_data]),
                    values=values,
                    mean=statistics.mean(values),
                    std=statistics.stdev(values) if len(values) > 1 else 0.0,
                    min_val=min(values),
                    max_val=max(values),
                    count=len(values),
                    sources=[f"metric_{i}" for i in range(len(values))]
                )
                
                aggregated[tag].append(aggregated_metric)
                
        return aggregated
        
    def _merge_records(self, records: List[LogRecord]) -> List[LogRecord]:
        """合并日志记录"""
        # 按时间戳排序
        return sorted(records, key=lambda r: r.timestamp)
        
    def _compute_aggregation_summary(self, 
                                   aggregated_metrics: Dict[str, List[AggregatedMetric]],
                                   merged_records: List[LogRecord]) -> Dict[str, Any]:
        """计算聚合摘要"""
        
        summary = {
            'total_records': len(merged_records),
            'metric_tags': list(aggregated_metrics.keys()),
            'metric_counts': {tag: len(metrics) for tag, metrics in aggregated_metrics.items()},
        }
        
        if merged_records:
            summary['time_range'] = {
                'start': min(r.timestamp for r in merged_records),
                'end': max(r.timestamp for r in merged_records)
            }
            
        # 计算每个指标的最终统计
        final_stats = {}
        for tag, metrics in aggregated_metrics.items():
            if metrics:
                final_values = [m.mean for m in metrics]
                final_stats[tag] = {
                    'final_value': final_values[-1],
                    'best_value': max(final_values) if 'acc' in tag.lower() else min(final_values),
                    'mean_value': statistics.mean(final_values),
                    'std_value': statistics.stdev(final_values) if len(final_values) > 1 else 0.0
                }
                
        summary['final_stats'] = final_stats
        
        return summary
        
    def _aggregate_across_ranks(self, 
                               rank_results: Dict[int, Dict[str, Any]],
                               method: str) -> Dict[str, Any]:
        """跨rank聚合结果"""
        
        # 合并所有rank的指标
        all_metrics = []
        all_records = []
        
        for rank, results in rank_results.items():
            # 添加rank信息到指标
            for tag, metrics in results['aggregated_metrics'].items():
                for metric in metrics:
                    metric.sources = [f"rank_{rank}"]
                    all_metrics.append(metric)
                    
            all_records.extend(results['merged_records'])
            
        # 重新聚合
        metrics_by_tag = defaultdict(lambda: defaultdict(list))
        for metric in all_metrics:
            metrics_by_tag[metric.tag][metric.step].append(metric)
            
        final_aggregated = {}
        for tag, step_metrics in metrics_by_tag.items():
            final_aggregated[tag] = []
            
            for step in sorted(step_metrics.keys()):
                step_data = step_metrics[step]
                
                # 收集所有值
                all_values = []
                all_sources = []
                for metric in step_data:
                    all_values.extend(metric.values)
                    all_sources.extend(metric.sources)
                    
                if all_values:
                    aggregated_metric = AggregatedMetric(
                        tag=tag,
                        step=step,
                        timestamp=statistics.mean([m.timestamp for m in step_data]),
                        values=all_values,
                        mean=statistics.mean(all_values),
                        std=statistics.stdev(all_values) if len(all_values) > 1 else 0.0,
                        min_val=min(all_values),
                        max_val=max(all_values),
                        count=len(all_values),
                        sources=list(set(all_sources))
                    )
                    
                    final_aggregated[tag].append(aggregated_metric)
                    
        return {
            'aggregated_metrics': final_aggregated,
            'merged_records': self._merge_records(all_records),
            'rank_info': {rank: len(results['merged_records']) for rank, results in rank_results.items()},
            'summary': self._compute_aggregation_summary(final_aggregated, all_records)
        }
        
    def _align_tag_metrics(self, 
                          source_metrics: Dict[str, List[MetricRecord]],
                          all_steps: List[int],
                          method: str) -> List[AggregatedMetric]:
        """对齐单个标签的指标"""
        
        aligned = []
        
        for step in all_steps:
            step_values = []
            step_sources = []
            step_timestamps = []
            
            # 收集该步数的所有值
            for source, metrics in source_metrics.items():
                # 查找最接近的步数
                closest_metric = None
                min_diff = float('inf')
                
                for metric in metrics:
                    diff = abs(metric.step - step)
                    if diff < min_diff:
                        min_diff = diff
                        closest_metric = metric
                        
                if closest_metric and min_diff <= 5:  # 允许5步的误差
                    step_values.append(closest_metric.value)
                    step_sources.append(source)
                    step_timestamps.append(closest_metric.timestamp)
                    
            if step_values:
                aggregated = AggregatedMetric(
                    tag=list(source_metrics.keys())[0] if source_metrics else "",
                    step=step,
                    timestamp=statistics.mean(step_timestamps),
                    values=step_values,
                    mean=statistics.mean(step_values),
                    std=statistics.stdev(step_values) if len(step_values) > 1 else 0.0,
                    min_val=min(step_values),
                    max_val=max(step_values),
                    count=len(step_values),
                    sources=step_sources
                )
                aligned.append(aggregated)
                
        return aligned
        
    def _aggregate_window_metrics(self, 
                                 window_data: List[Tuple[MetricRecord, str]],
                                 time_window: Tuple[float, float]) -> AggregatedMetric:
        """聚合时间窗口内的指标"""
        
        metrics, sources = zip(*window_data)
        values = [m.value for m in metrics]
        
        return AggregatedMetric(
            tag=metrics[0].tag,
            step=int(statistics.mean([m.step for m in metrics])),
            timestamp=statistics.mean([m.timestamp for m in metrics]),
            values=values,
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_val=min(values),
            max_val=max(values),
            count=len(values),
            sources=list(sources)
        )
        
    def _compare_run_metrics(self, 
                           run_results: Dict[str, Dict[str, Any]],
                           metric_tags: Optional[List[str]]) -> Dict[str, Any]:
        """比较运行指标"""
        
        comparison = {}
        
        # 收集所有指标标签
        all_tags = set()
        for results in run_results.values():
            all_tags.update(results['aggregated_metrics'].keys())
            
        if metric_tags:
            all_tags = all_tags.intersection(set(metric_tags))
            
        # 比较每个指标
        for tag in all_tags:
            tag_comparison = {}
            
            for run_name, results in run_results.items():
                if tag in results['aggregated_metrics']:
                    metrics = results['aggregated_metrics'][tag]
                    if metrics:
                        final_values = [m.mean for m in metrics]
                        tag_comparison[run_name] = {
                            'final_value': final_values[-1],
                            'best_value': max(final_values) if 'acc' in tag.lower() else min(final_values),
                            'mean_value': statistics.mean(final_values),
                            'convergence_step': len(final_values)
                        }
                        
            comparison[tag] = tag_comparison
            
        return comparison