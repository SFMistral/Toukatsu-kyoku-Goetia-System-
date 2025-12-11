"""
CSV导出器
"""
import csv
import io
from typing import List, Dict, Any, Optional
from datetime import datetime
from database.repositories.experiment_repository import ExperimentRepository
from database.repositories.metric_repository import MetricRepository
from database.repositories.task_repository import TaskRepository

class CSVExporter:
    """CSV数据导出器"""
    
    def __init__(self):
        self.experiment_repo = ExperimentRepository()
        self.metric_repo = MetricRepository()
        self.task_repo = TaskRepository()
    
    def export_experiments_to_csv(self, experiment_ids: List[int] = None, 
                                 include_metrics: bool = True) -> str:
        """导出实验数据到CSV"""
        output = io.StringIO()
        
        # 获取实验数据
        if experiment_ids:
            experiments = [self.experiment_repo.get_by_id(exp_id) 
                         for exp_id in experiment_ids if self.experiment_repo.get_by_id(exp_id)]
        else:
            experiments = self.experiment_repo.get_all(limit=1000)
        
        if not experiments:
            return ""
        
        # 定义CSV字段
        fieldnames = [
            'experiment_id', 'name', 'description', 'status',
            'task_id', 'node_id', 'user_id', 'final_score', 'best_epoch',
            'created_at', 'updated_at'
        ]
        
        if include_metrics:
            # 获取所有可能的指标名称
            all_metrics = set()
            for exp in experiments:
                metrics = self.metric_repo.get_unique_metric_names(exp.id)
                all_metrics.update(metrics)
            
            # 为每个指标添加字段
            for metric in sorted(all_metrics):
                fieldnames.extend([
                    f'{metric}_best', f'{metric}_final', f'{metric}_avg'
                ])
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        # 写入数据
        for exp in experiments:
            row = {
                'experiment_id': exp.id,
                'name': exp.name,
                'description': exp.description,
                'status': exp.status.value if exp.status else '',
                'task_id': exp.task_id,
                'node_id': exp.node_id,
                'user_id': exp.user_id,
                'final_score': exp.final_score,
                'best_epoch': exp.best_epoch,
                'created_at': exp.created_at.isoformat() if exp.created_at else '',
                'updated_at': exp.updated_at.isoformat() if exp.updated_at else ''
            }
            
            if include_metrics:
                # 添加指标数据
                for metric in all_metrics:
                    metric_records = self.metric_repo.get_by_experiment_and_metric(exp.id, metric)
                    if metric_records:
                        values = [r.metric_value for r in metric_records]
                        row[f'{metric}_best'] = max(values)
                        row[f'{metric}_final'] = values[-1] if values else None
                        row[f'{metric}_avg'] = sum(values) / len(values)
                    else:
                        row[f'{metric}_best'] = None
                        row[f'{metric}_final'] = None
                        row[f'{metric}_avg'] = None
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def export_metrics_to_csv(self, experiment_id: int) -> str:
        """导出指标数据到CSV"""
        output = io.StringIO()
        
        # 获取指标数据
        metrics = self.metric_repo.get_by_experiment_id(experiment_id)
        
        if not metrics:
            return ""
        
        fieldnames = [
            'metric_id', 'experiment_id', 'metric_name', 'metric_value',
            'epoch', 'step', 'created_at'
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for metric in metrics:
            writer.writerow({
                'metric_id': metric.id,
                'experiment_id': metric.experiment_id,
                'metric_name': metric.metric_name,
                'metric_value': metric.metric_value,
                'epoch': metric.epoch,
                'step': metric.step,
                'created_at': metric.created_at.isoformat() if metric.created_at else ''
            })
        
        return output.getvalue()
    
    def export_tasks_to_csv(self, task_ids: List[int] = None) -> str:
        """导出任务数据到CSV"""
        output = io.StringIO()
        
        # 获取任务数据
        if task_ids:
            tasks = [self.task_repo.get_by_id(task_id) 
                    for task_id in task_ids if self.task_repo.get_by_id(task_id)]
        else:
            tasks = self.task_repo.get_all(limit=1000)
        
        if not tasks:
            return ""
        
        fieldnames = [
            'task_id', 'name', 'description', 'status', 'priority',
            'node_id', 'user_id', 'created_at', 'updated_at'
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for task in tasks:
            writer.writerow({
                'task_id': task.id,
                'name': task.name,
                'description': task.description,
                'status': task.status.value if task.status else '',
                'priority': task.priority,
                'node_id': task.node_id,
                'user_id': task.user_id,
                'created_at': task.created_at.isoformat() if task.created_at else '',
                'updated_at': task.updated_at.isoformat() if task.updated_at else ''
            })
        
        return output.getvalue()
    
    def export_comparison_data(self, experiment_ids: List[int], 
                             metrics: List[str] = None) -> str:
        """导出实验比较数据"""
        output = io.StringIO()
        
        experiments = [self.experiment_repo.get_by_id(exp_id) 
                      for exp_id in experiment_ids if self.experiment_repo.get_by_id(exp_id)]
        
        if not experiments:
            return ""
        
        # 如果没有指定指标，获取所有指标
        if not metrics:
            all_metrics = set()
            for exp in experiments:
                exp_metrics = self.metric_repo.get_unique_metric_names(exp.id)
                all_metrics.update(exp_metrics)
            metrics = sorted(all_metrics)
        
        fieldnames = ['experiment_id', 'experiment_name'] + metrics
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for exp in experiments:
            row = {
                'experiment_id': exp.id,
                'experiment_name': exp.name
            }
            
            # 添加每个指标的最佳值
            for metric in metrics:
                best_record = self.metric_repo.get_best_metric_value(exp.id, metric, True)
                row[metric] = best_record.metric_value if best_record else None
            
            writer.writerow(row)
        
        return output.getvalue()