"""
指标仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc
from database.models.metric_record import MetricRecord
from database.connection import get_db_session

class MetricRepository:
    """指标数据访问层"""
    
    def create(self, metric_data: Dict[str, Any]) -> MetricRecord:
        """创建指标记录"""
        with get_db_session() as session:
            metric = MetricRecord(**metric_data)
            session.add(metric)
            session.flush()
            session.refresh(metric)
            session.expunge(metric)
            return metric
    
    def batch_create(self, metrics_data: List[Dict[str, Any]]) -> List[MetricRecord]:
        """批量创建指标记录"""
        with get_db_session() as session:
            metrics = [MetricRecord(**data) for data in metrics_data]
            session.add_all(metrics)
            session.flush()
            for metric in metrics:
                session.refresh(metric)
                session.expunge(metric)
            return metrics
    
    def get_by_id(self, metric_id: int) -> Optional[MetricRecord]:
        """根据ID获取指标记录"""
        with get_db_session() as session:
            return session.query(MetricRecord).filter(MetricRecord.id == metric_id).first()
    
    def get_by_experiment_id(self, experiment_id: int) -> List[MetricRecord]:
        """根据实验ID获取所有指标记录"""
        with get_db_session() as session:
            return session.query(MetricRecord).filter(
                MetricRecord.experiment_id == experiment_id
            ).order_by(asc(MetricRecord.epoch), asc(MetricRecord.step)).all()
    
    def get_by_experiment_and_metric(self, experiment_id: int, metric_name: str) -> List[MetricRecord]:
        """根据实验ID和指标名称获取记录"""
        with get_db_session() as session:
            return session.query(MetricRecord).filter(
                and_(
                    MetricRecord.experiment_id == experiment_id,
                    MetricRecord.metric_name == metric_name
                )
            ).order_by(asc(MetricRecord.epoch), asc(MetricRecord.step)).all()
    
    def get_latest_by_experiment(self, experiment_id: int, limit: int = 10) -> List[MetricRecord]:
        """获取实验的最新指标记录"""
        with get_db_session() as session:
            return session.query(MetricRecord).filter(
                MetricRecord.experiment_id == experiment_id
            ).order_by(desc(MetricRecord.created_at)).limit(limit).all()
    
    def get_best_metric_value(self, experiment_id: int, metric_name: str, 
                            is_higher_better: bool = True) -> Optional[MetricRecord]:
        """获取最佳指标值"""
        with get_db_session() as session:
            query = session.query(MetricRecord).filter(
                and_(
                    MetricRecord.experiment_id == experiment_id,
                    MetricRecord.metric_name == metric_name
                )
            )
            
            if is_higher_better:
                metric = query.order_by(desc(MetricRecord.metric_value)).first()
            else:
                metric = query.order_by(asc(MetricRecord.metric_value)).first()
            
            if metric:
                session.expunge(metric)
            return metric
    
    def get_metric_history(self, experiment_id: int, metric_name: str, 
                          epoch_start: int = None, epoch_end: int = None) -> List[MetricRecord]:
        """获取指标历史记录"""
        with get_db_session() as session:
            query = session.query(MetricRecord).filter(
                and_(
                    MetricRecord.experiment_id == experiment_id,
                    MetricRecord.metric_name == metric_name
                )
            )
            
            if epoch_start is not None:
                query = query.filter(MetricRecord.epoch >= epoch_start)
            if epoch_end is not None:
                query = query.filter(MetricRecord.epoch <= epoch_end)
            
            return query.order_by(asc(MetricRecord.epoch), asc(MetricRecord.step)).all()
    
    def get_metrics_by_epoch(self, experiment_id: int, epoch: int) -> List[MetricRecord]:
        """获取指定轮次的所有指标"""
        with get_db_session() as session:
            return session.query(MetricRecord).filter(
                and_(
                    MetricRecord.experiment_id == experiment_id,
                    MetricRecord.epoch == epoch
                )
            ).order_by(asc(MetricRecord.step)).all()
    
    def get_unique_metric_names(self, experiment_id: int) -> List[str]:
        """获取实验的所有指标名称"""
        with get_db_session() as session:
            result = session.query(MetricRecord.metric_name).filter(
                MetricRecord.experiment_id == experiment_id
            ).distinct().all()
            return [row[0] for row in result]
    
    def delete_by_experiment_id(self, experiment_id: int) -> int:
        """删除实验的所有指标记录"""
        with get_db_session() as session:
            count = session.query(MetricRecord).filter(
                MetricRecord.experiment_id == experiment_id
            ).count()
            session.query(MetricRecord).filter(
                MetricRecord.experiment_id == experiment_id
            ).delete()
            return count
    
    def get_metric_statistics(self, experiment_id: int, metric_name: str) -> Dict[str, float]:
        """获取指标统计信息"""
        with get_db_session() as session:
            from sqlalchemy import func
            
            result = session.query(
                func.min(MetricRecord.metric_value).label('min_value'),
                func.max(MetricRecord.metric_value).label('max_value'),
                func.avg(MetricRecord.metric_value).label('avg_value'),
                func.count(MetricRecord.metric_value).label('count')
            ).filter(
                and_(
                    MetricRecord.experiment_id == experiment_id,
                    MetricRecord.metric_name == metric_name
                )
            ).first()
            
            if result and result.count > 0:
                return {
                    'min': float(result.min_value),
                    'max': float(result.max_value),
                    'avg': float(result.avg_value),
                    'count': int(result.count)
                }
            return {'min': 0, 'max': 0, 'avg': 0, 'count': 0}