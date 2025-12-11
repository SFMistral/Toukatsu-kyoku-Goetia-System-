"""
指标记录模型
"""
from enum import Enum
from sqlalchemy import Column, String, Float, Integer, ForeignKey, Text, DateTime, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from .base import BaseModel, utc_now


class MetricPhase(Enum):
    """指标阶段枚举"""
    TRAIN = "train"    # 训练
    VAL = "val"        # 验证
    TEST = "test"      # 测试


class MetricRecord(BaseModel):
    """指标记录模型"""
    __tablename__ = 'metric_records'
    
    # 关联信息
    task_id = Column(Integer, ForeignKey('tasks.id'), nullable=False, comment='关联任务ID')
    experiment_id = Column(Integer, ForeignKey('experiments.id'), comment='关联实验ID')
    
    # 指标信息
    metric_name = Column(String(64), nullable=False, comment='指标名称')
    metric_value = Column(Float, nullable=False, comment='指标值')
    
    # 训练进度
    step = Column(Integer, comment='训练步数')
    epoch = Column(Integer, comment='训练轮次')
    phase = Column(SQLEnum(MetricPhase), default=MetricPhase.TRAIN, comment='阶段')
    
    # 时间信息
    timestamp = Column(DateTime, default=utc_now, comment='记录时间')
    
    # 额外信息
    meta_data = Column(Text, comment='元数据JSON')
    
    # 关系
    task = relationship("Task", back_populates="metric_records")
    experiment = relationship("Experiment")
    
    # 索引
    __table_args__ = (
        Index('idx_task_metric_step', 'task_id', 'metric_name', 'step'),
        Index('idx_task_phase', 'task_id', 'phase'),
        Index('idx_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<MetricRecord(id={self.id}, metric='{self.metric_name}', value={self.metric_value}, epoch={self.epoch})>"