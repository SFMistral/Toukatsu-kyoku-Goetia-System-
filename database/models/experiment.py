"""
实验模型
"""
import uuid
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, ForeignKey, Float, Enum as SQLEnum
from sqlalchemy.orm import relationship
from .base import BaseModel


class ExperimentStatus(Enum):
    """实验状态枚举"""
    ACTIVE = "active"        # 活跃
    ARCHIVED = "archived"    # 已归档
    DELETED = "deleted"      # 已删除
    CREATED = "created"      # 已创建
    RUNNING = "running"      # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    STOPPED = "stopped"      # 已停止


class Experiment(BaseModel):
    """实验模型"""
    __tablename__ = 'experiments'
    
    # 基本信息
    experiment_id = Column(String(64), unique=True, default=lambda: str(uuid.uuid4()), comment='实验唯一标识')
    name = Column(String(255), nullable=False, comment='实验名称')
    description = Column(Text, comment='实验描述')
    project = Column(String(255), comment='所属项目')
    tags = Column(Text, comment='标签列表JSON')
    status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.CREATED, nullable=False, comment='实验状态')
    
    # 实验配置
    config = Column(Text, comment='实验配置JSON')
    hyperparameters = Column(Text, comment='超参数JSON')
    
    # 最佳结果
    best_task_id = Column(Integer, comment='最佳任务ID')
    best_metric_value = Column(Float, comment='最佳指标值')
    best_metric_name = Column(String(64), comment='最佳指标名')
    final_score = Column(Float, comment='最终得分')
    best_epoch = Column(Integer, comment='最佳轮次')
    
    # 关联关系
    base_config_id = Column(Integer, ForeignKey('config_snapshots.id'), comment='基础配置快照ID')
    node_id = Column(Integer, ForeignKey('nodes.id'), comment='执行节点ID')
    user_id = Column(Integer, ForeignKey('users.id'), comment='创建用户ID')
    
    # 关系
    base_config = relationship("ConfigSnapshot", foreign_keys=[base_config_id])
    node = relationship("Node", back_populates="experiments")
    user = relationship("User", back_populates="experiments")
    tasks = relationship("Task", back_populates="experiment", foreign_keys="Task.experiment_id")
    comparison_records = relationship("ComparisonRecord", back_populates="experiment", foreign_keys="ComparisonRecord.experiment_id")
    reports = relationship("Report", back_populates="experiment", foreign_keys="Report.experiment_id")
    
    def __repr__(self):
        return f"<Experiment(id={self.id}, experiment_id='{self.experiment_id}', name='{self.name}', status='{self.status.value}')>"