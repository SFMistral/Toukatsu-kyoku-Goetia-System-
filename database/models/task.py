"""
任务模型
"""
import uuid
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, DateTime, Enum as SQLEnum
from sqlalchemy.orm import relationship
from .base import BaseModel


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"      # 等待中
    RUNNING = "running"      # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 已取消
    PAUSED = "paused"        # 已暂停


class TaskType(Enum):
    """任务类型枚举"""
    TRAIN = "train"      # 训练
    EVAL = "eval"        # 评估
    EXPORT = "export"    # 导出


class Task(BaseModel):
    """任务模型"""
    __tablename__ = 'tasks'
    
    # 基本信息
    task_id = Column(String(64), unique=True, default=lambda: str(uuid.uuid4()), comment='任务唯一标识UUID')
    name = Column(String(255), nullable=False, comment='任务名称')
    description = Column(Text, comment='任务描述')
    task_type = Column(SQLEnum(TaskType), default=TaskType.TRAIN, nullable=False, comment='任务类型')
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False, comment='任务状态')
    priority = Column(Integer, default=3, comment='任务优先级(1-5)')
    
    # 任务配置
    config = Column(Text, comment='任务配置JSON')
    command = Column(Text, comment='执行命令')
    
    # 进度信息
    progress = Column(Float, default=0.0, comment='进度百分比')
    current_epoch = Column(Integer, default=0, comment='当前epoch')
    total_epochs = Column(Integer, comment='总epoch数')
    
    # 时间信息
    submitted_at = Column(DateTime, comment='提交时间')
    started_at = Column(DateTime, comment='开始执行时间')
    completed_at = Column(DateTime, comment='完成时间')
    
    # 错误处理
    error_message = Column(Text, comment='错误信息')
    retry_count = Column(Integer, default=0, comment='重试次数')
    
    # 关联关系
    node_id = Column(Integer, ForeignKey('nodes.id'), comment='执行节点ID')
    user_id = Column(Integer, ForeignKey('users.id'), comment='创建用户ID')
    experiment_id = Column(Integer, ForeignKey('experiments.id'), comment='关联实验ID')
    config_snapshot_id = Column(Integer, ForeignKey('config_snapshots.id'), comment='配置快照ID')
    parent_task_id = Column(Integer, ForeignKey('tasks.id'), comment='父任务ID(恢复训练时)')
    
    # 关系
    node = relationship("Node", back_populates="tasks")
    user = relationship("User", back_populates="tasks")
    experiment = relationship("Experiment", back_populates="tasks", foreign_keys=[experiment_id])
    config_snapshot = relationship("ConfigSnapshot")
    parent_task = relationship("Task", remote_side="Task.id")
    
    # 子关系
    metric_records = relationship("MetricRecord", back_populates="task")
    checkpoints = relationship("Checkpoint", back_populates="task")
    artifacts = relationship("Artifact", back_populates="task")
    
    def __repr__(self):
        return f"<Task(id={self.id}, task_id='{self.task_id}', name='{self.name}', status='{self.status.value}')>"