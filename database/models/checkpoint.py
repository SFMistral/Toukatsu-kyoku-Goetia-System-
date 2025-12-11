"""
检查点模型
"""
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, ForeignKey, Float, Boolean, BigInteger, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class CheckpointType(Enum):
    """检查点类型枚举"""
    REGULAR = "regular"    # 常规检查点
    BEST = "best"          # 最佳检查点
    FINAL = "final"        # 最终检查点


class Checkpoint(BaseModel):
    """检查点模型"""
    __tablename__ = 'checkpoints'
    
    # 关联信息
    task_id = Column(Integer, ForeignKey('tasks.id'), nullable=False, comment='关联任务ID')
    experiment_id = Column(Integer, ForeignKey('experiments.id'), comment='关联实验ID')
    
    # 基本信息
    name = Column(String(255), nullable=False, comment='检查点名称')
    checkpoint_type = Column(SQLEnum(CheckpointType), default=CheckpointType.REGULAR, comment='检查点类型')
    
    # 训练进度
    epoch = Column(Integer, comment='保存时epoch')
    step = Column(Integer, comment='保存时step')
    
    # 文件信息
    file_path = Column(String(512), nullable=False, comment='本地文件路径')
    file_size = Column(BigInteger, comment='文件大小(字节)')
    file_hash = Column(String(64), comment='文件哈希(MD5/SHA256)')
    
    # 指标信息
    metric_name = Column(String(64), comment='关联指标名(best类型)')
    metric_value = Column(Float, comment='指标值')
    score = Column(Float, comment='评分')
    is_best = Column(Boolean, default=False, comment='是否为最佳检查点')
    
    # 云端同步
    is_uploaded = Column(Boolean, default=False, comment='是否已上传云端')
    cloud_path = Column(String(512), comment='云端路径')
    
    # 额外信息
    meta_data = Column(Text, comment='额外元数据JSON')
    description = Column(Text, comment='描述')
    
    # 关系
    task = relationship("Task", back_populates="checkpoints")
    experiment = relationship("Experiment")
    
    # 索引
    __table_args__ = (
        Index('idx_task_id', 'task_id'),
        Index('idx_checkpoint_type', 'checkpoint_type'),
        Index('idx_task_epoch', 'task_id', 'epoch'),
    )
    
    def __repr__(self):
        return f"<Checkpoint(id={self.id}, name='{self.name}', type='{self.checkpoint_type.value}', epoch={self.epoch})>"