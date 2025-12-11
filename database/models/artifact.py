"""
工件模型
"""
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, ForeignKey, BigInteger, Boolean, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class ArtifactType(Enum):
    """工件类型枚举"""
    LOG = "log"                    # 日志
    VISUALIZATION = "visualization"  # 可视化
    CONFIG = "config"              # 配置
    MODEL = "model"                # 模型
    DATA = "data"                  # 数据
    OTHER = "other"                # 其他


class Artifact(BaseModel):
    """工件模型"""
    __tablename__ = 'artifacts'
    
    # 关联信息
    task_id = Column(Integer, ForeignKey('tasks.id'), nullable=False, comment='关联任务ID')
    experiment_id = Column(Integer, ForeignKey('experiments.id'), comment='关联实验ID')
    
    # 基本信息
    name = Column(String(255), nullable=False, comment='工件名称')
    artifact_type = Column(SQLEnum(ArtifactType), default=ArtifactType.OTHER, nullable=False, comment='工件类型')
    
    # 文件信息
    file_path = Column(String(512), nullable=False, comment='本地文件路径')
    file_size = Column(BigInteger, comment='文件大小(字节)')
    mime_type = Column(String(64), comment='MIME类型')
    
    # 描述信息
    description = Column(Text, comment='描述')
    meta_data = Column(Text, comment='元数据JSON')
    tags = Column(Text, comment='标签JSON')
    
    # 云端同步
    is_uploaded = Column(Boolean, default=False, comment='是否已上传云端')
    cloud_path = Column(String(512), comment='云端路径')
    
    # 关系
    task = relationship("Task", back_populates="artifacts")
    experiment = relationship("Experiment")
    
    # 索引
    __table_args__ = (
        Index('idx_task_id', 'task_id'),
        Index('idx_artifact_type', 'artifact_type'),
    )
    
    def __repr__(self):
        return f"<Artifact(id={self.id}, name='{self.name}', type='{self.artifact_type.value}')>"