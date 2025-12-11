"""
环境快照模型
"""
from sqlalchemy import Column, String, Text, Integer, ForeignKey, Boolean, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class EnvironmentSnapshot(BaseModel):
    """环境快照模型"""
    __tablename__ = 'environment_snapshots'
    
    # 关联信息
    task_id = Column(Integer, ForeignKey('tasks.id'), unique=True, comment='关联任务ID')
    experiment_id = Column(Integer, ForeignKey('experiments.id'), comment='关联实验ID')
    
    # 基本信息
    name = Column(String(255), nullable=False, comment='快照名称')
    
    # Python环境
    python_version = Column(String(32), comment='Python版本')
    pytorch_version = Column(String(32), comment='PyTorch版本')
    
    # CUDA环境
    cuda_version = Column(String(32), comment='CUDA版本')
    cudnn_version = Column(String(32), comment='cuDNN版本')
    
    # 系统信息
    os_info = Column(String(128), comment='操作系统信息')
    
    # 包信息
    pip_packages = Column(Text, comment='pip包列表及版本JSON')
    conda_packages = Column(Text, comment='conda包列表JSON')
    packages = Column(Text, comment='包列表JSON')
    
    # 环境变量
    env_variables = Column(Text, comment='环境变量JSON(脱敏)')
    
    # Git信息
    git_commit = Column(String(64), comment='Git提交哈希')
    git_branch = Column(String(128), comment='Git分支')
    git_dirty = Column(Boolean, default=False, comment='是否有未提交更改')
    
    # 系统信息
    system_info = Column(Text, comment='系统信息JSON')
    gpu_info = Column(Text, comment='GPU信息JSON')
    
    # 额外信息
    description = Column(Text, comment='描述')
    meta_data = Column(Text, comment='元数据JSON')
    
    # 关系
    task = relationship("Task")
    experiment = relationship("Experiment")
    
    # 索引
    __table_args__ = (
        Index('idx_task_id', 'task_id'),
    )
    
    def __repr__(self):
        return f"<EnvironmentSnapshot(id={self.id}, name='{self.name}', python='{self.python_version}')>"