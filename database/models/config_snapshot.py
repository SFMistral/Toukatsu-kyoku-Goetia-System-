"""
配置快照模型
"""
from sqlalchemy import Column, String, Text, Integer, ForeignKey, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class ConfigSnapshot(BaseModel):
    """配置快照模型（不可变）"""
    __tablename__ = 'config_snapshots'
    
    # 基本信息
    name = Column(String(255), nullable=False, comment='快照名称')
    config_type = Column(String(100), nullable=False, comment='配置类型')
    
    # 配置内容
    config_hash = Column(String(64), unique=True, comment='配置哈希(内容寻址)')
    config_content = Column(Text, nullable=False, comment='完整配置内容JSON')
    config_yaml = Column(Text, comment='YAML格式(便于查看)')
    
    # 来源信息
    source_file = Column(String(512), comment='来源文件路径')
    overrides = Column(Text, comment='命令行覆盖参数JSON')
    
    # 关联信息
    experiment_id = Column(Integer, ForeignKey('experiments.id'), comment='关联实验ID')
    
    # 额外信息
    description = Column(Text, comment='描述')
    meta_data = Column(Text, comment='元数据JSON')
    
    # 关系
    experiment = relationship("Experiment", foreign_keys=[experiment_id])
    
    # 索引
    __table_args__ = (
        Index('idx_config_hash', 'config_hash'),
    )
    
    def __repr__(self):
        return f"<ConfigSnapshot(id={self.id}, name='{self.name}', hash='{self.config_hash[:8] if self.config_hash else None}...')>"