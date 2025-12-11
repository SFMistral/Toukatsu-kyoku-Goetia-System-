"""
比较记录模型
"""
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, ForeignKey, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class ComparisonType(Enum):
    """比较类型枚举"""
    METRIC = "metric"    # 指标对比
    CONFIG = "config"    # 配置对比
    CURVE = "curve"      # 曲线对比


class ComparisonRecord(BaseModel):
    """比较记录模型"""
    __tablename__ = 'comparison_records'
    
    # 基本信息
    name = Column(String(255), nullable=False, comment='比较名称')
    comparison_type = Column(SQLEnum(ComparisonType), default=ComparisonType.METRIC, nullable=False, comment='比较类型')
    
    # 比较内容
    task_ids = Column(Text, nullable=False, comment='参与比较的任务ID列表JSON')
    experiment_ids = Column(Text, comment='参与比较的实验ID列表JSON')
    comparison_config = Column(Text, comment='比较配置JSON(指标、维度等)')
    result_summary = Column(Text, comment='比较结果摘要JSON')
    
    # 关联信息
    experiment_id = Column(Integer, ForeignKey('experiments.id'), comment='所属实验ID')
    user_id = Column(Integer, ForeignKey('users.id'), comment='创建用户ID')
    
    # 额外信息
    notes = Column(Text, comment='备注')
    description = Column(Text, comment='描述')
    meta_data = Column(Text, comment='元数据JSON')
    
    # 关系
    experiment = relationship("Experiment", back_populates="comparison_records")
    user = relationship("User")
    
    # 索引
    __table_args__ = (
        Index('idx_experiment_id', 'experiment_id'),
        Index('idx_user_id', 'user_id'),
    )
    
    def __repr__(self):
        return f"<ComparisonRecord(id={self.id}, name='{self.name}', type='{self.comparison_type.value}')>"