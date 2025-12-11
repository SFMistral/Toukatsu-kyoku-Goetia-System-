"""
报告模型
"""
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, ForeignKey, DateTime, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from .base import BaseModel, utc_now


class ReportType(Enum):
    """报告类型枚举"""
    TASK_SUMMARY = "task_summary"            # 任务摘要
    EXPERIMENT_SUMMARY = "experiment_summary"  # 实验摘要
    COMPARISON = "comparison"                # 对比报告
    ANALYSIS = "analysis"                    # 分析报告


class ReportFormat(Enum):
    """报告格式枚举"""
    HTML = "html"        # HTML格式
    PDF = "pdf"          # PDF格式
    MARKDOWN = "markdown"  # Markdown格式
    JSON = "json"        # JSON格式


class Report(BaseModel):
    """报告模型"""
    __tablename__ = 'reports'
    
    # 基本信息
    title = Column(String(255), nullable=False, comment='报告标题')
    report_type = Column(SQLEnum(ReportType), nullable=False, comment='报告类型')
    format = Column(SQLEnum(ReportFormat), default=ReportFormat.HTML, comment='报告格式')
    
    # 报告内容
    content = Column(Text, comment='报告内容JSON(结构化)')
    summary = Column(Text, comment='报告摘要')
    
    # 报告配置
    template_config = Column(Text, comment='模板配置JSON')
    generation_config = Column(Text, comment='生成配置JSON')
    
    # 文件信息
    file_path = Column(String(512), comment='生成文件路径')
    
    # 时间信息
    generated_at = Column(DateTime, default=utc_now, comment='生成时间')
    
    # 关联信息
    task_id = Column(Integer, ForeignKey('tasks.id'), comment='关联任务ID')
    experiment_id = Column(Integer, ForeignKey('experiments.id'), comment='关联实验ID')
    user_id = Column(Integer, ForeignKey('users.id'), comment='生成者ID')
    
    # 额外信息
    description = Column(Text, comment='描述')
    meta_data = Column(Text, comment='元数据JSON')
    
    # 关系
    task = relationship("Task")
    experiment = relationship("Experiment", back_populates="reports")
    user = relationship("User")
    
    # 索引
    __table_args__ = (
        Index('idx_task_id', 'task_id'),
        Index('idx_experiment_id', 'experiment_id'),
        Index('idx_report_type', 'report_type'),
    )
    
    def __repr__(self):
        return f"<Report(id={self.id}, title='{self.title}', type='{self.report_type.value}')>"