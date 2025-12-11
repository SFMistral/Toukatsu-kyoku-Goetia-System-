"""
系统日志模型
"""
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, ForeignKey, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "debug"        # 调试
    INFO = "info"          # 信息
    WARNING = "warning"    # 警告
    ERROR = "error"        # 错误
    CRITICAL = "critical"  # 严重


class SystemLog(BaseModel):
    """系统日志模型"""
    __tablename__ = 'system_logs'
    
    # 日志信息
    log_level = Column(SQLEnum(LogLevel), nullable=False, comment='日志级别')
    level = Column(String(20), nullable=False, comment='日志级别字符串')
    category = Column(String(64), comment='日志类别')
    message = Column(Text, nullable=False, comment='日志消息')
    details = Column(Text, comment='详细信息JSON')
    
    # 来源信息
    logger_name = Column(String(255), comment='记录器名称')
    module = Column(String(255), comment='模块名')
    function = Column(String(255), comment='函数名')
    line_number = Column(Integer, comment='行号')
    
    # 请求信息
    source_ip = Column(String(45), comment='来源IP')
    request_path = Column(String(512), comment='请求路径')
    request_method = Column(String(10), comment='请求方法')
    response_code = Column(Integer, comment='响应状态码')
    duration_ms = Column(Integer, comment='处理耗时(毫秒)')
    
    # 关联信息
    user_id = Column(Integer, ForeignKey('users.id'), comment='操作用户ID')
    task_id = Column(Integer, ForeignKey('tasks.id'), comment='关联任务ID')
    experiment_id = Column(Integer, ForeignKey('experiments.id'), comment='关联实验ID')
    
    # 额外信息
    extra_data = Column(Text, comment='额外数据JSON')
    
    # 关系
    user = relationship("User")
    task = relationship("Task")
    experiment = relationship("Experiment")
    
    # 索引
    __table_args__ = (
        Index('idx_log_level', 'log_level'),
        Index('idx_category', 'category'),
        Index('idx_user_id', 'user_id'),
        Index('idx_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level='{self.log_level.value}', message='{self.message[:50]}...')>"