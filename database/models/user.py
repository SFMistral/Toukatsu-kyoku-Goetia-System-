"""
用户模型
"""
from enum import Enum
from sqlalchemy import Column, String, Text, Boolean, DateTime, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class UserRole(Enum):
    """用户角色枚举"""
    ADMIN = "admin"      # 管理员
    USER = "user"        # 普通用户
    VIEWER = "viewer"    # 只读用户


class User(BaseModel):
    """用户模型"""
    __tablename__ = 'users'
    
    # 基本信息
    username = Column(String(64), nullable=False, unique=True, comment='用户名')
    email = Column(String(255), unique=True, comment='邮箱')
    password_hash = Column(String(255), comment='密码哈希')
    
    # 用户信息
    full_name = Column(String(255), comment='全名')
    avatar_url = Column(String(500), comment='头像URL')
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False, comment='用户角色')
    
    # 用户状态
    is_active = Column(Boolean, default=True, comment='是否激活')
    is_admin = Column(Boolean, default=False, comment='是否管理员')
    last_login = Column(DateTime, comment='最后登录时间')
    
    # API密钥
    api_key = Column(String(64), unique=True, comment='API密钥')
    api_key_expires = Column(DateTime, comment='API密钥过期时间')
    
    # 额外信息
    preferences = Column(Text, comment='用户偏好设置JSON')
    meta_data = Column(Text, comment='元数据JSON')
    
    # 关系
    tasks = relationship("Task", back_populates="user")
    experiments = relationship("Experiment", back_populates="user")
    
    # 索引
    __table_args__ = (
        Index('idx_username', 'username'),
        Index('idx_email', 'email'),
        Index('idx_api_key', 'api_key'),
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role.value}')>"