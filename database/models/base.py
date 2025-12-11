"""
基础模型类
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, DateTime, Boolean
from sqlalchemy.ext.declarative import declared_attr
from database.connection import Base


def utc_now():
    """获取当前UTC时间"""
    return datetime.now(timezone.utc)


class TimestampMixin:
    """时间戳Mixin"""
    created_at = Column(DateTime, default=utc_now, nullable=False, comment='创建时间')
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False, comment='更新时间')


class SoftDeleteMixin:
    """软删除Mixin"""
    is_deleted = Column(Boolean, default=False, nullable=False, comment='软删除标记')
    deleted_at = Column(DateTime, nullable=True, comment='删除时间')
    
    def soft_delete(self):
        """软删除"""
        self.is_deleted = True
        self.deleted_at = utc_now()
    
    def restore(self):
        """恢复软删除"""
        self.is_deleted = False
        self.deleted_at = None


class SerializeMixin:
    """序列化Mixin"""
    
    def to_dict(self, exclude=None):
        """转换为字典"""
        exclude = exclude or []
        result = {}
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                # 处理datetime类型
                if isinstance(value, datetime):
                    value = value.isoformat()
                result[column.name] = value
        return result
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建实例"""
        return cls(**{
            k: v for k, v in data.items() 
            if hasattr(cls, k)
        })


class BaseModel(Base, TimestampMixin, SoftDeleteMixin, SerializeMixin):
    """所有模型的基类"""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')
    
    @declared_attr
    def __tablename__(cls):
        """自动生成表名"""
        return cls.__name__.lower()
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id})>"