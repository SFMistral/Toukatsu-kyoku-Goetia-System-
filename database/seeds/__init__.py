"""
数据库种子数据模块
"""
from .initial_data import create_initial_data
from .demo_data import create_demo_data

__all__ = ['create_initial_data', 'create_demo_data']