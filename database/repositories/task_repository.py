"""
任务仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from database.models.task import Task, TaskStatus
from database.connection import get_db_session

class TaskRepository:
    """任务数据访问层"""
    
    def create(self, task_data: Dict[str, Any]) -> Task:
        """创建任务"""
        with get_db_session() as session:
            task = Task(**task_data)
            session.add(task)
            session.flush()
            session.refresh(task)
            session.expunge(task)
            return task
    
    def get_by_id(self, task_id: int) -> Optional[Task]:
        """根据ID获取任务"""
        with get_db_session() as session:
            return session.query(Task).filter(Task.id == task_id).first()
    
    def get_by_name(self, name: str) -> Optional[Task]:
        """根据名称获取任务"""
        with get_db_session() as session:
            return session.query(Task).filter(Task.name == name).first()
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Task]:
        """获取所有任务"""
        with get_db_session() as session:
            return session.query(Task).offset(offset).limit(limit).all()
    
    def get_by_status(self, status: TaskStatus) -> List[Task]:
        """根据状态获取任务"""
        with get_db_session() as session:
            return session.query(Task).filter(Task.status == status).all()
    
    def get_by_node_id(self, node_id: int) -> List[Task]:
        """根据节点ID获取任务"""
        with get_db_session() as session:
            return session.query(Task).filter(Task.node_id == node_id).all()
    
    def get_by_user_id(self, user_id: int) -> List[Task]:
        """根据用户ID获取任务"""
        with get_db_session() as session:
            return session.query(Task).filter(Task.user_id == user_id).all()
    
    def get_pending_tasks(self, priority_min: int = 1) -> List[Task]:
        """获取等待中的任务"""
        with get_db_session() as session:
            return session.query(Task).filter(
                and_(
                    Task.status == TaskStatus.PENDING,
                    Task.priority >= priority_min
                )
            ).order_by(Task.priority.desc(), Task.created_at.asc()).all()
    
    def update(self, task_id: int, update_data: Dict[str, Any]) -> Optional[Task]:
        """更新任务"""
        with get_db_session() as session:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                for key, value in update_data.items():
                    setattr(task, key, value)
                session.flush()
                session.refresh(task)
                session.expunge(task)
            return task
    
    def update_status(self, task_id: int, status: TaskStatus) -> Optional[Task]:
        """更新任务状态"""
        return self.update(task_id, {'status': status})
    
    def delete(self, task_id: int) -> bool:
        """删除任务"""
        with get_db_session() as session:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                session.delete(task)
                return True
            return False
    
    def search(self, keyword: str, limit: int = 50) -> List[Task]:
        """搜索任务"""
        with get_db_session() as session:
            return session.query(Task).filter(
                or_(
                    Task.name.contains(keyword),
                    Task.description.contains(keyword)
                )
            ).limit(limit).all()
    
    def count_by_status(self) -> Dict[str, int]:
        """按状态统计任务数量"""
        with get_db_session() as session:
            result = {}
            for status in TaskStatus:
                count = session.query(Task).filter(Task.status == status).count()
                result[status.value] = count
            return result