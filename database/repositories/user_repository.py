"""
用户仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from database.models.user import User
from database.connection import get_db_session

class UserRepository:
    """用户数据访问层"""
    
    def create(self, user_data: Dict[str, Any]) -> User:
        """创建用户"""
        with get_db_session() as session:
            user = User(**user_data)
            session.add(user)
            session.flush()
            session.refresh(user)
            # 确保对象可以在会话外使用
            session.expunge(user)
            return user
    
    def get_by_id(self, user_id: int) -> Optional[User]:
        """根据ID获取用户"""
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                session.expunge(user)
            return user
    
    def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        with get_db_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if user:
                session.expunge(user)
            return user
    
    def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        with get_db_session() as session:
            return session.query(User).filter(User.email == email).first()
    
    def get_all_active(self, limit: int = 100, offset: int = 0) -> List[User]:
        """获取所有激活用户"""
        with get_db_session() as session:
            return session.query(User).filter(
                User.is_active == True
            ).offset(offset).limit(limit).all()
    
    def get_admins(self) -> List[User]:
        """获取所有管理员"""
        with get_db_session() as session:
            return session.query(User).filter(
                and_(
                    User.is_admin == True,
                    User.is_active == True
                )
            ).all()
    
    def update(self, user_id: int, update_data: Dict[str, Any]) -> Optional[User]:
        """更新用户"""
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                for key, value in update_data.items():
                    setattr(user, key, value)
                session.flush()
                session.refresh(user)
            return user
    
    def update_password(self, user_id: int, password_hash: str) -> Optional[User]:
        """更新用户密码"""
        return self.update(user_id, {'password_hash': password_hash})
    
    def activate_user(self, user_id: int) -> Optional[User]:
        """激活用户"""
        return self.update(user_id, {'is_active': True})
    
    def deactivate_user(self, user_id: int) -> Optional[User]:
        """停用用户"""
        return self.update(user_id, {'is_active': False})
    
    def set_admin(self, user_id: int, is_admin: bool = True) -> Optional[User]:
        """设置管理员权限"""
        return self.update(user_id, {'is_admin': is_admin})
    
    def delete(self, user_id: int) -> bool:
        """删除用户"""
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                session.delete(user)
                return True
            return False
    
    def search(self, keyword: str, limit: int = 50) -> List[User]:
        """搜索用户"""
        with get_db_session() as session:
            return session.query(User).filter(
                or_(
                    User.username.contains(keyword),
                    User.full_name.contains(keyword),
                    User.email.contains(keyword)
                )
            ).limit(limit).all()
    
    def count_active_users(self) -> int:
        """统计激活用户数量"""
        with get_db_session() as session:
            return session.query(User).filter(User.is_active == True).count()
    
    def exists_username(self, username: str) -> bool:
        """检查用户名是否存在"""
        with get_db_session() as session:
            return session.query(User).filter(User.username == username).first() is not None
    
    def exists_email(self, email: str) -> bool:
        """检查邮箱是否存在"""
        with get_db_session() as session:
            return session.query(User).filter(User.email == email).first() is not None