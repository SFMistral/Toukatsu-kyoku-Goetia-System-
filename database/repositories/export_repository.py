"""
导出仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from database.models.export_record import ExportRecord
from database.connection import get_db_session

class ExportRepository:
    """导出数据访问层"""
    
    def create(self, export_data: Dict[str, Any]) -> ExportRecord:
        """创建导出记录"""
        with get_db_session() as session:
            export_record = ExportRecord(**export_data)
            session.add(export_record)
            session.flush()
            session.refresh(export_record)
            return export_record
    
    def get_by_id(self, export_id: int) -> Optional[ExportRecord]:
        """根据ID获取导出记录"""
        with get_db_session() as session:
            return session.query(ExportRecord).filter(ExportRecord.id == export_id).first()
    
    def get_by_user_id(self, user_id: int) -> List[ExportRecord]:
        """根据用户ID获取导出记录"""
        with get_db_session() as session:
            return session.query(ExportRecord).filter(
                ExportRecord.user_id == user_id
            ).order_by(desc(ExportRecord.created_at)).all()
    
    def get_by_type(self, export_type: str, limit: int = 50) -> List[ExportRecord]:
        """根据导出类型获取记录"""
        with get_db_session() as session:
            return session.query(ExportRecord).filter(
                ExportRecord.export_type == export_type
            ).order_by(desc(ExportRecord.created_at)).limit(limit).all()
    
    def get_recent_exports(self, limit: int = 20) -> List[ExportRecord]:
        """获取最近的导出记录"""
        with get_db_session() as session:
            return session.query(ExportRecord).order_by(
                desc(ExportRecord.created_at)
            ).limit(limit).all()
    
    def update(self, export_id: int, update_data: Dict[str, Any]) -> Optional[ExportRecord]:
        """更新导出记录"""
        with get_db_session() as session:
            export_record = session.query(ExportRecord).filter(ExportRecord.id == export_id).first()
            if export_record:
                for key, value in update_data.items():
                    setattr(export_record, key, value)
                session.flush()
                session.refresh(export_record)
            return export_record
    
    def delete(self, export_id: int) -> bool:
        """删除导出记录"""
        with get_db_session() as session:
            export_record = session.query(ExportRecord).filter(ExportRecord.id == export_id).first()
            if export_record:
                session.delete(export_record)
                return True
            return False
    
    def get_export_types(self) -> List[str]:
        """获取所有导出类型"""
        with get_db_session() as session:
            result = session.query(ExportRecord.export_type).distinct().all()
            return [row[0] for row in result]
    
    def search_by_name(self, keyword: str, limit: int = 50) -> List[ExportRecord]:
        """根据名称搜索导出记录"""
        with get_db_session() as session:
            return session.query(ExportRecord).filter(
                ExportRecord.name.contains(keyword)
            ).order_by(desc(ExportRecord.created_at)).limit(limit).all()