"""
报告仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, or_
from database.models.report import Report
from database.connection import get_db_session

class ReportRepository:
    """报告数据访问层"""
    
    def create(self, report_data: Dict[str, Any]) -> Report:
        """创建报告"""
        with get_db_session() as session:
            report = Report(**report_data)
            session.add(report)
            session.flush()
            session.refresh(report)
            return report
    
    def get_by_id(self, report_id: int) -> Optional[Report]:
        """根据ID获取报告"""
        with get_db_session() as session:
            return session.query(Report).filter(Report.id == report_id).first()
    
    def get_by_user_id(self, user_id: int) -> List[Report]:
        """根据用户ID获取报告"""
        with get_db_session() as session:
            return session.query(Report).filter(
                Report.user_id == user_id
            ).order_by(desc(Report.created_at)).all()
    
    def get_by_type(self, report_type: str, limit: int = 50) -> List[Report]:
        """根据报告类型获取报告"""
        with get_db_session() as session:
            return session.query(Report).filter(
                Report.report_type == report_type
            ).order_by(desc(Report.created_at)).limit(limit).all()
    
    def get_recent_reports(self, limit: int = 20) -> List[Report]:
        """获取最近的报告"""
        with get_db_session() as session:
            return session.query(Report).order_by(
                desc(Report.created_at)
            ).limit(limit).all()
    
    def update(self, report_id: int, update_data: Dict[str, Any]) -> Optional[Report]:
        """更新报告"""
        with get_db_session() as session:
            report = session.query(Report).filter(Report.id == report_id).first()
            if report:
                for key, value in update_data.items():
                    setattr(report, key, value)
                session.flush()
                session.refresh(report)
            return report
    
    def delete(self, report_id: int) -> bool:
        """删除报告"""
        with get_db_session() as session:
            report = session.query(Report).filter(Report.id == report_id).first()
            if report:
                session.delete(report)
                return True
            return False
    
    def get_report_types(self) -> List[str]:
        """获取所有报告类型"""
        with get_db_session() as session:
            result = session.query(Report.report_type).distinct().all()
            return [row[0] for row in result]
    
    def search(self, keyword: str, limit: int = 50) -> List[Report]:
        """搜索报告"""
        with get_db_session() as session:
            return session.query(Report).filter(
                or_(
                    Report.title.contains(keyword),
                    Report.description.contains(keyword),
                    Report.summary.contains(keyword)
                )
            ).order_by(desc(Report.created_at)).limit(limit).all()