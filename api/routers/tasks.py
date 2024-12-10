from fastapi import APIRouter, HTTPException
from typing import Optional
from api.services.tasks import TaskService

router = APIRouter()
task_service = TaskService()

@router.get("/rtsp")
async def list_rtsp_tasks(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 10
):
    """获取RTSP任务列表"""
    return await task_service.list_rtsp_tasks(status, skip, limit)

@router.get("/rtsp/stats")
async def get_rtsp_tasks_stats():
    """获取RTSP任务统计信息"""
    print("收到统计请求")
    return await task_service.get_rtsp_tasks_stats()

@router.delete("/rtsp/cleanup")
async def cleanup_rtsp_tasks(
    status: Optional[str] = None,
    force: bool = False
):
    """清理RTSP分析任务"""
    print(f"收到清理请求: status={status}, force={force}")
    return await task_service.cleanup_rtsp_tasks(status, force)

@router.get("/rtsp/{task_id}")
async def get_rtsp_task(task_id: str):
    """获取RTSP分析任务状态"""
    return await task_service.get_rtsp_task(task_id)

@router.delete("/rtsp/{task_id}")
async def stop_rtsp_analysis(task_id: str):
    """停止RTSP流分析"""
    return await task_service.stop_rtsp_analysis(task_id)

@router.delete("/rtsp/{task_id}/force")
async def force_delete_rtsp_task(task_id: str):
    """强制删除RTSP分析任务"""
    return await task_service.force_delete_rtsp_task(task_id) 