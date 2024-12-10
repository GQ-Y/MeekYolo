import logging
import asyncio
from api.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

# 创建全局任务管理器实例
task_manager = TaskManager()

async def initialize_managers():
    """初始化所有管理器"""
    logger.info("初始化管理器...")
    
    try:
        # 初始化任务管理器
        await task_manager.initialize()
        
        # 同步现有RTSP任务
        from api.services.analysis import AnalysisService
        await task_manager.sync_tasks(AnalysisService._rtsp_tasks)
        
        # 启动失败任务检查循环
        asyncio.create_task(task_manager.check_failed_tasks())
        logger.info("任务管理器初始化成功，已启动失败任务检查循环")
        
    except Exception as e:
        logger.error(f"任务管理器初始化失败: {str(e)}")
        raise

async def shutdown_managers():
    """关闭所有管理器"""
    logger.info("关闭管理器...")
    
    try:
        await task_manager.shutdown()
        logger.info("任务管理器已关闭")
    except Exception as e:
        logger.error(f"任务管理器关闭失败: {str(e)}") 