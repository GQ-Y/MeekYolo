import logging
from api.services.reconnect import reconnect_manager
from api.core.managers import initialize_managers, shutdown_managers

logger = logging.getLogger(__name__)

async def startup_event():
    """程序启动时的事件处理"""
    logger.info("程序启动...")
    
    # 启动重连管理器
    await reconnect_manager.start_reconnect_monitor()
    
    # 初始化所有管理器
    await initialize_managers()
    
    logger.info("启动完成")

async def shutdown_event():
    """程序关闭时的事件处理"""
    logger.info("程序关闭...")
    
    await reconnect_manager.stop_reconnect_monitor()
    await shutdown_managers() 