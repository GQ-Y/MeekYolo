from api.services.reconnect import reconnect_manager

async def startup_event():
    """应用启动时的事件处理"""
    # 启动重连监控
    await reconnect_manager.start_reconnect_monitor()

async def shutdown_event():
    """应用关闭时的事件处理"""
    # 停止重连监控
    await reconnect_manager.stop_reconnect_monitor() 