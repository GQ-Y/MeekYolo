from fastapi import FastAPI
from api.core.config import settings
from api.core.events import startup_event, shutdown_event
from api.routers import analysis_router, tasks_router, models_router, config_router, system_router

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_tags=[
            {
                "name": "analysis",
                "description": "分析相关接口，包括图片、视频、RTSP流分析"
            },
            {
                "name": "tasks",
                "description": "任务管理接口"
            },
            {
                "name": "models",
                "description": "模型管理接口"
            },
            {
                "name": "config",
                "description": "配置管理接口"
            },
            {
                "name": "system",
                "description": "系统接口"
            }
        ]
    )
    
    # 注册事件处理器
    app.add_event_handler("startup", startup_event)
    app.add_event_handler("shutdown", shutdown_event)
    
    # 注册路由
    app.include_router(analysis_router, prefix="/analyze", tags=["analysis"])
    app.include_router(tasks_router, prefix="/tasks", tags=["tasks"])
    print("注册路由:")
    print(f"- models_router: {models_router}")
    app.include_router(models_router, prefix="/models", tags=["models"])
    app.include_router(config_router, prefix="/config", tags=["config"])
    app.include_router(system_router, prefix="/system", tags=["system"])
    
    return app 