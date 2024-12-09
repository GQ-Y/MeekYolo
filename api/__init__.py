from fastapi import FastAPI
from api.core.config import settings
from api.core.events import startup_event, shutdown_event
from api.routers import analysis, tasks, models, config, system

# API标签定义
tags_metadata = [
    {
        "name": "analysis",
        "description": "分析相关接口，包括图片、视频、RTSP流分析"
    },
    {
        "name": "tasks", 
        "description": "任务管理接口，用于查询、删除和管理分析任务"
    },
    {
        "name": "models",
        "description": "模型管理接口，用于上传、切换和管理检测模型"
    },
    {
        "name": "config",
        "description": "配置管理接口，用于设置回调等系统配置"
    },
    {
        "name": "system",
        "description": "系统接口，如健康检查等"
    }
]

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="YOLO Detection API",
        description="""
        YOLO目标检测API服务
        
        ## 功能特点
        * 支持图片、视频、RTSP流分析
        * 支持多任务并行处理
        * 实时回调推送
        * 自动重连机制
        """,
        version="1.0.0",
        openapi_tags=tags_metadata,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 注册事件处理
    app.add_event_handler("startup", startup_event)
    app.add_event_handler("shutdown", shutdown_event)

    # 注册路由
    app.include_router(analysis.router, prefix="/analyze", tags=["analysis"])
    app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
    app.include_router(models.router, prefix="/models", tags=["models"]) 
    app.include_router(config.router, prefix="/config", tags=["config"])
    app.include_router(system.router, tags=["system"])

    return app 