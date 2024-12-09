from fastapi import APIRouter
from api.models.config import CallbackConfig
from api.services.config import ConfigService

router = APIRouter()
config_service = ConfigService()

@router.post("/callback")
async def set_callback_url(config: CallbackConfig):
    """设置回调API地址"""
    return await config_service.set_callback_url(config)

@router.get("/callback")
async def get_callback_url():
    """获取当前回调配置"""
    return await config_service.get_callback_url()

@router.delete("/callback")
async def delete_callback_url():
    """删除回调API地址"""
    return await config_service.delete_callback_url() 