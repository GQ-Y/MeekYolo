from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    健康检查接口
    
    返回系统运行状态
    """
    return {"status": "ok"} 