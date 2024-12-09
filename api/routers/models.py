from fastapi import APIRouter, File, UploadFile, HTTPException
from api.services.model import ModelService
from api.models.models import ModelInfo, ModelList

router = APIRouter()
model_service = ModelService()

@router.get("", response_model=ModelList)
async def get_models():
    """获取模型列表"""
    return await model_service.get_model_list()

@router.post("/upload", response_model=ModelInfo)
async def upload_model(file: UploadFile = File(...)):
    """
    上传模型
    
    - **file**: 模型文件(zip格式)
    
    模型文件须包含:
    - best.pt: 模型文件
    - data.yaml: 配置文件
    """
    return await model_service.upload_model(file)

@router.post("/{model_code}/set")
async def set_model(model_code: str):
    """
    设置当前使用的模型
    
    - **model_code**: 模型编码
    """
    await model_service.set_current_model(model_code)
    return {"message": "模型设置成功"} 