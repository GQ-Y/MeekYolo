from pydantic import BaseModel, Field
from typing import Dict, Optional

class ModelInfo(BaseModel):
    """模型信息"""
    name: str = Field(..., description="模型名称")
    code: str = Field(..., description="模型编码")
    version: str = Field("1.0.0", description="版本号")
    author: str = Field("", description="作者")
    description: str = Field("", description="描述")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "工地车辆检测模型",
                "code": "model-gcc",
                "version": "1.0.0",
                "author": "AI Team",
                "description": "用于检测渣土车、挖掘机、吊车等工程车辆"
            }
        }

class ModelList(BaseModel):
    """模型列表"""
    models: list[ModelInfo]
    current_model: str  # 当前使用的模型编码 