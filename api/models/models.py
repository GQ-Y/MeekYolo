from pydantic import BaseModel
from datetime import datetime

class ModelInfo(BaseModel):
    """模型信息"""
    code: str                # 模型编码
    version: str            # 模型版本
    name: str              # 模型名称
    description: str       # 模型描述
    author: str           # 作者
    create_time: datetime  # 创建时间
    update_time: datetime  # 更新时间
    path: str             # 模型路径

class ModelList(BaseModel):
    """模型列表"""
    models: list[ModelInfo]
    current_model: str  # 当前使用的模型编码 