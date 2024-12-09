from pydantic import BaseModel

class CallbackConfig(BaseModel):
    """回调API配置"""
    url: str 