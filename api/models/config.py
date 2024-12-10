from pydantic import BaseModel

class RetryConfig(BaseModel):
    """重试配置"""
    max_retries: int = 10
    retry_delay: float = 1.0
    timeout: int = 10

class CallbackConfig(BaseModel):
    """回调配置"""
    url: str
    enabled: bool = True
    max_retries: int = 10
    retry_delay: float = 1.0
    timeout: int = 10

    class Config:
        json_schema_extra = {
            "example": {
                "url": "http://example.com/callback",
                "enabled": True,
                "max_retries": 10,
                "retry_delay": 1.0,
                "timeout": 10
            }
        }