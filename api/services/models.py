from api.core.config import settings

class ModelService:
    def __init__(self):
        self.model_config = settings.model  # 使用新的配置结构
        self.model_management = settings.model_management
        # ... 其余代码保持不变 ... 