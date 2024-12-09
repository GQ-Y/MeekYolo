from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional
import yaml
import os

class Settings(BaseSettings):
    """应用配置"""
    # 基础配置
    PROJECT_NAME: str = "YOLO Detection API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # RTSP连接配置
    RTSP_CONNECTION_CONFIG: Dict[str, Any] = {
        'max_retries': 3,
        'retry_delay': 5,
        'connection_timeout': 10,
        'read_timeout': 5
    }
    
    # 回调配置
    CALLBACK_CONFIG: Dict[str, Any] = {
        'enabled': False,
        'url': None,
        'retry': {
            'max_retries': 10,
            'retry_delay': 1.0,
            'timeout': 10
        }
    }
    
    # 模型配置
    MODEL_CONFIG: Dict[str, str] = {
        'model_dir': 'model',
        'temp_dir': 'temp'
    }

    def model_post_init(self, _: Any) -> None:
        """pydantic v2 中的初始化后处理"""
        self.load_yaml_config()
    
    def load_yaml_config(self) -> None:
        """从yaml加载配置"""
        config_path = 'config/config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            # 更新回调配置
            if 'callback' in yaml_config:
                self.CALLBACK_CONFIG['enabled'] = yaml_config['callback'].get('enabled', False)
                self.CALLBACK_CONFIG['url'] = yaml_config['callback'].get('url')
                if 'retry' in yaml_config['callback']:
                    self.CALLBACK_CONFIG['retry'].update(yaml_config['callback']['retry'])

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 