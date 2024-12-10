from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional, List
import yaml
import os

class Settings(BaseSettings):
    """应用配置"""
    # 基础配置
    PROJECT_NAME: str = "YOLO V11 Core API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # 回调配置
    callback: Dict[str, Any] = {
        'enabled': False,
        'url': None,
        'retry': {
            'max_retries': 10,
            'retry_delay': 1.0,
            'timeout': 10
        }
    }
    
    # 控制台配置
    console: Dict[str, bool] = {
        'enabled': True,
        'show_details': True,
        'show_separator': True,
        'show_time': True,
        'show_total': True
    }
    
    # 显示配置
    display: Dict[str, Any] = {
        'show_fps': True,
        'show_window': True,
        'window_name': 'MeekYolo Detection'
    }
    
    # 环境配置
    environment: Dict[str, bool] = {
        'enable_gui': True,
        'is_docker': False
    }
    
    # 模型配置
    model: Dict[str, Any] = {
        'conf_thres': 0.5,
        'path': 'model/model-gcc/best.pt'
    }
    
    # 模型管理配置
    model_management: Dict[str, Any] = {
        'model_dir': 'model',
        'temp_dir': 'temp/models',
        'model_config_template': {
            'name': '',
            'code': '',
            'version': '1.0.0',
            'author': '',
            'description': '',
            'create_time': '',
            'update_time': ''
        },
        'required_files': ['best.pt', 'data.yaml']
    }
    
    # 打印配置
    print: Dict[str, bool] = {
        'enabled': True
    }
    
    # 源配置
    source: Dict[str, Any] = {
        'type': 'rtsp',
        'image': {
            'path': 'data/test.png',
            'save_path': 'results/test_result.png'
        },
        'images': {
            'formats': ['.jpg', '.jpeg', '.png'],
            'input_dir': 'data/images',
            'save_dir': 'results/images'
        },
        'video': {
            'path': 'data/video/test2.mp4',
            'save_path': 'results/video/test_result2.mp4',
            'fps': 30
        },
        'rtsp': {
            'url': None,
            'ffmpeg_options': ['?tcp']
        }
    }
    
    # 跟踪配置
    tracking: Dict[str, bool] = {
        'enabled': True,
        'persist': True
    }
    
    # 可视化配置
    visualization: Dict[str, Any] = {
        'show_anchor': True,
        'show_box': True,
        'show_class': True,
        'show_confidence': True,
        'show_line': True,
        'show_position': True,
        'show_size': True,
        'show_track_id': True,
        'style': {
            'chinese_text_size': 20,
            'colors': {
                'text': [255, 255, 255],
                'background': [0, 0, 0]
            },
            'font_scale': 0.6,
            'margin': 5,
            'thickness': 2
        }
    }
    
    # ZLMediaKit配置
    zlmediakit: Dict[str, Any] = {
        'enabled': True,
        'host': 'http://localhost:8000',
        'secret': '035c73f7-bb6b-4889-a715-d9eb2d1925cc',
        'rtsp': {
            'port': 8554,
            'tcp_mode': True
        }
    }

    class Config:
        extra = "allow"  # 允许额外的字段
        protected_namespaces = ('settings_',)  # 修改受保护的命名空间

    @classmethod
    def load_from_yaml(cls) -> 'Settings':
        config_path = "config/config.yaml"
        default_config_path = "config/default_config.yaml"
        
        # 如果存在用户配置，直接使用
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                if config_data:
                    return cls(**config_data)
            except Exception as e:
                print(f"Warning: Failed to load user config: {e}")
        
        # 如果用户配置不存在或加载失败，使用默认配置
        try:
            with open(default_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # 确保配置目录存在
            os.makedirs("config", exist_ok=True)
            
            # 保存默认配置为用户配置
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
            return cls(**config_data)
        except Exception as e:
            print(f"Error: Failed to load default config: {e}")
            raise

settings = Settings.load_from_yaml() 