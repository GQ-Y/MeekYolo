import yaml
import os
from api.core.config import settings
from api.models.config import CallbackConfig

class ConfigService:
    def __init__(self):
        self.config = settings
        self.config_file = 'config/config.yaml'
    
    async def set_callback_url(self, config: CallbackConfig):
        """设置回调URL"""
        try:
            # 读取当前配置
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
            else:
                yaml_config = {}
            
            # 确保callback配置存在
            if 'callback' not in yaml_config:
                yaml_config['callback'] = {}
            
            # 更新配置
            yaml_config['callback'].update({
                'enabled': config.enabled,
                'url': config.url,
                'retry': {
                    'max_retries': config.max_retries,
                    'retry_delay': config.retry_delay,
                    'timeout': config.timeout
                }
            })
            
            # 保存到配置文件
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False)
            
            # 更新运行时配置
            self.config.callback.update({
                'enabled': config.enabled,
                'url': config.url,
                'retry': {
                    'max_retries': config.max_retries,
                    'retry_delay': config.retry_delay,
                    'timeout': config.timeout
                }
            })
            
            return {
                "message": "回调地址设置成功",
                "callback_url": config.url,
                "retry_config": {
                    'max_retries': config.max_retries,
                    'retry_delay': config.retry_delay,
                    'timeout': config.timeout
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"设置回调地址失败: {str(e)}"
            }
    
    async def get_callback_url(self):
        """获取回调配置"""
        return {
            "enabled": self.config.callback.get('enabled', False),
            "callback_url": self.config.callback.get('url'),
            "retry_config": self.config.callback.get('retry', {
                'max_retries': 10,
                'retry_delay': 1.0,
                'timeout': 10
            })
        }
    
    async def delete_callback_url(self):
        """删除回调配置"""
        try:
            # 更新运行时配置
            self.config.callback.update({
                'enabled': False,
                'url': None
            })
            
            # 读取当前配置
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                # 更新配置
                if 'callback' in yaml_config:
                    yaml_config['callback']['enabled'] = False
                    yaml_config['callback']['url'] = None
                    
                    # 保存到配置文件
                    with open(self.config_file, 'w') as f:
                        yaml.dump(yaml_config, f, default_flow_style=False)
            
            return {"message": "回调地址已删除"}
        except Exception as e:
            return {
                "status": "error",
                "message": f"删除回调地址失败: {str(e)}"
            }