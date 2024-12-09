import yaml
from api.core.config import settings
from api.models.config import CallbackConfig

class ConfigService:
    def __init__(self):
        self.config = settings
    
    async def set_callback_url(self, config: CallbackConfig):
        """设置回调URL"""
        self.config.CALLBACK_CONFIG['enabled'] = True
        self.config.CALLBACK_CONFIG['url'] = config.url
        
        # 保存到配置文件
        with open('config/config.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        yaml_config['callback']['enabled'] = True
        yaml_config['callback']['url'] = config.url
        
        with open('config/config.yaml', 'w') as f:
            yaml.dump(yaml_config, f)
        
        return {
            "message": "回调地址设置成功",
            "callback_url": config.url,
            "retry_config": self.config.CALLBACK_CONFIG['retry']
        }
    
    async def get_callback_url(self):
        """获取回调配置"""
        return {
            "enabled": self.config.CALLBACK_CONFIG['enabled'],
            "callback_url": self.config.CALLBACK_CONFIG['url'],
            "retry_config": self.config.CALLBACK_CONFIG['retry']
        }
    
    async def delete_callback_url(self):
        """删除回调配置"""
        self.config.CALLBACK_CONFIG['enabled'] = False
        self.config.CALLBACK_CONFIG['url'] = None
        
        # 更新配置文件
        with open('config/config.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        yaml_config['callback']['enabled'] = False
        yaml_config['callback']['url'] = None
        
        with open('config/config.yaml', 'w') as f:
            yaml.dump(yaml_config, f)
        
        return {"message": "回调地址已删除"} 