import asyncio
import logging
from typing import Optional
from .zlmediakit_sdk import ZLMediaKitSDK
import yaml

logger = logging.getLogger(__name__)

class RTSPProxy:
    def __init__(self):
        self.sdk = ZLMediaKitSDK()
        self.players = {}
        
        # 从配置文件加载配置
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['zlmediakit']
    
    async def create_proxy(self, url: str, stream_id: str) -> tuple:
        """创建RTSP代理"""
        try:
            # 从配置获取端口号
            rtsp_port = self.config['rtsp'].get('port', 8554)
            
            # 创建代理播放器
            player = self.sdk.create_proxy(url, stream_id)
            if not player:
                return None, None
            
            # 生成代理URL，使用配置的端口
            proxy_url = f"rtsp://localhost:{rtsp_port}/live/{stream_id}"
            
            return player, proxy_url
            
        except Exception as e:
            logger.error(f"创建RTSP代理失败: {str(e)}")
            return None, None
    
    def close_proxy(self, player):
        """关闭RTSP代理"""
        try:
            if isinstance(player, tuple):
                player = player[0]  # 获取元组中的第一个元素
            if player:
                self.sdk.close_proxy(player)
            return True
            
        except Exception as e:
            logger.error(f"关闭RTSP代理失败: {str(e)}")
            return False
    
    async def _check_proxy_status(self, task_id: str) -> bool:
        """检查代理状态"""
        try:
            if task_id not in self.players:
                return False
            
            # 简单检查播放器实例是否存在
            player = self.players[task_id]
            return bool(player)
            
        except Exception as e:
            logger.error(f"检查代理状态失败: {str(e)}")
            return False