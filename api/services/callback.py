import aiohttp
import asyncio
import logging
from typing import Dict
from api.core.config import settings

# 初始化logger
logger = logging.getLogger(__name__)

class CallbackService:
    def __init__(self):
        self.config = settings.CALLBACK_CONFIG
    
    async def send_callback_with_retry(self, task_id: str, data: dict):
        """带重试的回调发送"""
        if not self.config['enabled'] or not self.config['url']:
            return
        
        retry_config = self.config['retry']
        max_retries = retry_config['max_retries']
        retry_delay = retry_config['retry_delay']
        timeout = retry_config['timeout']
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.config['url'],
                        json={"task_id": task_id, **data},
                        timeout=timeout
                    ) as response:
                        if response.status == 200:
                            return True
                        
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"回调发送失败(HTTP {response.status})，{retry_count}/{max_retries} 次重试中...")
                    await asyncio.sleep(retry_delay)
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"回调发送失败({str(e)})，{retry_count}/{max_retries} 次重试中...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"回调发送最终失败: {str(e)}")
                    return False
        
        logger.error(f"回调发送失败，已达到最大重试次数({max_retries})")
        return False
    
    async def send_callback(self, task_id: str, data: dict):
        """发送回调通知(不阻塞主流程)"""
        if not self.config['enabled'] or not self.config['url']:
            return
        
        # 创建新的任务处理回调发送(包括重试)
        asyncio.create_task(self.send_callback_with_retry(task_id, data))