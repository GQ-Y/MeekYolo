import asyncio
import cv2
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any
from api.core.config import settings
from api.models.tasks import RtspAnalysisTask

class RtspReconnectManager:
    def __init__(self):
        self.reconnect_interval = 60  # 重连间隔(秒)
        self.running = True
        self.reconnect_task = None
        self.rtsp_tasks: Dict[str, RtspAnalysisTask] = {}
        self._process_rtsp_task_callback: Optional[Callable] = None
        self._send_callback: Optional[Callable] = None
    
    def register_callbacks(self, 
                         process_rtsp_task: Callable,
                         send_callback: Callable):
        """注册回调函数"""
        self._process_rtsp_task_callback = process_rtsp_task
        self._send_callback = send_callback

    async def start_reconnect_monitor(self):
        """启动重连监控"""
        if self.reconnect_task is None:
            self.running = True
            self.reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def stop_reconnect_monitor(self):
        """停止重连监控"""
        self.running = False
        if self.reconnect_task:
            self.reconnect_task.cancel()
            self.reconnect_task = None
    
    async def _reconnect_loop(self):
        """重连循环"""
        while self.running:
            try:
                # 检查所有离线的任务
                offline_tasks = [
                    task_id for task_id, task in self.rtsp_tasks.items()
                    if task.status == "offline"
                ]
                
                for task_id in offline_tasks:
                    task = self.rtsp_tasks[task_id]
                    # 检查是否到达重连时间
                    if (task.last_reconnect is None or 
                        (datetime.now() - task.last_reconnect).total_seconds() >= self.reconnect_interval):
                        await self._try_reconnect(task_id)
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                print(f"重连监控异常: {str(e)}")
                await asyncio.sleep(10)
    
    async def _try_reconnect(self, task_id: str):
        """尝试重连"""
        if not self._process_rtsp_task_callback or not self._send_callback:
            print("回调函数未注册")
            return

        task = self.rtsp_tasks[task_id]
        task.last_reconnect = datetime.now()
        task.reconnect_count += 1
        
        try:
            # 测试连接
            cap = cv2.VideoCapture(task.rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"重连失败 - 任务 {task_id}, 尝试次数: {task.reconnect_count}")
                cap.release()
                return
            
            # 读取一帧测试
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"重连失败(无法读取帧) - 任务 {task_id}")
                return
            
            print(f"重连成功 - 任务 {task_id}")
            # 更新任务状态
            task.status = "pending"
            
            # 重新启动检测任务
            asyncio.create_task(self._process_rtsp_task_callback(
                task_id=task_id,
                rtsp_url=task.rtsp_url,
                output_rtmp=task.stream_url,
                callback_interval=task.callback_interval
            ))
            
            # 发送重连成功通知
            await self._send_callback(task_id, {
                "status": "reconnected",
                "rtsp_url": task.rtsp_url,
                "stream_url": task.stream_url,
                "reconnect_info": {
                    "count": task.reconnect_count,
                    "time": task.last_reconnect.isoformat(),
                    "success": True
                }
            })
            
        except Exception as e:
            print(f"重连异常 - 任务 {task_id}: {str(e)}")
            # 发送重连失败通知
            await self._send_callback(task_id, {
                "status": "reconnect_failed",
                "error": str(e),
                "reconnect_info": {
                    "count": task.reconnect_count,
                    "time": task.last_reconnect.isoformat(),
                    "success": False
                }
            })

reconnect_manager = RtspReconnectManager() 