import cv2
import numpy as np
import uuid
from datetime import datetime
from fastapi import HTTPException, UploadFile
from typing import List, Dict
from api.models.analysis import (
    ImageRequest, VideoRequest, RtspRequest,
    AnalysisResult
)
from api.models.tasks import (
    VideoAnalysisTask,
    RtspAnalysisTask
)
from api.core.config import settings
from api.utils.image import decode_image, encode_image
from api.services.callback import CallbackService
from yolo_rtsp import MeekYolo
import os
import asyncio
import sys
import logging
from api.core.managers import task_manager

logger = logging.getLogger(__name__)

class AnalysisService:
    # 将所有共享状态作为类变量
    _rtsp_tasks: Dict[str, RtspAnalysisTask] = {}
    _rtsp_detectors: Dict[str, MeekYolo] = {}
    
    def __init__(self):
        self.callback_service = CallbackService()
        self.video_tasks: Dict[str, VideoAnalysisTask] = {}
        self.model_config = settings.model
        self.source_config = settings.source
        self.visualization_config = settings.visualization
        self.tracking_config = settings.tracking
        
        # 注册回调函数到重连管理
        from api.services.reconnect import reconnect_manager
        reconnect_manager.register_callbacks(
            process_rtsp_task=self._process_rtsp_task,
            send_callback=self.callback_service.send_callback
        )
        reconnect_manager.rtsp_tasks = self._rtsp_tasks

    @property
    def rtsp_detectors(self) -> Dict[str, MeekYolo]:
        """提供rtsp_detectors的访问接口"""
        return self._rtsp_detectors

    @property
    def rtsp_tasks(self) -> Dict[str, RtspAnalysisTask]:
        """提供rtsp_tasks的访问接口"""
        return self._rtsp_tasks

    @rtsp_tasks.setter
    def rtsp_tasks(self, value: Dict[str, RtspAnalysisTask]):
        """设置rtsp_tasks"""
        self._rtsp_tasks = value

    async def analyze_images(self, request: ImageRequest) -> List[AnalysisResult]:
        """分析图片"""
        try:
            detector = MeekYolo()
            results = []
            
            for image_data in request.images:
                frame = decode_image(image_data, request.is_base64)
                detections = detector.process_frame(frame)
                processed_frame = detector.draw_results(frame, detections)
                result_base64 = encode_image(processed_frame)
                
                results.append(AnalysisResult(
                    image_base64=result_base64,
                    detections=self.format_detections(detections)
                ))
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def analyze_uploaded_images(self, files: List[UploadFile]) -> List[AnalysisResult]:
        """分析上传的图片文件"""
        try:
            if not files:
                raise HTTPException(status_code=400, detail="未提供文件")
            
            detector = MeekYolo()
            results = []
            
            for file in files:
                try:
                    if not file.filename:
                        raise HTTPException(status_code=400, detail="文件名为空")
                    
                    if not file.content_type:
                        raise HTTPException(status_code=400, detail="无法确定文件类型")
                    
                    if not file.content_type.startswith('image/'):
                        raise HTTPException(
                            status_code=400, 
                            detail=f"文件 {file.filename} 不是图片格式 (content-type: {file.content_type})"
                        )
                    
                    # 读取文件内容
                    contents = await file.read()
                    if len(contents) == 0:
                        raise HTTPException(status_code=400, detail="文件为空")
                    
                    # 解码图片
                    nparr = np.frombuffer(contents, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        raise HTTPException(status_code=400, detail="无法解码图片文件")
                    
                    # 处理图片
                    detections = detector.process_frame(frame)
                    processed_frame = detector.draw_results(frame, detections)
                    result_base64 = encode_image(processed_frame)
                    
                    results.append(AnalysisResult(
                        image_base64=result_base64,
                        detections=self.format_detections(detections)
                    ))
                    
                except Exception as e:
                    print(f"处理文件 {file.filename} 失败: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"处理文件 {file.filename} 失败: {str(e)}")
            
            return results
            
        except Exception as e:
            print(f"分析上传文件失败: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def start_video_analysis(self, request: VideoRequest) -> VideoAnalysisTask:
        """启动视频分析"""
        task_id = str(uuid.uuid4())
        
        # 创建任务记录
        task = VideoAnalysisTask(
            task_id=task_id,
            status="pending",
            progress=0,
            created_at=datetime.now()
        )
        self.video_tasks[task_id] = task
        
        # 创建保存目录
        save_dir = os.path.join("results", "videos", task_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置存储路径
        result_path = os.path.join(save_dir, "result.mp4")
        
        # 启动异步处理
        asyncio.create_task(self._process_video_task(
            task_id=task_id,
            video_url=request.video_url,
            result_path=result_path
        ))
        
        return task

    async def _process_video_task(self, task_id: str, video_url: str, result_path: str):
        """处理视频分析任务"""
        try:
            task = self.video_tasks[task_id]
            task.status = "processing"
            await self.callback_service.send_callback(task_id, {
                "status": "processing",
                "progress": 0
            })
            
            # 创建检测器实例
            detector = MeekYolo()
            
            # 定义进度回调函数
            def progress_callback(progress: float):
                task.progress = progress
                asyncio.create_task(self.callback_service.send_callback(task_id, {
                    "status": "processing",
                    "progress": progress
                }))
            
            # 处理视频
            detector.process_video(
                video_url,
                result_path,
                progress_callback=progress_callback
            )
            
            # 更新任务状态
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result_path = result_path
            task.download_url = f"/download/video/{task_id}"
            
            # 发送完成通知
            await self.callback_service.send_callback(task_id, {
                "status": "completed",
                "progress": 1.0,
                "result_path": result_path,
                "download_url": f"/download/video/{task_id}"
            })
            
        except Exception as e:
            # 更新失败状态
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()
            
            # 发送失败通知
            await self.callback_service.send_callback(task_id, {
                "status": "failed",
                "error": str(e)
            })

    async def start_rtsp_analysis(self, request: RtspRequest) -> RtspAnalysisTask:
        """启动RTSP流分析"""
        try:
            # 检查是否存在相同的RTSP地址
            existing_task = None
            for task in self._rtsp_tasks.values():
                if task.rtsp_url == request.rtsp_url:
                    existing_task = task
                    break
            
            if existing_task:
                # 如果任务存在,根据状态返回不同的错误信息
                if existing_task.status in ["processing", "pending"]:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "该RTSP地址已在分析中",
                            "task_id": existing_task.task_id,
                            "status": existing_task.status,
                            "created_at": existing_task.created_at.isoformat()
                        }
                    )
                elif existing_task.status == "offline":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "该RTSP地址连接已断开，正在尝试重连",
                            "task_id": existing_task.task_id,
                            "status": existing_task.status,
                            "reconnect_count": existing_task.reconnect_count,
                            "last_reconnect": existing_task.last_reconnect.isoformat() if existing_task.last_reconnect else None
                        }
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": f"该RTSP地址已存在任务（状态：{existing_task.status}），请先删除原任务",
                            "task_id": existing_task.task_id,
                            "status": existing_task.status,
                            "stopped_at": existing_task.stopped_at.isoformat() if existing_task.stopped_at else None
                        }
                    )
            
            # 创建新任务
            task_id = str(uuid.uuid4())
            output_rtmp = request.output_rtmp or f"rtmp://your_server/live/{task_id}"
            
            # 创建任务记录
            task = RtspAnalysisTask(
                task_id=task_id,
                status="pending",
                rtsp_url=request.rtsp_url,
                stream_url=output_rtmp,
                created_at=datetime.now(),
                callback_interval=request.callback_interval
            )
            self._rtsp_tasks[task_id] = task
            
            # 在任务管理器中创建任务
            await task_manager.create_task(task_id, request.rtsp_url)
            
            # 启动异步处理
            asyncio.create_task(self._process_rtsp_task(
                task_id=task_id,
                rtsp_url=request.rtsp_url,
                output_rtmp=output_rtmp,
                callback_interval=request.callback_interval
            ))
            
            return task
            
        except Exception as e:
            logger.error(f"启动RTSP分析失败: {str(e)}")
            raise

    async def _process_rtsp_task(self, task_id: str, rtsp_url: str, output_rtmp: str, callback_interval: float):
        """处理RTSP分析任务"""
        try:
            task = self._rtsp_tasks[task_id]
            task.status = "processing"
            
            # 创建检测器实例
            detector = MeekYolo()
            self._rtsp_detectors[task_id] = detector
            
            # 配置检测器
            detector.config.update({
                'environment': {
                    'enable_gui': False
                },
                'display': {
                    'show_window': False
                },
                'source': {
                    'type': 'rtsp',
                    'rtsp': {
                        'url': rtsp_url,
                        'timeout': settings.source['rtsp'].get('timeout', 5),
                        'connection_timeout': settings.source['rtsp'].get('connection_timeout', 10),
                        'read_timeout': settings.source['rtsp'].get('read_timeout', 5),
                        'max_retries': settings.source['rtsp'].get('max_retries', 3),
                        'retry_delay': settings.source['rtsp'].get('retry_delay', 5)
                    }
                }
            })
            
            if output_rtmp:
                detector.config['output'] = {
                    'enabled': True,
                    'rtmp_url': output_rtmp
                }
            
            # 设置任务信息和回调函数
            async def callback_wrapper(task_id: str, data: dict):
                try:
                    await self.callback_service.send_callback(task_id, data)
                except Exception as e:
                    logger.error(f"回调函数异常: {str(e)}")
            
            # 设置任务信息
            detector.set_task_info(
                task_id=task_id,
                callback_func=lambda data: asyncio.create_task(callback_wrapper(task_id, data)),
                callback_interval=callback_interval
            )
            
            # 发送开始通知
            await self.callback_service.send_callback(task_id, {
                "status": "processing",
                "rtsp_url": rtsp_url,
                "stream_url": output_rtmp
            })
            
            # 运行检测器
            await detector.process_rtsp(rtsp_url)
            
        except Exception as e:
            logger.error(f"处理RTSP分析任务异常: {str(e)}")
            task.status = "failed"
            task.error = str(e)
            
            # 更新任务管理器中的任务状态
            await task_manager.update_task_status(task_id, 'failed')
            
            try:
                # 发送错误通知
                await self.callback_service.send_callback(task_id, {
                    "status": "failed",
                    "error": str(e)
                })
            except Exception as callback_error:
                logger.error(f"发送错误通知失败: {str(callback_error)}")
            
        finally:
            # 清理资源
            if task_id in self._rtsp_detectors:
                try:
                    detector = self._rtsp_detectors[task_id]
                    detector.stop()  # 停止检测器
                    del self._rtsp_detectors[task_id]
                except Exception as cleanup_error:
                    logger.error(f"清理资源失败: {str(cleanup_error)}")

    async def analyze_rtsp(self, url: str) -> dict:
        try:
            # 创建检测器实例
            detector = MeekYolo()
            
            # 测试RTSP连接
            try:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    raise Exception("无法打开RTSP流")
                cap.release()
            except Exception as e:
                raise Exception(f"RTSP连接测试失败: {str(e)}")
            
            # 返回结果
            return {
                "status": "success",
                "url": url
            }
            
        except Exception as e:
            logger.error(f"RTSP分析失败: {str(e)}")
            raise

    async def test_rtsp_connection(self, request: RtspRequest) -> dict:
        """测试RTSP流连接"""
        try:
            # 创建检测器实例
            detector = MeekYolo()
            
            # 测试RTSP连接
            try:
                # 使用 apipreference 参数指定使用 FFMPEG
                cap = cv2.VideoCapture(request.rtsp_url, apiPreference=cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    raise Exception("无法打开RTSP流")
                
                # 获取视频流信息
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # 读取一帧测试
                ret, frame = cap.read()
                if not ret:
                    raise Exception("无法读取视频帧")
                    
                cap.release()
                
                # 返回详细信息
                return {
                    "status": "success",
                    "url": request.rtsp_url,
                    "stream_info": {
                        "width": width,
                        "height": height,
                        "fps": fps
                    },
                    "message": "RTSP流连接测试成功"
                }
                
            except Exception as e:
                raise Exception(f"RTSP连接测试失败: {str(e)}")
                
        except Exception as e:
            logger.error(f"RTSP连接测试失败: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "url": request.rtsp_url,  # 只使用 URL 字符串
                    "message": str(e)
                }
            )

    def format_detections(self, results) -> List[dict]:
        """格式化检测结果"""
        formatted = []
        for box, score, cls_name, track_id in results:
            x1, y1, x2, y2 = map(int, box)
            formatted.append({
                "track_id": int(track_id),
                "class": cls_name,
                "confidence": float(score),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })
        return formatted 

    async def add_rtsp_task(self, task_id: str, rtsp_url: str):
        """添加RTSP分析任务"""
        try:
            # ... 现有的添加任务代码 ...
            
            # 同步到任务管理器
            await task_manager.sync_tasks(self._rtsp_tasks)
            
        except Exception as e:
            logger.error(f"添加RTSP任务失败: {str(e)}")
            raise

    async def update_rtsp_task_status(self, task_id: str, status: str):
        """更新RTSP任务状态"""
        try:
            if task_id in self._rtsp_tasks:
                self._rtsp_tasks[task_id].status = status
                # 同步到任务管理器
                await task_manager.sync_tasks(self._rtsp_tasks)
                
        except Exception as e:
            logger.error(f"更新任务状态失败: {str(e)}")
            raise