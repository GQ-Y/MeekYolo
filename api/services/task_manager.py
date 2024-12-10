import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
import cv2

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self):
        # 使用字典存储任务信息
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
        self.failed_tasks_checker = None
        logger.info("任务管理器实例创建完成")
        
    async def initialize(self):
        """初始化任务管理器"""
        if self.initialized:
            return
            
        try:
            self.initialized = True
            logger.info(f"任务管理器初始化完成，当前任务数: {len(self.tasks)}")
            
        except Exception as e:
            logger.error(f"任务管理器初始化失败: {str(e)}")
            raise
            
    async def shutdown(self):
        """关闭任务管理器"""
        try:
            logger.info("开始关闭任务管理器...")
            
            # 停止所有任务
            for task_id in list(self.tasks.keys()):
                try:
                    await self.delete_task(task_id)
                except Exception as e:
                    logger.error(f"关闭任务 {task_id} 时发生错误: {str(e)}")
            
            self.initialized = False
            logger.info("任务管理器已关闭")
            
        except Exception as e:
            logger.error(f"关闭任务管理器时发生错误: {str(e)}")
            
    async def start_failed_tasks_checker(self):
        """启动失败任务检查器"""
        self.failed_tasks_checker = asyncio.create_task(self.check_failed_tasks())
        logger.info("失败任务检查器已启动")
        
    async def check_failed_tasks(self):
        """检查失败任务并尝试重新连接"""
        logger.info("开始运行失败任务检查循环")
        check_count = 0
        
        while True:
            try:
                check_count += 1
                # 获取所有任务
                all_tasks = list(self.tasks.values())
                failed_tasks = [task for task in all_tasks if task.get('status') == 'failed']
                
                logger.info(f"当前任务总数: {len(all_tasks)}")
                logger.info(f"失败任务数量: {len(failed_tasks)}")
                
                if failed_tasks:
                    logger.info("开始处理失败任务...")
                    for task in failed_tasks:
                        task_id = task.get('task_id')
                        rtsp_url = task.get('rtsp_url')
                        logger.info(f"尝试恢复任务: {task_id} ({rtsp_url})")
                        
                        try:
                            # 尝试重新连接RTSP流
                            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                            if not cap.isOpened():
                                logger.warning(f"任务 {task_id} 重连失败: 无法打开RTSP流")
                                continue
                                
                            # 测试读取帧
                            ret, frame = cap.read()
                            if not ret:
                                logger.warning(f"任务 {task_id} 重连失败: 无法读取视频帧")
                                continue
                                
                            cap.release()
                            
                            # 连接成功，更新任务状态
                            self.tasks[task_id]['status'] = 'pending'
                            self.tasks[task_id]['updated_at'] = datetime.now().isoformat()
                            logger.info(f"任务 {task_id} 重连成功，状态更新为 pending")
                            
                            # 重新启动RTSP分析任务
                            from api.services.analysis import AnalysisService
                            analysis_service = AnalysisService()
                            asyncio.create_task(analysis_service._process_rtsp_task(
                                task_id=task_id,
                                rtsp_url=rtsp_url,
                                output_rtmp=task.get('stream_url', ''),
                                callback_interval=task.get('callback_interval', 1.0)
                            ))
                            
                        except Exception as e:
                            logger.error(f"任务 {task_id} 重连失败: {str(e)}")
                            continue
                
            except Exception as e:
                logger.error(f"检查失败任务时发生错误: {str(e)}")
            
            # 每分钟检查一次
            await asyncio.sleep(60)

    async def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务
        
        Returns:
            List[Dict[str, Any]]: 任务列表，每个任务包含:
                {
                    'task_id': str,      # 任务ID
                    'rtsp_url': str,     # RTSP流地址
                    'status': str,       # 任务状态 (pending/running/failed)
                    'created_at': str,   # 创建时间
                    'updated_at': str    # 更新时间
                }
        """
        try:
            # 返回所有任务的列表
            return list(self.tasks.values())
        except Exception as e:
            logger.error(f"获取所有任务时发生错误: {str(e)}")
            return []

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取指定任务的信息
        
        Args:
            task_id (str): 任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 任务信息，如果任务不存在则返回None
        """
        try:
            return self.tasks.get(task_id)
        except Exception as e:
            logger.error(f"获取任务 {task_id} 时发生错误: {str(e)}")
            return None

    async def create_task(self, task_id: str, rtsp_url: str) -> Dict[str, Any]:
        """创建新任务
        
        Args:
            task_id (str): 任务ID
            rtsp_url (str): RTSP流地址
            
        Returns:
            Dict[str, Any]: 创建的任务信息
        """
        try:
            # 检查任务是��已存在
            if task_id in self.tasks:
                logger.warning(f"任务已存在: {task_id}")
                return self.tasks[task_id]
            
            now = datetime.now().isoformat()
            task = {
                'task_id': task_id,
                'rtsp_url': rtsp_url,
                'status': 'pending',  # 初始状态为pending
                'created_at': now,
                'updated_at': now
            }
            self.tasks[task_id] = task
            logger.info(f"创建任务成功: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"创建任务 {task_id} 时发生错误: {str(e)}")
            raise

    async def update_task_status(self, task_id: str, status: str) -> bool:
        """更新任务状态"""
        try:
            # 验证状态是否有效
            if status not in ['pending', 'running', 'failed']:
                raise ValueError(f"无效的任务状态: {status}")
            
            # 如果任务不存在，先创建任务
            if task_id not in self.tasks:
                logger.warning(f"任务不存在，将创建新任务: {task_id}")
                # 从RTSP任务中获取URL
                from api.services.analysis import AnalysisService
                rtsp_task = AnalysisService._rtsp_tasks.get(task_id)
                if rtsp_task:
                    await self.create_task(task_id, rtsp_task.rtsp_url)
                else:
                    raise KeyError(f"任务不存在: {task_id}")
            
            # 更新状态和更新时间
            self.tasks[task_id]['status'] = status
            self.tasks[task_id]['updated_at'] = datetime.now().isoformat()
            
            logger.info(f"已更新任务 {task_id} 的状态为 {status}")
            return True
            
        except Exception as e:
            logger.error(f"更新任务 {task_id} 状态时发生错误: {str(e)}")
            raise

    async def delete_task(self, task_id: str) -> bool:
        """删除任务
        
        Args:
            task_id (str): 任务ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.info(f"已删除任务: {task_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除任务 {task_id} 时发生错误: {str(e)}")
            return False

    async def get_failed_tasks(self) -> List[Dict[str, Any]]:
        """获取所有失败状态的任务
        
        Returns:
            List[Dict[str, Any]]: 失败任务列表
        """
        try:
            failed_tasks = [
                task for task in self.tasks.values()
                if task.get('status') == 'failed'
            ]
            logger.info(f"获取到 {len(failed_tasks)} 个失败状态的任务")
            return failed_tasks
        except Exception as e:
            logger.error(f"获取失败任务列表时发生错误: {str(e)}")
            return []

    async def sync_tasks(self, rtsp_tasks: dict):
        """同步RTSP任务到任务管理器
        
        Args:
            rtsp_tasks (dict): RTSP任务字典，key为task_id
        """
        try:
            logger.info(f"开始同步RTSP任务，任务数量: {len(rtsp_tasks)}")
            
            # 更新或添加任务
            for task_id, task in rtsp_tasks.items():
                if task_id not in self.tasks:
                    # 新任务
                    self.tasks[task_id] = {
                        'task_id': task_id,
                        'rtsp_url': task.rtsp_url,  # 假设task对象有rtsp_url属性
                        'status': task.status,      # 假设task对象有status属性
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }
                else:
                    # 更新现有任务状态
                    self.tasks[task_id]['status'] = task.status
                    self.tasks[task_id]['updated_at'] = datetime.now().isoformat()
            
            logger.info(f"RTSP任务同步完成，当前任务总数: {len(self.tasks)}")
            
        except Exception as e:
            logger.error(f"同步RTSP任务失败: {str(e)}")