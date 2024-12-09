from typing import Optional, Dict
from datetime import datetime
from fastapi import HTTPException
from api.models.tasks import VideoAnalysisTask, RtspAnalysisTask
from api.services.callback import CallbackService

class TaskService:
    def __init__(self):
        self.callback_service = CallbackService()
        self.rtsp_tasks: Dict[str, RtspAnalysisTask] = {}
        self.video_tasks: Dict[str, VideoAnalysisTask] = {}
        
    async def list_rtsp_tasks(self, status: Optional[str], skip: int, limit: int):
        """获取RTSP任务列表"""
        tasks = []
        for task in self.rtsp_tasks.values():
            if status and task.status != status:
                continue
            tasks.append(task)
        
        # 排序:先按状态(运行中的在前)再按创建时间(新的在前)
        tasks.sort(key=lambda x: (
            0 if x.status in ["processing", "pending", "offline"] else 1,
            x.created_at
        ), reverse=True)
        
        total = len(tasks)
        tasks = tasks[skip:skip + limit]
        
        return {
            "total": total,
            "tasks": tasks,
            "page": {
                "skip": skip,
                "limit": limit,
                "total_pages": (total + limit - 1) // limit
            }
        }
    
    async def get_rtsp_task(self, task_id: str):
        """获取RTSP任务状态"""
        if task_id not in self.rtsp_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
        return self.rtsp_tasks[task_id]
    
    async def stop_rtsp_analysis(self, task_id: str):
        """停止RTSP流分析"""
        if task_id not in self.rtsp_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task = self.rtsp_tasks[task_id]
        
        try:
            if task.status in ["processing", "offline"]:
                # 停止检测器
                from api.services.analysis import AnalysisService
                analysis_service = AnalysisService()
                if task_id in analysis_service.rtsp_detectors:
                    detector = analysis_service.rtsp_detectors[task_id]
                    detector.stop()
                    del analysis_service.rtsp_detectors[task_id]
                
                # 关闭代理
                await analysis_service.rtsp_proxy.close_proxy(task_id)
                
                task.status = "stopped"
                task.stopped_at = datetime.now()
                
                # 发送停止通知
                await self.callback_service.send_callback(task_id, {
                    "status": "stopped",
                    "stopped_at": task.stopped_at.isoformat()
                })
                
                return {"message": "分析已停止"}
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"任务当前状态({task.status})不允许停止"
                )
                
        except Exception as e:
            print(f"停止任务失败: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"停止任务失败: {str(e)}"
            )
    
    async def force_delete_rtsp_task(self, task_id: str):
        """强制删除RTSP任务"""
        if task_id not in self.rtsp_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task = self.rtsp_tasks[task_id]
        
        try:
            # 如果任务正在运行,先停止它
            if task.status in ["processing", "offline"]:
                from api.services.analysis import AnalysisService
                analysis_service = AnalysisService()
                if task_id in analysis_service.rtsp_detectors:
                    detector = analysis_service.rtsp_detectors[task_id]
                    detector.stop()
                    del analysis_service.rtsp_detectors[task_id]
                
                # 关闭代理
                await analysis_service.rtsp_proxy.close_proxy(task_id)
            
            # 从任务列表中删除
            del self.rtsp_tasks[task_id]
            
            return {
                "message": "任务已强制删除",
                "task_id": task_id,
                "rtsp_url": task.rtsp_url
            }
            
        except Exception as e:
            print(f"强制删除任务失败: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"强制删除任务失败: {str(e)}"
            )
    
    async def cleanup_rtsp_tasks(self, status: Optional[str], force: bool):
        """清理RTSP任务"""
        print(f"\n接收到清理请求: status={status}, force={force}")
        
        # 获取清理前的统计信息
        before_stats = {
            "total": len(self.rtsp_tasks),
            "by_status": {}
        }
        for task in self.rtsp_tasks.values():
            before_stats["by_status"][task.status] = before_stats["by_status"].get(task.status, 0) + 1
        
        print(f"\n开始清理任务:")
        print(f"当前任务总数: {before_stats['total']}")
        print(f"按状态统计: {before_stats['by_status']}")
        print(f"清理参数: status={status}, force={force}")
        
        # 准备清理结果
        cleanup_results = {
            "success": [],
            "skipped": [],
            "failed": []
        }
        
        # 如果没有任务,直接返回成功
        if not self.rtsp_tasks:
            print("没有需要清理的任务")
            return {
                "status": "completed",
                "message": "没有需要清理的任务",
                "filter": {
                    "status": status,
                    "force": force
                },
                "stats": {
                    "before": before_stats,
                    "after": {
                        "total": 0,
                        "by_status": {}
                    },
                    "cleaned": 0,
                    "skipped": 0,
                    "failed": 0
                },
                "results": cleanup_results
            }
        
        # 获取要清理的任务ID列表
        tasks_to_cleanup = []
        for task_id, task in self.rtsp_tasks.items():
            print(f"检查任务 {task_id}: 状态={task.status}")
            if status is None or task.status == status:
                tasks_to_cleanup.append(task_id)
                print(f"- 任务 {task_id} 将被清理")
        
        # 如果没有匹配的任务,返回成功但说明没有匹配任务
        if not tasks_to_cleanup:
            print(f"没有匹配状态 {status} 的任务")
            return {
                "status": "completed",
                "message": f"没有匹配状态 {status} 的任务",
                "filter": {
                    "status": status,
                    "force": force
                },
                "stats": {
                    "before": before_stats,
                    "after": before_stats,
                    "cleaned": 0,
                    "skipped": 0,
                    "failed": 0
                },
                "results": cleanup_results
            }
        
        print(f"\n待清理任务数: {len(tasks_to_cleanup)}")
        
        # 执行清理
        from api.services.analysis import AnalysisService
        analysis_service = AnalysisService()
        
        for task_id in tasks_to_cleanup:
            task = self.rtsp_tasks[task_id]
            try:
                print(f"\n处理任务 {task_id}:")
                print(f"- 状态: {task.status}")
                print(f"- URL: {task.rtsp_url}")
                
                # 如果任务正在运行且不是强制清理,则跳过
                if not force and task.status in ["processing", "pending", "offline"]:
                    print(f"- 跳过: 任务正在运行且非强制清理")
                    cleanup_results["skipped"].append({
                        "task_id": task_id,
                        "status": task.status,
                        "reason": "Task is running and force=false"
                    })
                    continue
                
                # 如果任务正在运行,先停止检测器
                if task.status in ["processing", "offline"] and task_id in analysis_service.rtsp_detectors:
                    print(f"- 停止检测器")
                    detector = analysis_service.rtsp_detectors[task_id]
                    detector.stop()
                    del analysis_service.rtsp_detectors[task_id]
                    
                    # 关闭代理
                    await analysis_service.rtsp_proxy.close_proxy(task_id)
                
                # 从任务列表中删除
                print(f"- 删除任务")
                del self.rtsp_tasks[task_id]
                
                cleanup_results["success"].append({
                    "task_id": task_id,
                    "status": task.status,
                    "rtsp_url": task.rtsp_url
                })
                print(f"- 清理成功")
                
            except Exception as e:
                print(f"- 清理失败: {str(e)}")
                cleanup_results["failed"].append({
                    "task_id": task_id,
                    "status": task.status,
                    "error": str(e)
                })
        
        # 获取清理后的统计信息
        after_stats = {
            "total": len(self.rtsp_tasks),
            "by_status": {}
        }
        for task in self.rtsp_tasks.values():
            after_stats["by_status"][task.status] = after_stats["by_status"].get(task.status, 0) + 1
        
        print(f"\n清理完成:")
        print(f"- 成功: {len(cleanup_results['success'])}")
        print(f"- 跳过: {len(cleanup_results['skipped'])}")
        print(f"- 失败: {len(cleanup_results['failed'])}")
        print(f"剩余任务数: {after_stats['total']}")
        
        return {
            "status": "completed",
            "message": "清理完成",
            "filter": {
                "status": status,
                "force": force
            },
            "stats": {
                "before": before_stats,
                "after": after_stats,
                "cleaned": len(cleanup_results["success"]),
                "skipped": len(cleanup_results["skipped"]),
                "failed": len(cleanup_results["failed"])
            },
            "results": cleanup_results
        }
    
    async def get_rtsp_tasks_stats(self):
        """获取RTSP任务统计"""
        # 获取检测器数量
        from api.services.analysis import AnalysisService
        analysis_service = AnalysisService()
        active_detectors = len(analysis_service.rtsp_detectors)
        
        # 基础统计信息
        stats = {
            "total": len(self.rtsp_tasks),
            "by_status": {},
            "active_detectors": active_detectors,
            "details": {
                "processing": [],
                "offline": [],
                "stopped": [],
                "failed": []
            }
        }
        
        # 按状态统计任务数量并收集详细信息
        for task_id, task in self.rtsp_tasks.items():
            # 更新状态计数
            stats["by_status"][task.status] = stats["by_status"].get(task.status, 0) + 1
            
            # 收集任务详细信息
            task_info = {
                "task_id": task_id,
                "rtsp_url": task.rtsp_url,
                "created_at": task.created_at.isoformat(),
                "stream_url": task.stream_url
            }
            
            # 根据状态添加特定信息
            if task.status == "processing":
                if task_id in analysis_service.rtsp_detectors:
                    detector = analysis_service.rtsp_detectors[task_id]
                    task_info.update({
                        "fps": detector.get_fps(),
                        "frame_count": detector.get_frame_count(),
                        "running_time": (datetime.now() - task.created_at).total_seconds()
                    })
                stats["details"]["processing"].append(task_info)
                
            elif task.status == "offline":
                task_info.update({
                    "reconnect_count": task.reconnect_count,
                    "last_reconnect": task.last_reconnect.isoformat() if task.last_reconnect else None,
                    "error": task.error
                })
                stats["details"]["offline"].append(task_info)
                
            elif task.status == "stopped":
                task_info.update({
                    "stopped_at": task.stopped_at.isoformat() if task.stopped_at else None,
                    "running_time": (task.stopped_at - task.created_at).total_seconds() if task.stopped_at else None
                })
                stats["details"]["stopped"].append(task_info)
                
            elif task.status == "failed":
                task_info.update({
                    "error": task.error,
                    "failed_at": task.stopped_at.isoformat() if task.stopped_at else None
                })
                stats["details"]["failed"].append(task_info)
        
        # 添加系统资源使用情况
        try:
            import psutil
            stats["system"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        except ImportError:
            print("psutil not installed, skipping system stats")
        
        # 添加时间信息
        stats["timestamp"] = datetime.now().isoformat()
        
        return stats 