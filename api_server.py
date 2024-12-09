from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import base64
import cv2
import numpy as np
import uuid
import threading
import requests
from yolo_rtsp import MeekYolo
import os
from datetime import datetime
import aiohttp
import asyncio
import yaml
import zipfile
import shutil
from packaging import version
import json
import sys

# 在文件开头添加 API 标签定义
tags_metadata = [
    {
        "name": "analysis",
        "description": "分析相关接口，包括图片、视频、RTSP流分析"
    },
    {
        "name": "tasks",
        "description": "任务管理接口，用于查询、删除和管理分析任务"
    },
    {
        "name": "models",
        "description": "模型管理接口，用于上传、切换和管理检测模型"
    },
    {
        "name": "config",
        "description": "配置管理接口，用于设置回调等系统配置"
    },
    {
        "name": "system",
        "description": "系统���口，如健康检查等"
    }
]

# 首先定义所有的模型类
class ImageRequest(BaseModel):
    """图片分析请求模型"""
    images: List[str]  # 图片URL列表或base64字符串列表
    is_base64: bool = False  # 是否为base64格式

class VideoRequest(BaseModel):
    """视频分析请求模型"""
    video_url: str

class RtspRequest(BaseModel):
    """RTSP分析请求模型"""
    rtsp_url: str
    output_rtmp: Optional[str] = None  # 可选的RTMP推流地址
    callback_interval: float = 1.0  # 回调间隔（秒），默认1秒

class AnalysisResult(BaseModel):
    """分析结果模型"""
    image_base64: Optional[str] = None  # 处理后的图片base64
    detections: List[dict]  # 检测结果
    stream_url: Optional[str] = None  # RTSP推流地址

class CallbackConfig(BaseModel):
    """回调API配置"""
    url: str

class VideoAnalysisTask(BaseModel):
    """视频分析任务状态"""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    result_path: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class RtspAnalysisTask(BaseModel):
    """RTSP分析任务状态"""
    task_id: str
    status: str  # pending, processing, stopped, failed, offline
    error: Optional[str] = None
    rtsp_url: str
    stream_url: Optional[str] = None
    created_at: datetime
    stopped_at: Optional[datetime] = None
    reconnect_count: int = 0
    last_reconnect: Optional[datetime] = None
    callback_interval: float = 1.0  # 添加回调间隔字段
    last_callback: Optional[datetime] = None  # 添加上次回调时间字段

# 添加模型相关的模型类
class ModelInfo(BaseModel):
    """模型信息"""
    code: str                # 模型编码
    version: str            # 模型版本
    name: str              # 模型名称
    description: str       # 模型描述
    author: str           # 作者
    create_time: datetime  # 创建时间
    update_time: datetime  # 更新时间
    path: str             # 模型路径

class ModelList(BaseModel):
    """模型列表"""
    models: List[ModelInfo]
    current_model: str  # 当前使用的模型编码

# 模理类
class ModelManager:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config['model_management']
        self.model_dir = self.model_config['model_dir']
        self.temp_dir = self.model_config['temp_dir']
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def get_model_list(self) -> ModelList:
        """获取模型列表"""
        models = []
        for model_dir in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
                
            config_path = os.path.join(model_path, 'data.yaml')
            if not os.path.exists(config_path):
                continue
                
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = yaml.safe_load(f)
                    if not self._validate_model_config(model_config):
                        continue
                    
                    models.append(ModelInfo(
                        code=model_config['code'],
                        version=model_config['version'],
                        name=model_config['name'],
                        description=model_config['description'],
                        author=model_config['author'],
                        create_time=datetime.fromisoformat(model_config['create_time']),
                        update_time=datetime.fromisoformat(model_config['update_time']),
                        path=model_path
                    ))
            except Exception as e:
                print(f"读取模型配置失败: {str(e)}")
                continue
        
        return ModelList(
            models=models,
            current_model=self.config['model']['path'].split('/')[-2]  # 获取当前模型编码
        )
    
    def _validate_model_config(self, config: dict) -> bool:
        """验证模型配置是否整"""
        required_fields = [
            'code', 'version', 'name', 'description',
            'author', 'create_time', 'update_time'
        ]
        
        # 检查必需字段是否存在
        if not all(field in config for field in required_fields):
            print(f"模型配置缺少必字段，需要: {required_fields}")
            print(f"当前配置: {list(config.keys())}")
            return False
        
        # 检查字段值是否有效
        if not config['code'] or not config['version'] or not config['name']:
            print("模型配置中的必字段值不能为空")
            return False
        
        return True
    
    async def upload_model(self, file: UploadFile) -> ModelInfo:
        """上传模型"""
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="只支持zip格式的模型件")
        
        # 保存上传的文件
        temp_zip = os.path.join(self.temp_dir, file.filename)
        temp_extract = os.path.join(self.temp_dir, 'extract')
        
        try:
            # 确保临时目录存在
            os.makedirs(self.temp_dir, exist_ok=True)
            
            print(f"\n开始处理模型上传:")
            print(f"文件名: {file.filename}")
            print(f"临时压缩包路径: {temp_zip}")
            print(f"临时解压目录: {temp_extract}")
            
            # 保存上传的文件
            with open(temp_zip, 'wb') as f:
                content = await file.read()
                f.write(content)
            print(f"文件保存到: {temp_zip}")
            
            # 清理并创建解压目录
            if os.path.exists(temp_extract):
                shutil.rmtree(temp_extract)
            os.makedirs(temp_extract)
            
            # 解压文件
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"\n压缩包内文件列表:")
                for f in file_list:
                    print(f"  - {f}")
                
                # 检查是否有子目录
                root_files = [f for f in file_list if '/' not in f]
                if not root_files:
                    print("\n警告: 所有文件都在子目录中")
                    # 获第一级目录
                    top_dir = file_list[0].split('/')[0]
                    print(f"使用顶级目录: {top_dir}")
                
                zip_ref.extractall(temp_extract)
                print(f"\n文件已解压到: {temp_extract}")
                
                # 列出解压后的文件
                print("\n解压目录内容:")
                for root, dirs, files in os.walk(temp_extract):
                    level = root.replace(temp_extract, '').count(os.sep)
                    indent = ' ' * 4 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for f in files:
                        print(f"{subindent}{f}")
            
            # 确定配置文件和模型文件的路径
            if root_files:
                config_path = os.path.join(temp_extract, 'data.yaml')
                model_path = os.path.join(temp_extract, 'best.pt')
            else:
                config_path = os.path.join(temp_extract, top_dir, 'data.yaml')
                model_path = os.path.join(temp_extract, top_dir, 'best.pt')
            
            print(f"\n检查必需文:")
            print(f"配置文件路径: {config_path} (存在: {os.path.exists(config_path)})")
            print(f"模型文件路径: {model_path} (存在: {os.path.exists(model_path)})")
            
            if not (os.path.exists(config_path) and os.path.exists(model_path)):
                raise HTTPException(status_code=400, detail="模型文件不完整")
            
            # 读取配置
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
                print(f"\n模型配置内容:")
                print(yaml.dump(model_config, allow_unicode=True))
                
                if not self._validate_model_config(model_config):
                    raise HTTPException(status_code=400, detail="模型配置不完整")
            
            # 检查是否存在相同编码的模型
            model_code = model_config['code']
            target_dir = os.path.join(self.model_dir, model_code)
            
            print(f"\n目标目录: {target_dir}")
            if os.path.exists(target_dir):
                print("发现已存在的模型，准备更新...")
                shutil.rmtree(target_dir)
            
            # 移动文件到目标目录
            if root_files:
                # 直接���动解压目
                shutil.move(temp_extract, target_dir)
            else:
                # 移动子目录内
                source_dir = os.path.join(temp_extract, top_dir)
                shutil.move(source_dir, target_dir)
            
            print(f"文件已移动到目标目录")
            
            # 返回模型信息
            return ModelInfo(
                code=model_config['code'],
                version=model_config['version'],
                name=model_config['name'],
                description=model_config['description'],
                author=model_config['author'],
                create_time=datetime.fromisoformat(model_config['create_time']),
                update_time=datetime.fromisoformat(model_config['update_time']),
                path=target_dir
            )
            
        except Exception as e:
            print(f"\n处理失败: {str(e)}")
            raise
            
        finally:
            # 清理临时文件
            print("\n清理临时文件...")
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            if os.path.exists(temp_extract):
                shutil.rmtree(temp_extract)
    
    def set_current_model(self, model_code: str):
        """设置当前使用的模型"""
        # 验证模型是否存
        model_path = os.path.join(self.model_dir, model_code, 'best.pt')
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 更新配置
        self.config['model']['path'] = model_path
        
        # 保存配置
        with open('config/config.yaml', 'w') as f:
            yaml.dump(self.config, f)

# 然后创建FastAPI实例
app = FastAPI(
    title="YOLO Detection API",
    description="""
    YOLO目标检测API服务
    
    ## 功能特点
    * 支持图片、视频、RTSP流分析
    * 支持多任务并行理
    * 实时回调推送
    * 自动重连机
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 最后定义全局变量
rtsp_tasks: Dict[str, RtspAnalysisTask] = {}
video_tasks: Dict[str, VideoAnalysisTask] = {}
callback_config = {
    'enabled': False,
    'url': None,
    'retry': {
        'max_retries': 10,
        'retry_delay': 1.0,
        'timeout': 10
    }
}

# 配置文件载回调设置
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    if config.get('callback', {}).get('enabled', False):
        callback_config['enabled'] = True
        callback_config['url'] = config['callback']['url']
        # 加载重试配置
        if 'retry' in config['callback']:
            callback_config['retry'].update(config['callback']['retry'])

# 添加一个字典来单独存储检测器实例
rtsp_detectors: Dict[str, MeekYolo] = {}

# 创建模型管理器实例
model_manager = ModelManager()

# 在其他类定义之后，添加 RtspReconnectManager 类
class RtspReconnectManager:
    def __init__(self):
        self.reconnect_interval = 60  # 重连间隔（秒）
        self.running = True
        self.reconnect_task = None
    
    async def start_reconnect_monitor(self):
        """启动连监控"""
        if self.reconnect_task is None:
            self.running = True
            self.reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def stop_reconnect_monitor(self):
        """停止监控"""
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
                    task_id for task_id, task in rtsp_tasks.items()
                    if task.status == "offline"
                ]
                
                for task_id in offline_tasks:
                    task = rtsp_tasks[task_id]
                    # 检查是否到达重连时间
                    if (task.last_reconnect is None or 
                        (datetime.now() - task.last_reconnect).total_seconds() >= self.reconnect_interval):
                        await self._try_reconnect(task_id)
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                print(f"重连监控常: {str(e)}")
                await asyncio.sleep(10)
    
    async def _try_reconnect(self, task_id: str):
        """尝试重连"""
        task = rtsp_tasks[task_id]
        task.last_reconnect = datetime.now()
        task.reconnect_count += 1
        
        # 计算下次重试延迟（指数退避）
        next_delay = min(
            task.reconnect_delay * (task.retry_backoff ** (task.reconnect_count - 1)),
            task.max_retry_interval
        )
        task.reconnect_delay = next_delay
        
        try:
            # 创建临时检测器测试连接
            cap = cv2.VideoCapture(task.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TRANSPORT_TCP)
            
            if not cap.isOpened():
                print(f"重连失败 - 任务 {task_id}, 尝试次数: {task.reconnect_count}, 下次重试延迟: {next_delay}秒")
                cap.release()
                return
            
            # 读取一帧测试连接
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"重连失败（无法读取帧）- 任务 {task_id}, 尝试次数: {task.reconnect_count}")
                return
            
            print(f"重连成功 - 任务 {task_id}, 尝试数: {task.reconnect_count}")
            # 重新启动任务
            task.status = "pending"
            asyncio.create_task(process_rtsp_task(
                task_id,
                task.rtsp_url,
                task.stream_url
            ))
            
        except Exception as e:
            print(f"重连异常 - 任务 {task_id}: {str(e)}")

# 在全局变量定义部分添加
rtsp_detectors: Dict[str, MeekYolo] = {}
reconnect_manager = RtspReconnectManager()  # 创建重连管理器实例

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

def decode_image(image_data: str, is_base64: bool) -> np.ndarray:
    """解码图片数据"""
    try:
        if is_base64:
            # 解码base64
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            # 从URL读取图片
            resp = requests.get(image_data, verify=False)
            img_array = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解码失败: {str(e)}")

def encode_image(image: np.ndarray) -> str:
    """编码图片base64"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def format_detections(results) -> List[dict]:
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

async def process_image_file(file: UploadFile, detector: MeekYolo) -> AnalysisResult:
    """处理上传的图片文件"""
    try:
        # 检查文件大小
        file_size = 0
        contents = bytearray()
        
        # 分块读取文件，避免内存问题
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            file_size += len(chunk)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="文件大小超过限制(最大10MB)")
            contents.extend(chunk)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="文件为空")
        
        # 验证文件类型
        allowed_types = {'image/jpeg', 'image/png', 'image/jpg'}
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file.content_type}. 支持的类型: {', '.join(allowed_types)}"
            )
        
        # 解码片
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="无法解码图片文件，请确保文件格式正确")
        
        # 处理图片
        detections = detector.process_frame(frame)
        processed_frame = detector.draw_results(frame, detections)
        result_base64 = encode_image(processed_frame)
        formatted_detections = format_detections(detections)
        
        return AnalysisResult(
            image_base64=result_base64,
            detections=formatted_detections
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"处理图片失败: {str(e)}")  # 添加日志输出
        raise HTTPException(status_code=400, detail=f"处理图片失败: {str(e)}")

@app.post("/analyze/image", response_model=List[AnalysisResult])
async def analyze_image(request: ImageRequest):
    """分析单张或多张图片 (通过URL或base64)"""
    try:
        detector = MeekYolo()
        results = []
        
        for image_data in request.images:
            frame = decode_image(image_data, request.is_base64)
            detections = detector.process_frame(frame)
            processed_frame = detector.draw_results(frame, detections)
            result_base64 = encode_image(processed_frame)
            formatted_detections = format_detections(detections)
            
            results.append(AnalysisResult(
                image_base64=result_base64,
                detections=formatted_detections
            ))
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/image/upload", tags=["analysis"], response_model=List[AnalysisResult])
async def analyze_uploaded_images(
    files: List[UploadFile] = File(
        ...,
        description="要分析的图片文件（支持多选）",
        media_type="image/*"
    )
):
    """
    分析上传的图片文件
    
    - **files**: 要分析的图片文件列表（支持多选）
    """
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
                
                result = await process_image_file(file, detector)
                results.append(result)
                
            except Exception as e:
                print(f"处理文件 {file.filename} 失败: {str(e)}")  # 添加日志输出
                raise HTTPException(status_code=400, detail=f"处理文件 {file.filename} 失败: {str(e)}")
        
        return results
        
    except Exception as e:
        print(f"分析上传文件失败: {str(e)}")  # 添加日志输出
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/video", tags=["analysis"], response_model=VideoAnalysisTask)
async def analyze_video(request: VideoRequest):
    """
    提交视频分析任务
    
    - **video_url**: 视频文件URL
    """
    task_id = str(uuid.uuid4())
    
    # 创建任务记录
    video_tasks[task_id] = VideoAnalysisTask(
        task_id=task_id,
        status="pending",
        progress=0,
        created_at=datetime.now()
    )
    
    # 启异步处理
    asyncio.create_task(process_video_task(task_id, request.video_url))
    
    return video_tasks[task_id]

@app.post("/analyze/rtsp", tags=["analysis"])
async def analyze_rtsp(request: RtspRequest):
    """启动RTSP流分析"""
    # 检查是否存在相同的RTSP地址
    existing_task = None
    for task in rtsp_tasks.values():
        if task.rtsp_url == request.rtsp_url:
            existing_task = task
            break
    
    if existing_task:
        # 如果任务存在，根据状态返回不同的错误信息
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
            # 如果任务状态是 stopped 或 failed，可以选择重新启动或返回错误
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"该RTSP地址已存在任务（状态：{existing_task.status}），请先删除原任务",
                    "task_id": existing_task.task_id,
                    "status": existing_task.status,
                    "stopped_at": existing_task.stopped_at.isoformat() if existing_task.stopped_at else None
                }
            )
    
    # 如果不存在重复任务，创建新任务
    task_id = str(uuid.uuid4())
    output_rtmp = request.output_rtmp or f"rtmp://your_server/live/{task_id}"
    
    # 创建任务记录
    rtsp_tasks[task_id] = RtspAnalysisTask(
        task_id=task_id,
        status="pending",
        rtsp_url=request.rtsp_url,
        stream_url=output_rtmp,
        created_at=datetime.now(),
        callback_interval=request.callback_interval
    )
    
    # 启动异步处理
    asyncio.create_task(process_rtsp_task(
        task_id,
        request.rtsp_url,
        output_rtmp,
        request.callback_interval
    ))
    
    return rtsp_tasks[task_id]

@app.get("/analyze/rtsp/{task_id}", tags=["analysis"])
async def get_rtsp_task(task_id: str):
    """获取RTSP分析任务状态"""
    if task_id not in rtsp_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return rtsp_tasks[task_id]

@app.delete("/analyze/rtsp/{task_id}", tags=["analysis"])
async def stop_rtsp_analysis(task_id: str):
    """停止RTSP流分析"""
    if task_id not in rtsp_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = rtsp_tasks[task_id]
    detector = rtsp_detectors.get(task_id)
    
    if task.status in ["processing", "offline"]:
        # 停止检测器
        if detector:
            detector.stop()
            del rtsp_detectors[task_id]
        
        task.status = "stopped"
        task.stopped_at = datetime.now()
        
        # 发送停止通知
        await send_callback(task_id, {
            "status": "stopped",
            "stopped_at": task.stopped_at.isoformat()
        })
    
    return {"message": "分析已停止"}

@app.get("/health", tags=["system"])
async def health_check():
    """
    健康检查接口
    
    返回系统运行状态
    """
    return {"status": "ok"}

@app.get("/analyze/video/{task_id}", tags=["analysis"], response_model=VideoAnalysisTask)
async def get_video_task(task_id: str):
    """
    获取视频分析任务状态
    
    - **task_id**: 任务ID
    """
    if task_id not in video_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return video_tasks[task_id]

@app.get("/download/video/{task_id}", tags=["analysis"])
async def download_video_result(task_id: str):
    """
    下载频分析结果
    
    - **task_id**: 任务ID
    """
    if task_id not in video_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = video_tasks[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    if not task.result_path or not os.path.exists(task.result_path):
        raise HTTPException(status_code=404, detail="结果文件不存在")
    
    return FileResponse(
        task.result_path,
        media_type="video/mp4",
        filename=f"result_{task_id}.mp4"
    )

# 修改原有的视频分析相关代码
@app.post("/analyze/video/sync", tags=["analysis"], response_model=List[AnalysisResult])
async def analyze_video_sync(request: VideoRequest):
    """
    同步分析视频（不推荐用于大文件）
    
    - **video_url**: 视频文件URL
    
    注意：此接口会同步返回结果，不适合处理大文件
    """
    detector = MeekYolo()
    results = []
    
    cap = cv2.VideoCapture(request.video_url)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="无法打开视频文件")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = detector.process_frame(frame)
            formatted_detections = format_detections(detections)
            results.append(AnalysisResult(detections=formatted_detections))
    
    finally:
        cap.release()
    
    return results

# 修改回调相关的函数
async def send_callback_with_retry(task_id: str, data: dict):
    """带重试的回调发送"""
    if not callback_config['enabled'] or not callback_config['url']:
        return
    
    retry_config = callback_config['retry']
    max_retries = retry_config['max_retries']
    retry_delay = retry_config['retry_delay']
    timeout = retry_config['timeout']
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    callback_config['url'], 
                    json={"task_id": task_id, **data},
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        return True
                    
            # 如果状态码不是200，等待后试
            retry_count += 1
            if retry_count < max_retries:
                print(f"回调发送失败(HTTP {response.status})，{retry_count}/{max_retries} 次重试中...")
                await asyncio.sleep(retry_delay)
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"回调发送失败({str(e)})，{retry_count}/{max_retries} 次重试中...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"回调发送最终失败: {str(e)}")
                return False
    
    print(f"回调发送失败，已达到最大重试次数({max_retries})")
    return False

async def send_callback(task_id: str, data: dict):
    """发送回调通知（不阻塞主流程）"""
    if not callback_config['enabled'] or not callback_config['url']:
        return
    
    # 创建新的任务处理回调发送（包括重试）
    asyncio.create_task(send_callback_with_retry(task_id, data))

async def process_video_task(task_id: str, video_url: str):
    """异步处理视频分析任务"""
    try:
        # 更新任务状态
        video_tasks[task_id].status = "processing"
        await send_callback(task_id, {
            "status": "processing",
            "progress": 0
        })
        
        # 创建保存目录
        save_dir = os.path.join("results", "videos", task_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置存路径
        result_path = os.path.join(save_dir, "result.mp4")
        
        # 创建检测器实例
        detector = MeekYolo()
        
        # 定义进度回调函数
        def progress_callback(progress: float):
            video_tasks[task_id].progress = progress
            asyncio.create_task(send_callback(task_id, {
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
        video_tasks[task_id].status = "completed"
        video_tasks[task_id].completed_at = datetime.now()
        video_tasks[task_id].result_path = result_path
        video_tasks[task_id].download_url = f"/download/video/{task_id}"
        
        # 发送完成通知
        await send_callback(task_id, {
            "status": "completed",
            "progress": 1.0,
            "result_path": result_path,
            "download_url": f"/download/video/{task_id}"
        })
        
    except Exception as e:
        # 更新失败状态
        video_tasks[task_id].status = "failed"
        video_tasks[task_id].error = str(e)
        video_tasks[task_id].completed_at = datetime.now()
        
        # 发送失败通知
        await send_callback(task_id, {
            "status": "failed",
            "error": str(e)
        })

# 添加回调配置相关的API接口
@app.post("/config/callback", tags=["config"])
async def set_callback_url(config: CallbackConfig):
    """设置回调API地址"""
    global callback_config
    callback_config['enabled'] = True
    callback_config['url'] = config.url
    return {
        "message": "回调地址设置成功", 
        "callback_url": config.url,
        "retry_config": callback_config['retry']
    }

@app.get("/config/callback", tags=["config"])
async def get_callback_url():
    """
    获取当前回调配置
    
    返回：
    - enabled: 是否启用回调
    - callback_url: 回调地址
    - retry_config: 重试配置
    """
    return {
        "enabled": callback_config['enabled'],
        "callback_url": callback_config['url'],
        "retry_config": callback_config['retry']
    }

@app.delete("/config/callback", tags=["config"])
async def delete_callback_url():
    """
    删除回调API地址
    """
    global callback_config
    callback_config['enabled'] = False
    callback_config['url'] = None
    return {"message": "回调地址已删除"}

async def process_rtsp_task(task_id: str, rtsp_url: str, output_rtmp: Optional[str] = None, callback_interval: float = 1.0):
    """异步处理 RTSP 流分析任务"""
    try:
        # 更新任务状态
        task = rtsp_tasks[task_id]
        print(f"\n开始处理RTSP任务 {task_id}:")
        print(f"- 原始状态: {task.status}")
        task.status = "processing"
        print(f"- 更新状态为: {task.status}")
        
        # 构建 RTSP URL 参数
        rtsp_params = {
            'tcp': {
                'url': f"{rtsp_url}?rtsp_transport=tcp&buffer_size=1024000&max_delay=500000&stimeout=20000000&reorder_queue_size=0",
                'tried': False,
                'success': False
            },
            'udp': {
                'url': f"{rtsp_url}?rtsp_transport=udp&buffer_size=1024000&max_delay=500000&stimeout=20000000",
                'tried': False,
                'success': False
            }
        }
        
        # 尝试建立连接
        max_retries = 3
        retry_count = 0
        success = False
        working_url = None
        
        while retry_count < max_retries and not success:
            # 首先尝试 TCP
            if not rtsp_params['tcp']['tried']:
                print(f"\n尝试 TCP 连接 (重试 {retry_count + 1}/{max_retries}):")
                print(f"URL: {rtsp_params['tcp']['url']}")
                
                cap = cv2.VideoCapture(rtsp_params['tcp']['url'], cv2.CAP_FFMPEG)
                rtsp_params['tcp']['tried'] = True
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print("TCP 连接成功")
                        rtsp_params['tcp']['success'] = True
                        success = True
                        working_url = rtsp_params['tcp']['url']
                    else:
                        print("TCP 连接成功无法读取帧")
                else:
                    print("TCP 连接失败")
                cap.release()
            
            # 如果 TCP 失败，尝试 UDP
            if not success and not rtsp_params['udp']['tried']:
                print(f"\n尝试 UDP 连接 (重试 {retry_count + 1}/{max_retries}):")
                print(f"URL: {rtsp_params['udp']['url']}")
                
                cap = cv2.VideoCapture(rtsp_params['udp']['url'], cv2.CAP_FFMPEG)
                rtsp_params['udp']['tried'] = True
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print("UDP 连接成功")
                        rtsp_params['udp']['success'] = True
                        success = True
                        working_url = rtsp_params['udp']['url']
                    else:
                        print("UDP 连接成功但无法读取帧")
                else:
                    print("UDP 连接失败")
                cap.release()
            
            # 如果两种方式都尝试过但都失败了
            if rtsp_params['tcp']['tried'] and rtsp_params['udp']['tried'] and not success:
                retry_count += 1
                # 重置尝试状态，准备下一轮
                rtsp_params['tcp']['tried'] = False
                rtsp_params['udp']['tried'] = False
                if retry_count < max_retries:
                    print(f"\n等待 2 秒后进行第 {retry_count + 1} 次重试...")
                    await asyncio.sleep(2)
        
        if not success:
            raise Exception("无法建立稳定的RTSP连接，请检查URL或网络状态")
        
        # 创建检测器实例
        detector = MeekYolo()
        rtsp_detectors[task_id] = detector
        
        # 配置检测器
        detector.config['environment']['enable_gui'] = False
        detector.config['display']['show_window'] = False
        detector.config['source']['type'] = 'rtsp'
        detector.config['source']['rtsp']['url'] = working_url  # 使用成功连接的URL
        
        if output_rtmp:
            detector.config['output'] = {
                'enabled': True,
                'rtmp_url': output_rtmp
            }
        
        # 设置任务信息
        detector.set_task_info(
            task_id,
            callback_func=send_callback,
            callback_interval=callback_interval
        )
        
        # 发送开始通知
        await send_callback(task_id, {
            "status": "processing",
            "rtsp_url": rtsp_url,
            "stream_url": output_rtmp,
            "connection_type": "tcp" if rtsp_params['tcp']['success'] else "udp"
        })
        
        # 启动重连监控
        await reconnect_manager.start_reconnect_monitor()
        
        # 运行检测器
        await detector.run()
        
    except Exception as e:
        print(f"\nRTSP 任务异常: {str(e)}")
        # 更新失败状态
        task = rtsp_tasks[task_id]
        print(f"- 原始状态: {task.status}")
        task.status = "failed"
        print(f"- 更新状态为: {task.status}")
        task.error = str(e)
        task.stopped_at = datetime.now()
        
        # 清理检测
        if task_id in rtsp_detectors:
            del rtsp_detectors[task_id]
        
        # 发送失败通知
        await send_callback(task_id, {
            "status": "failed",
            "error": str(e)
        })

# 添加模型管理关的API接口
@app.get("/models", tags=["models"])
async def get_models():
    """获取模型列表"""
    return model_manager.get_model_list()

@app.post("/models/upload", tags=["models"], response_model=ModelInfo)
async def upload_model(file: UploadFile = File(...)):
    """
    上传模型
    
    - **file**: 模型文件（zip格式）
    
    模型文件须包含：
    - best.pt: 模型文件
    - data.yaml: 配置文件
    """
    return await model_manager.upload_model(file)

@app.post("/models/{model_code}/set", tags=["models"])
async def set_model(model_code: str):
    """
    设置当前使用的模型
    
    - **model_code**: 模型编码
    """
    model_manager.set_current_model(model_code)
    return {"message": "模型设置成功"}

# 在应用启动时启动重连监控
@app.on_event("startup")
async def startup_event():
    await reconnect_manager.start_reconnect_monitor()

# 在应用关闭时停止重连监控
@app.on_event("shutdown")
async def shutdown_event():
    await reconnect_manager.stop_reconnect_monitor()

# 添加删除任务的接口
@app.delete("/analyze/rtsp/{task_id}/force")
async def force_delete_rtsp_task(task_id: str):
    """强制删除RTSP分析任务"""
    if task_id not in rtsp_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = rtsp_tasks[task_id]
    
    # 如果任务正在运行，先停止它
    if task.status in ["processing", "offline"] and task_id in rtsp_detectors:
        detector = rtsp_detectors[task_id]
        detector.stop()
        del rtsp_detectors[task_id]
    
    # 从任务列表中删除
    del rtsp_tasks[task_id]
    
    return {
        "message": "任务已强删除",
        "task_id": task_id,
        "rtsp_url": task.rtsp_url
    }

# 添加任务管理相关的接口
@app.get("/tasks/rtsp", tags=["tasks"])
async def list_rtsp_tasks(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 10
):
    """
    获取RTSP任务列表
    
    - **status**: 可选的状态过滤（processing, stopped, failed, offline）
    - **skip**: 跳过的记录数
    - **limit**: 返回的最大记录数
    """
    tasks = []
    for task in rtsp_tasks.values():
        if status and task.status != status:
            continue
        tasks.append(task)
    
    # 排序：先按状态（运行中的在前），再按创建时间（新的在前）
    tasks.sort(key=lambda x: (
        0 if x.status in ["processing", "pending", "offline"] else 1,
        x.created_at
    ), reverse=True)
    
    # 分页
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

@app.delete("/tasks/rtsp/cleanup", tags=["tasks"])
async def cleanup_rtsp_tasks(
    status: Optional[str] = None,
    force: bool = False
):
    """
    清理RTSP分析任务
    
    - **status**: 可选的状态过滤（processing, stopped, failed, offline）
    - **force**: 是否强制清理运行中的任务
    """
    print(f"\n接收到清理请求: status={status}, force={force}")
    
    # 获取清理前的统计信息
    before_stats = {
        "total": len(rtsp_tasks),
        "by_status": {}
    }
    for task in rtsp_tasks.values():
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
    if not rtsp_tasks:
        print("没有需要清理的任务")
        return JSONResponse(
            status_code=200,
            content={
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
        )
    
    # 获取要清理的任务ID列表
    tasks_to_cleanup = []
    for task_id, task in rtsp_tasks.items():
        print(f"检查任务 {task_id}: 状态={task.status}")
        if status is None or task.status == status:
            tasks_to_cleanup.append(task_id)
            print(f"- 任务 {task_id} 将被清理")
    
    # 如果没有匹配的任务,返回成功但说明没有匹配任务
    if not tasks_to_cleanup:
        print(f"没有匹配状态 {status} 的任务")
        return JSONResponse(
            status_code=200,
            content={
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
        )
    
    print(f"\n待清理任务数: {len(tasks_to_cleanup)}")
    
    # 执行清理
    for task_id in tasks_to_cleanup:
        task = rtsp_tasks[task_id]
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
            if task.status in ["processing", "offline"] and task_id in rtsp_detectors:
                print(f"- 停止检测器")
                detector = rtsp_detectors[task_id]
                detector.stop()
                del rtsp_detectors[task_id]
            
            # 从任务列表中删除
            print(f"- 删除任务")
            del rtsp_tasks[task_id]
            
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
        "total": len(rtsp_tasks),
        "by_status": {}
    }
    for task in rtsp_tasks.values():
        after_stats["by_status"][task.status] = after_stats["by_status"].get(task.status, 0) + 1
    
    print(f"\n清理完成:")
    print(f"- 成功: {len(cleanup_results['success'])}")
    print(f"- 跳过: {len(cleanup_results['skipped'])}")
    print(f"- 失败: {len(cleanup_results['failed'])}")
    print(f"剩余任务数: {after_stats['total']}")
    
    return JSONResponse(
        status_code=200,
        content={
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
    )

@app.get("/tasks/rtsp/stats", tags=["tasks"])
async def get_rtsp_tasks_stats():
    """
    获取RTSP任务统计信息
    
    返回:
    - 总任务数
    - 各状态任务数量
    - 活动检测器数
    """
    stats = {
        "total": len(rtsp_tasks),
        "by_status": {},
        "active_detectors": len(rtsp_detectors)
    }
    
    # 统计各状态的任务数量
    for task in rtsp_tasks.values():
        stats["by_status"][task.status] = stats["by_status"].get(task.status, 0) + 1
    
    return stats

@app.post("/analyze/rtsp/test", tags=["analysis"])
async def test_rtsp_connection(request: RtspRequest):
    """测试 RTSP 连接"""
    try:
        print(f"\n开始测试 RTSP 连接:")
        print(f"原始 URL: {request.rtsp_url}")
        
        # 测试原始 URL
        print("\n1. 测试原始 URL:")
        cap = cv2.VideoCapture(request.rtsp_url, cv2.CAP_FFMPEG)
        result = {
            "url": request.rtsp_url,
            "tests": {
                "original": {
                    "is_opened": cap.isOpened(),
                    "error": None
                }
            }
        }
        cap.release()
        
        # 测试 TCP 连接
        print("\n2. 测试 TCP 连接:")
        rtsp_url_tcp = (
            f"{request.rtsp_url}?"
            "rtsp_transport=tcp&"
            "buffer_size=1024000&"
            "max_delay=500000&"
            "stimeout=20000000"
        )
        print(f"TCP URL: {rtsp_url_tcp}")
        cap = cv2.VideoCapture(rtsp_url_tcp, cv2.CAP_FFMPEG)
        result["tests"]["tcp"] = {
            "url": rtsp_url_tcp,
            "is_opened": cap.isOpened(),
            "error": None
        }
        
        if cap.isOpened():
            ret, frame = cap.read()
            result["tests"]["tcp"]["can_read_frame"] = ret
            if ret:
                result["tests"]["tcp"]["frame_info"] = {
                    "shape": frame.shape,
                    "type": str(frame.dtype)
                }
            result["tests"]["tcp"]["properties"] = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
                "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
            }
        cap.release()
        
        # 测试 UDP 连接
        print("\n3. 测试 UDP 连接:")
        rtsp_url_udp = (
            f"{request.rtsp_url}?"
            "rtsp_transport=udp&"
            "buffer_size=1024000&"
            "max_delay=500000&"
            "stimeout=20000000"
        )
        print(f"UDP URL: {rtsp_url_udp}")
        cap = cv2.VideoCapture(rtsp_url_udp, cv2.CAP_FFMPEG)
        result["tests"]["udp"] = {
            "url": rtsp_url_udp,
            "is_opened": cap.isOpened(),
            "error": None
        }
        
        if cap.isOpened():
            ret, frame = cap.read()
            result["tests"]["udp"]["can_read_frame"] = ret
            if ret:
                result["tests"]["udp"]["frame_info"] = {
                    "shape": frame.shape,
                    "type": str(frame.dtype)
                }
            result["tests"]["udp"]["properties"] = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
                "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
            }
        cap.release()
        
        # 使用 ffprobe 获取更多信息
        print("\n4. 使用 ffprobe 测试:")
        try:
            import subprocess
            result_tcp = subprocess.run(
                ['ffprobe', '-v', 'error', '-rtsp_transport', 'tcp', request.rtsp_url],
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            result["tests"]["ffprobe_tcp"] = {
                "returncode": result_tcp.returncode,
                "error": result_tcp.stderr
            }
            
            result_udp = subprocess.run(
                ['ffprobe', '-v', 'error', '-rtsp_transport', 'udp', request.rtsp_url],
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            result["tests"]["ffprobe_udp"] = {
                "returncode": result_udp.returncode,
                "error": result_udp.stderr
            }
        except Exception as e:
            result["tests"]["ffprobe"] = {
                "error": f"ffprobe 测试失败: {str(e)}"
            }
        
        # 添加系统信息
        result["system_info"] = {
            "opencv_version": cv2.__version__,
            "ffmpeg_support": cv2.videoio_registry.getBackendName(cv2.CAP_FFMPEG),
            "platform": sys.platform
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"RTSP 连接测试失败: {str(e)}",
                "url": request.rtsp_url,
                "error": str(e),
                "system_info": {
                    "opencv_version": cv2.__version__,
                    "ffmpeg_support": cv2.videoio_registry.getBackendName(cv2.CAP_FFMPEG),
                    "platform": sys.platform
                }
            }
        )

if __name__ == "__main__":
    print("请使用 run.py 启动完整服务")