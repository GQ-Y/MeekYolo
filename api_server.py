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
    status: str  # pending, processing, stopped, failed
    error: Optional[str] = None
    rtsp_url: str
    stream_url: Optional[str] = None
    created_at: datetime
    stopped_at: Optional[datetime] = None

# 然后创建FastAPI实例
app = FastAPI(
    title="YOLO Detection API",
    description="YOLO目标检测API服务",
    version="1.0.0",
    docs_url=None,  # 禁用默认的swagger-ui路由
    redoc_url=None  # 禁用默认的redoc路由
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

# 从配置文件加载回调设置
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
    """编码图片为base64"""
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
        
        # 解码图片
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

@app.post("/analyze/image/upload", response_model=List[AnalysisResult])
async def analyze_uploaded_images(
    files: List[UploadFile] = File(
        ...,
        description="选择要分析的图片文件（支持多选）",
        media_type="image/*"
    )
):
    """分析上传的图片文件"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="未提供文件")
        
        detector = MeekYolo()
        results = []
        
        for file in files:
            try:
                if not file.filename:
                    raise HTTPException(status_code=400, detail="文件名为��")
                
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

@app.post("/analyze/video", response_model=VideoAnalysisTask)
async def analyze_video(request: VideoRequest):
    """提交视频分析任务"""
    task_id = str(uuid.uuid4())
    
    # 创建任务记录
    video_tasks[task_id] = VideoAnalysisTask(
        task_id=task_id,
        status="pending",
        progress=0,
        created_at=datetime.now()
    )
    
    # 启动异步处理
    asyncio.create_task(process_video_task(task_id, request.video_url))
    
    return video_tasks[task_id]

@app.post("/analyze/rtsp", response_model=RtspAnalysisTask)
async def analyze_rtsp(request: RtspRequest):
    """启动RTSP流分析"""
    task_id = str(uuid.uuid4())
    
    # 生成默认的 RTMP 推流地址（如果未提供）
    output_rtmp = request.output_rtmp or f"rtmp://your_server/live/{task_id}"
    
    # 创建任务记录
    rtsp_tasks[task_id] = RtspAnalysisTask(
        task_id=task_id,
        status="pending",
        rtsp_url=request.rtsp_url,
        stream_url=output_rtmp,
        created_at=datetime.now()
    )
    
    # 启动异步处理
    asyncio.create_task(process_rtsp_task(
        task_id,
        request.rtsp_url,
        output_rtmp
    ))
    
    return rtsp_tasks[task_id]

@app.get("/analyze/rtsp/{task_id}", response_model=RtspAnalysisTask)
async def get_rtsp_task(task_id: str):
    """获取RTSP分析任务状态"""
    if task_id not in rtsp_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return rtsp_tasks[task_id]

@app.delete("/analyze/rtsp/{task_id}")
async def stop_rtsp_analysis(task_id: str):
    """停止RTSP流分析"""
    if task_id not in rtsp_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = rtsp_tasks[task_id]
    detector = rtsp_detectors.get(task_id)
    
    if task.status == "processing" and detector:
        # 停止检测器
        detector.stop()
        task.status = "stopped"
        task.stopped_at = datetime.now()
        
        # 清理检测器
        del rtsp_detectors[task_id]
        
        # 发送停止通知
        await send_callback(task_id, {
            "status": "stopped",
            "stopped_at": task.stopped_at.isoformat()
        })
    
    return {"message": "分析已停止"}

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}

@app.get("/analyze/video/{task_id}", response_model=VideoAnalysisTask)
async def get_video_task(task_id: str):
    """获取视频分析任务状态"""
    if task_id not in video_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return video_tasks[task_id]

@app.get("/download/video/{task_id}")
async def download_video_result(task_id: str):
    """下载视频分析结果"""
    if task_id not in video_tasks:
        raise HTTPException(status_code=404, detail="��务不存在")
    
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
@app.post("/analyze/video/sync", response_model=List[AnalysisResult])
async def analyze_video_sync(request: VideoRequest):
    """同步分析视频 (不推荐用于大文件)"""
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
                    
            # 如果状态码不是200，等待后重试
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
    
    # 创建新的任务来处理回调发送（包括重试）
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
        
        # 设置保存路径
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
@app.post("/config/callback")
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

@app.get("/config/callback")
async def get_callback_url():
    """获取当前回调配置"""
    return {
        "enabled": callback_config['enabled'],
        "callback_url": callback_config['url'],
        "retry_config": callback_config['retry']
    }

@app.delete("/config/callback")
async def delete_callback_url():
    """删除回调API地址"""
    global callback_config
    callback_config['enabled'] = False
    callback_config['url'] = None
    return {"message": "回调地址已删除"}

async def process_rtsp_task(task_id: str, rtsp_url: str, output_rtmp: Optional[str] = None):
    """异步处理 RTSP 流分析任务"""
    try:
        # 更新任务状态
        task = rtsp_tasks[task_id]
        task.status = "processing"
        
        # 创建检测器实例
        detector = MeekYolo()
        rtsp_detectors[task_id] = detector  # 保存检测器实例
        
        # 禁用 GUI 显示
        detector.config['environment']['enable_gui'] = False
        detector.config['display']['show_window'] = False
        
        # 设置 RTSP 源
        detector.config['source']['type'] = 'rtsp'
        detector.config['source']['rtsp']['url'] = rtsp_url
        
        # 如果提供了 RTMP 地址，设置输出流
        if output_rtmp:
            detector.config['output'] = {
                'enabled': True,
                'rtmp_url': output_rtmp
            }
        
        # 设置任务信息
        detector.set_task_info(
            task_id,
            callback_func=send_callback
        )
        
        # 启动分析
        await send_callback(task_id, {
            "status": "processing",
            "rtsp_url": rtsp_url,
            "stream_url": output_rtmp
        })
        
        # 直接运行异步检测器
        await detector.run()
        
    except Exception as e:
        # 更新失败状态
        task.status = "failed"
        task.error = str(e)
        task.stopped_at = datetime.now()
        
        # 清理检测器
        if task_id in rtsp_detectors:
            del rtsp_detectors[task_id]
        
        # 发送失败通知
        await send_callback(task_id, {
            "status": "failed",
            "error": str(e)
        })

if __name__ == "__main__":
    print("请使用 run.py 启动完整服务")