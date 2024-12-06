from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import cv2
import numpy as np
import io
from PIL import Image
import uuid
import threading
from yolo_rtsp import MeekYolo
import uvicorn
import requests

app = FastAPI(
    title="MeekYolo API",
    description="目标检测与跟踪系统API服务",
    docs_url=None  # 禁用默认的swagger UI
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境中应该限制具体的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

@app.get("/docs", response_class=HTMLResponse)
async def custom_swagger_ui_html():
    """自定义Swagger UI页面"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API文档",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png",
        # 自定义配置
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,  # 隐藏Models部分
            "tryItOutEnabled": True,  # 默认启用Try it out
            "displayRequestDuration": True,  # 显示请求耗时
            "filter": True,  # 启用过滤功能
        }
    )

# 存储RTSP分析任务
rtsp_tasks = {}

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

class AnalysisResult(BaseModel):
    """分析结果模型"""
    image_base64: Optional[str] = None  # 处理后的图片base64
    detections: List[dict]  # 检测结果
    stream_url: Optional[str] = None  # RTSP推流地址

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
            resp = requests.get(image_data, verify=False)  # 添加verify=False以处理HTTPS
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
        # 读取上传的文件内容
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="无法解码图片文件")
        
        # 处理图片
        detections = detector.process_frame(frame)
        
        # 绘制结果
        processed_frame = detector.draw_results(frame, detections)
        
        # 编码处理后的图片
        result_base64 = encode_image(processed_frame)
        
        # 格式化检测结果
        formatted_detections = format_detections(detections)
        
        return AnalysisResult(
            image_base64=result_base64,
            detections=formatted_detections
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"处理图片失败: {str(e)}")

@app.post("/analyze/image", response_model=List[AnalysisResult])
async def analyze_image(request: ImageRequest):
    """分析单张或多张图片 (通过URL或base64)"""
    try:
        detector = MeekYolo()
        results = []
        
        for image_data in request.images:
            # 解码图片
            frame = decode_image(image_data, request.is_base64)
            
            # 处理图片
            detections = detector.process_frame(frame)
            
            # 绘制结果
            processed_frame = detector.draw_results(frame, detections)
            
            # 编码处理后的图片
            result_base64 = encode_image(processed_frame)
            
            # 格式化检测结果
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
    detector = MeekYolo()
    results = []
    
    for file in files:
        # 检查文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 不是图片格式")
        
        # 处理图片
        result = await process_image_file(file, detector)
        results.append(result)
    
    return results

@app.post("/analyze/video", response_model=List[AnalysisResult])
async def analyze_video(request: VideoRequest):
    """分析视频文件"""
    detector = MeekYolo()
    results = []
    
    # 打开视频
    cap = cv2.VideoCapture(request.video_url)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="无法打开视频文件")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            detections = detector.process_frame(frame)
            
            # 格式化检测结果
            formatted_detections = format_detections(detections)
            results.append(AnalysisResult(detections=formatted_detections))
    
    finally:
        cap.release()
    
    return results

@app.post("/analyze/rtsp", response_model=AnalysisResult)
async def analyze_rtsp(request: RtspRequest):
    """启动RTSP流分析"""
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    
    # 创建推流地址
    stream_url = f"rtmp://your_server/live/{task_id}"
    
    # 创建检测器实例
    detector = MeekYolo()
    
    # 配置RTSP源
    detector.config['source']['type'] = 'rtsp'
    detector.config['source']['rtsp']['url'] = request.rtsp_url
    
    # 启动分析线程
    def run_analysis():
        rtsp_tasks[task_id]['detector'] = detector
        detector.run()
    
    analysis_thread = threading.Thread(target=run_analysis)
    rtsp_tasks[task_id] = {
        'thread': analysis_thread,
        'detector': None,
        'stream_url': stream_url
    }
    analysis_thread.start()
    
    return AnalysisResult(
        stream_url=stream_url,
        detections=[]  # 初始返回空检测结果
    )

@app.delete("/analyze/rtsp/{task_id}")
async def stop_rtsp_analysis(task_id: str):
    """停止RTSP流分析"""
    if task_id not in rtsp_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = rtsp_tasks[task_id]
    if task['detector']:
        task['detector'].running = False
    task['thread'].join()
    del rtsp_tasks[task_id]
    
    return {"message": "分析已停止"}

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}

if __name__ == "__main__":
    print("请使用 run.py 启动完整服务") 