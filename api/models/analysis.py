from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

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