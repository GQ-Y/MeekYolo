from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from api.models.analysis import ImageRequest, VideoRequest, RtspRequest, AnalysisResult
from api.services.analysis import AnalysisService
from api.services.callback import CallbackService

router = APIRouter()
analysis_service = AnalysisService()
callback_service = CallbackService()

@router.post("/image", response_model=List[AnalysisResult])
async def analyze_image(request: ImageRequest):
    """分析单张或多张图片 (通过URL或base64)"""
    return await analysis_service.analyze_images(request)

@router.post("/image/upload", response_model=List[AnalysisResult])
async def analyze_uploaded_images(files: List[UploadFile] = File(...)):
    """分析上传的图片文件"""
    return await analysis_service.analyze_uploaded_images(files)

@router.post("/video")
async def analyze_video(request: VideoRequest):
    """提交视频分析任务"""
    return await analysis_service.start_video_analysis(request)

@router.post("/rtsp")
async def analyze_rtsp(request: RtspRequest):
    """启动RTSP流分析"""
    return await analysis_service.start_rtsp_analysis(request)

@router.post("/rtsp/test")
async def test_rtsp_connection(request: RtspRequest):
    """测试RTSP连接"""
    return await analysis_service.test_rtsp_connection(request) 