from pydantic import BaseModel
from typing import Optional
from datetime import datetime

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
    callback_interval: float = 1.0
    last_callback: Optional[datetime] = None 