import cv2
from typing import Tuple, Optional

async def test_rtsp_connection(url: str) -> Tuple[bool, Optional[str]]:
    """测试RTSP连接"""
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return False, "无法打开RTSP流"
            
        # 尝试读取一帧
        ret, frame = cap.read()
        if not ret:
            return False, "无法读取视频帧"
            
        cap.release()
        return True, None
        
    except Exception as e:
        return False, str(e)

def get_rtsp_info(url: str) -> dict:
    """获取RTSP流信息"""
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise Exception("无法打开RTSP流")
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
        
    except Exception as e:
        raise Exception(f"获取RTSP流信息失败: {str(e)}") 