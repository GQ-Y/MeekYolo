import cv2
import numpy as np
import base64
import requests
from fastapi import HTTPException

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