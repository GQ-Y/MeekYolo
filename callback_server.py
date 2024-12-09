from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime
import json

app = FastAPI(title="Callback Test Server")

@app.post("/")
async def callback_handler(request: Request):
    """处理回调请求"""
    try:
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取回调数据
        data = await request.json()
        
        # 打印回调信息
        print(f"\n[{current_time}] 收到回调:")
        print("-" * 50)
        
        # 获取基本信息
        task_id = data.get("task_id", "未知")
        status = data.get("status", "未知")
        
        print(f"任务ID: {task_id}")
        print(f"状态: {status}")
        
        # 根据不同状态打印不同信息
        if status == "processing":
            if "progress" in data:
                print(f"进度: {data['progress'] * 100:.1f}%")
            if "rtsp_url" in data:
                print(f"RTSP地址: {data['rtsp_url']}")
            if "stream_url" in data:
                print(f"推流地址: {data['stream_url']}")
            if "detections" in data:
                detections = data["detections"]
                print(f"\n检测到 {len(detections)} 个目标:")
                for i, det in enumerate(detections, 1):
                    print(f"\n目标 {i}:")
                    print(f"  类别: {det.get('class', '未知')}")
                    print(f"  跟踪ID: {det.get('track_id', '未知')}")
                    print(f"  置信度: {det.get('confidence', 0):.2f}")
                    if "bbox" in det:
                        bbox = det["bbox"]
                        print(f"  位置: ({bbox.get('x1', 0)}, {bbox.get('y1', 0)}) - ({bbox.get('x2', 0)}, {bbox.get('y2', 0)})")
                        print(f"  大小: {bbox.get('width', 0)}x{bbox.get('height', 0)}")
            if "timestamp" in data:
                print(f"\n时间戳: {data['timestamp']}")
                
        elif status == "completed":
            print(f"进度: 100%")
            if "result_path" in data:
                print(f"结果路径: {data['result_path']}")
            if "download_url" in data:
                print(f"下载地址: {data['download_url']}")
                
        elif status == "failed":
            print(f"错误信息: {data.get('error', '未知错误')}")
            
        elif status == "stopped":
            print(f"停止时间: {data.get('stopped_at', '未知')}")
        
        # 打印完整的原始数据（格式化后的）
        print("\n原始数据:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        print("-" * 50)
        return {"status": "success"}
        
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理回调失败:")
        print(f"错误信息: {str(e)}")
        print("-" * 50)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("\n回调测试服务器启动中...")
    print("监听地址: http://localhost:8081")
    print("等待接收回调...\n")
    uvicorn.run(app, host="0.0.0.0", port=8081) 