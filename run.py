import threading
import uvicorn
import os
from pathlib import Path
from api_server import app
from yolo_rtsp import MeekYolo
from setup_static import setup_static_files

def run_api_server():
    """运行API服务器"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_cli():
    """运行命令行界面"""
    detector = MeekYolo()
    detector.run()

if __name__ == "__main__":
    # 确保static目录和文件存在
    if not Path("static").exists() or not all(Path("static").glob("*")):
        setup_static_files()
    
    # 启动API服务器线程
    api_thread = threading.Thread(target=run_api_server)
    api_thread.daemon = True  # 设置为守护线程，这样主程序退出时API服务也会退出
    api_thread.start()
    
    # 运行命令行界面
    run_cli() 