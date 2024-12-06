import os
import requests
from pathlib import Path

def setup_static_files():
    """设置静态文件"""
    # 创建static目录
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # 定义需要下载的文件
    files = {
        "swagger-ui.css": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.11.0/swagger-ui.css",
        "swagger-ui-bundle.js": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.11.0/swagger-ui-bundle.js",
        "favicon.png": "https://fastapi.tiangolo.com/img/favicon.png"
    }
    
    # 下载文件
    for filename, url in files.items():
        file_path = static_dir / filename
        if not file_path.exists():
            print(f"下载 {filename}...")
            response = requests.get(url)
            response.raise_for_status()
            file_path.write_bytes(response.content)
            print(f"{filename} 下载完成")
        else:
            print(f"{filename} 已存在")

if __name__ == "__main__":
    setup_static_files() 