import os
import requests

def setup_swagger_ui():
    """下载并设置Swagger UI文件"""
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # Swagger UI 必要文件列表
    files = {
        "swagger-ui-bundle.js": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        "swagger-ui.css": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        "swagger-ui-standalone-preset.js": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js",
        "favicon.png": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/favicon-32x32.png"
    }
    
    for filename, url in files.items():
        filepath = os.path.join(static_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {filename}")

if __name__ == "__main__":
    setup_swagger_ui() 