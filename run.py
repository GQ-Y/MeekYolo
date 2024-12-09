import os
import uvicorn
from api_server import app
from yolo_rtsp import MeekYolo
from setup_static import setup_swagger_ui
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# 确保static目录存在
if not os.path.exists("static"):
    os.makedirs("static")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 自定义swagger-ui路由
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png"
    )

# 添加根路径重定向
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

def run_cli():
    """运行命令行界面"""
    detector = MeekYolo()
    detector.run()

if __name__ == "__main__":
    # 首先设置静态文件
    setup_swagger_ui()
    
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000) 