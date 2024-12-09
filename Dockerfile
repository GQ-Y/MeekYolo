# 使用Python 3.9作为基础镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # 添加网络工具
    curl \
    wget \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV OPENCV_VIDEOIO_PRIORITY_V4L=100
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV DOCKER_ENV=true

# 设置pip源为国内镜像（可选，如果下载过慢可以启用）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制项目文件
COPY . .

# 设置静态文件
RUN python setup_static.py

# 添加健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 设置入口点
CMD ["python", "run.py"] 

# macOS不需要这部分，因为我们使用本地动态库