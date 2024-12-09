# 使用Python 3.9作为基础镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV OPENCV_VIDEOIO_PRIORITY_V4L=100
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV DOCKER_ENV=true

# 复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制项目文件
COPY . .

# 设置静态文件
RUN python setup_static.py

# 暴露端口
EXPOSE 8000

# 设置入口点
CMD ["python", "run.py"] 