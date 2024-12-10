# 第一阶段: 编译ZLMediaKit
FROM ubuntu:22.04 AS builder

# 设置环境变量避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装编译工具
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libssl-dev \
    libmysqlclient-dev \
    libx264-dev \
    libfaac-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /build

# 编译安装libmp4v2
RUN git clone https://github.com/enzo1982/mp4v2.git && \
    cd mp4v2 && \
    autoreconf -fiv && \
    ./configure && \
    make -j4 && \
    make install && \
    ldconfig

# 复制本地的ZLMediaKit代码
COPY ZLMediaKit/ ZLMediaKit/

# 编译ZLMediaKit
RUN cd ZLMediaKit && \
    rm -rf build && \
    mkdir -p build && \
    cd build && \
    cmake .. -DENABLE_API=ON -DENABLE_API_STATIC_LIB=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j4 && \
    make install && \
    echo "=== Installed files in /usr/local ===" && \
    find /usr/local -type f -name "*mk*" && \
    echo "=== Source header files ===" && \
    find ../src -name "*.h"

# 第二阶段: Python运行环境
FROM python:3.9-slim

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    libssl3 \
    libmariadb3 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN useradd -m -s /bin/bash appuser

# 复制ZLMediaKit编译结果
COPY --from=builder /usr/local/lib/libmk_api.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libmk.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libmpeg.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libzlmediakit.so* /usr/local/lib/
COPY --from=builder /usr/local/include/mk_* /usr/local/include/
COPY --from=builder /build/ZLMediaKit/src/*.h /usr/local/include/
COPY --from=builder /build/ZLMediaKit/src/Extension/*.h /usr/local/include/Extension/
COPY --from=builder /usr/local/lib/libmp4v2.* /usr/local/lib/
COPY --from=builder /usr/local/include/mp4v2 /usr/local/include/mp4v2

# 创建必要的目录
RUN mkdir -p /usr/local/include/Extension

# 更新库缓存
RUN ldconfig && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && \
    ldconfig

# 设置库搜索路径
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 设置工作目录
WORKDIR /app

# 复制项目文件和默认配置
COPY . .
COPY config/default_config.yaml /app/config/default_config.yaml

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "pydantic[email]>=2.0" pydantic-settings && \
    pip install --no-cache-dir "lap>=0.5.12" && \
    pip install --upgrade pip

# 创建必要的目录并设置权限（在切换用户前）
RUN mkdir -p /home/appuser/ZLMediaKit/release/linux/Release && \
    mkdir -p /app/config && \
    mkdir -p /app/data && \
    mkdir -p /app/model && \
    mkdir -p /app/results && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /home/appuser/ZLMediaKit && \
    chown -R appuser:appuser /app/config && \
    chown -R appuser:appuser /app/data && \
    chown -R appuser:appuser /app/model && \
    chown -R appuser:appuser /app/results && \
    chown -R appuser:appuser /app/logs && \
    ln -s /usr/local/lib/libmk_api.so /home/appuser/ZLMediaKit/release/linux/Release/libmk_api.so

# 切换到非root用户
USER appuser

# 设置Python路径
ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages:/home/appuser/.local/lib/python3.9/site-packages:$PYTHONPATH
ENV PATH=/home/appuser/.local/bin:$PATH

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python", "run.py"]

# 添加健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1