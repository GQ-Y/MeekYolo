#!/bin/bash

# 启动虚拟显示服务器
Xvfb :99 -screen 0 1024x768x24 -ac &

# 等待Xvfb启动
sleep 1

# 设置显示并启动应用
export DISPLAY=:99
exec python run.py 