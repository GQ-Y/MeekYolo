# MeekYolo 目标检测与跟踪系统

## 项目介绍

MeekYolo是一个基于YOLOv11的目标检测与跟踪系统，支持多种输入源（RTSP流、单张图片、批量图片、视频文件），可以进行实时目标检测、目标跟踪，并提供丰富的可视化配置选项。

## 主要功能

- 多输入源支持：
  - RTSP视频流实时检测
  - 单张图片检测
  - 批量图片检测
  - 视频文件检测
- 目标检测与跟踪：
  - 支持多目标检测
  - 实时目标跟踪
  - 目标ID保持
- 可视化功能：
  - 目标框显示
  - 目标类别显示
  - 跟踪ID显示
  - 置信度显示
  - 位置信息显示
  - 尺寸信息显示
  - FPS显示
- 结果输出：
  - 图片结果保存
  - 视频结果保存
  - 控制台信息输出

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- OpenCV 4.5+
- Ultralytics 8.0+

## 环境安装

1. 创建虚拟环境：

   ```bash
   python -m venv venv
   source venv/bin/activate # Linux/Mac
   venv\Scripts\activate # Windows
   ```
2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 快速启动

1. 配置config.yaml文件
2. 运行程序：
   ```bash
   python yolo_rtsp.py
   ```
3. 输入命令控制程序：
   - `start` - 开始分析
   - `stop` - 停止分析
   - `help` - 显示帮助信息
   - `status` - 显示当前状态
   - `config` - 显示当前配置
   - `quit`/`exit` - 退出程序

## 使用说明

### 1. RTSP流检测

```yaml
source:
type: "rtsp"
rtsp:
url: "rtsp://your_rtsp_url"
ffmpeg_options:
"?tcp"
```

### 2. 单张图片检测

```yaml
source:
type: "image"
image:
path: "data/test.jpg"
save_path: "results/test_result.jpg"
```

### 3. 批量图片检测

```yaml
source:
type: "images"
images:
input_dir: "data/images"
save_dir: "results/images"
formats: [".jpg", ".jpeg", ".png"]
```

### 4. 视频文件检测

```yaml
source:
type: "video"
video:
path: "data/test.mp4"
save_path: "results/test_result.mp4"
fps: 30
```

## 配置文件说明

### 1. 输入源配置 (source)

```yaml
source:
type: "rtsp"/"image"/"images"/"video" # 输入源类型
```

### 2. 模型配置 (model)

```yaml
model:
path: "model/best.pt" # 模型路径
conf_thres: 0.5 # 置信度阈值
```

### 3. 显示配置 (display)

```yaml
display:
window_name: "MeekYolo" # 窗口名称
show_fps: true # 是否显示FPS
```

### 4. 控制台输出配置 (console)

```yaml
console:
enabled: false # 是否启用控制台输出
show_time: false # 显示时间戳
show_total: false # 显示总目标数
show_details: false # 显示详细信息
show_separator: false # 显示分隔线
```

### 5. 可视化视化配置 (visualization)

```yaml
visualization:
show_box: true # 显示目标框
show_class: true # 显示类别
show_track_id: true # 显示跟踪ID
show_confidence: true # 显示置信度
show_position: true # 显示位置信息
show_size: true # 显示尺寸信息
show_anchor: true # 显示锚点
show_line: true # 显示连接线
style:
font_scale: 0.6 # 字体大小
thickness: 2 # 线条粗细
chinese_text_size: 20 # 中文字体大小
margin: 5 # 边距
colors:
text: [255,255,255] # 文本颜色
background: [0,0,0] # 背景颜色
```

### 6. 跟踪配置 (tracking)

```yaml
tracking:
enabled: true # 是否启用跟踪
persist: true # 是否保持ID
```

### 7. 打印控制 (print)

```yaml
print:
enabled: false # 是否允许任何打印输出
```

## 注意事项

1. 确保模型文件存在于指定路径
2. 检查输入源路径的正确性
3. 确保有足够的磁盘空间保存结果
4. 对于RTSP流，确保网络连接稳定
5. 建议根据实际需求调整置信度阈值

## 常见问题

1. 如果出现中文显示问题，请检查字体文件路径
2. 如果检测效果不理想，可以调整置信度阈值
3. 如果跟踪不稳定，可以调整persist参数
4. 如果FPS过低，可以考虑关闭一些可视化选项

## 更新日志

- 2024.12.6: 初始版本发布
  - 支持多种输入源
  - 添加目标跟踪功能
  - 完善可视化配置
  - 支持中文类别显示
  - 支持读取yolo配置文件描述
