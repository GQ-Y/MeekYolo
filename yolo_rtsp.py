import cv2
import numpy as np
import yaml
import time
import torch
import torch.nn as nn
from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import threading
import queue
import sys
from threading import Thread
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import base64
from utils.rtsp_proxy import RTSPProxy
import logging
from api.core.managers import task_manager

logger = logging.getLogger(__name__)

class CommandReader(Thread):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.running = True
        self.daemon = True  # 设置为守护线程

    def run(self):
        while self.running:
            try:
                command = input("请输入命令> ").strip()
                if command:  # 只处理非空命令
                    self.queue.put(command.lower())
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

    def stop(self):
        self.running = False
        # 清空队列
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

class RtspFrameReader:
    def __init__(self, rtsp_url: str, queue_size: int = 30):
        self.rtsp_url = rtsp_url
        self.frame_queue = asyncio.Queue(maxsize=queue_size)
        self.running = False
        self.read_task = None
        
        # 检测系统平台
        self.platform = sys.platform
        
        # 初始化硬件解码配置
        self.hw_decoders = {
            'win32': [
                {'api': cv2.CAP_MSMF, 'name': 'MSMF'},  # Windows Media Foundation
                {'api': cv2.CAP_DSHOW, 'name': 'DirectShow'}, 
            ],
            'linux': [
                {'api': cv2.CAP_FFMPEG, 'name': 'VAAPI', 'options': 'hw_device_type=vaapi'},
                {'api': cv2.CAP_FFMPEG, 'name': 'NVDEC', 'options': 'hw_device_type=cuda'},
            ],
            'darwin': [
                {'api': cv2.CAP_FFMPEG, 'name': 'VideoToolbox', 'options': 'videotoolbox'},
            ]
        }
        
    async def _init_capture(self):
        """初始化视频捕获"""
        # 获取当前平台的解码器列表
        decoders = self.hw_decoders.get(self.platform, [])
        if not decoders:
            decoders = [{'api': cv2.CAP_FFMPEG, 'name': 'Software'}]
        
        # 构建基础RTSP URL参数
        base_params = (
            "rtsp_transport=tcp&"
            "buffer_size=1024000&"
            "max_delay=500000&"
            "stimeout=20000000&"
            "reorder_queue_size=0"
        )
        
        # 尝试每个解码器
        for decoder in decoders:
            try:
                print(f"\n尝试使用 {decoder['name']} 解码器:")
                
                # 构建完整的URL
                url = f"{self.rtsp_url}?{base_params}"
                if 'options' in decoder:
                    url += f"&{decoder['options']}"
                print(f"URL: {url}")
                
                # 创建VideoCapture实例
                cap = cv2.VideoCapture(url, decoder['api'])
                
                # 设置缓冲区大小
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                # 测试是否可以读取帧
                if not cap.isOpened():
                    print(f"{decoder['name']} 解码器打开失败")
                    cap.release()
                    continue
                    
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"{decoder['name']} 解码器无法读取帧")
                    cap.release()
                    continue
                
                print(f"{decoder['name']} 解码器初始化成功")
                
                # 获取视频信息
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"视频信息: {width}x{height} @ {fps}fps")
                
                return cap
                
            except Exception as e:
                print(f"{decoder['name']} 解码器初始化失败: {str(e)}")
                continue
        
        # 如果所有解码器都失败,使用默认的软解
        print("\n所有硬件解码器都失败,使用软解码:")
        url = f"{self.rtsp_url}?{base_params}"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if not cap.isOpened():
            raise Exception("无法初始化视频捕获")
            
        return cap
        
    async def start(self):
        """启动读取任务"""
        self.running = True
        self.read_task = asyncio.create_task(self._read_frames())
        
    async def stop(self):
        """停止读取任务"""
        self.running = False
        if self.read_task:
            await self.read_task
        
    async def get_frame(self) -> Optional[np.ndarray]:
        """获取一帧图像"""
        try:
            return await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
        
    async def _read_frames(self):
        """异步读取帧"""
        try:
            # 初始化视频捕获
            cap = await self._init_capture()
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.1)
                    continue
                    
                # 如果队列满了，移除最旧的帧
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                
                await self.frame_queue.put(frame)
                await asyncio.sleep(0.001)  # 避免CPU占用过高
                
        except Exception as e:
            print(f"读取帧异常: {str(e)}")
            raise
            
        finally:
            if 'cap' in locals():
                cap.release()

class MeekYolo:
    def __init__(self, config_path='config/config.yaml'):
        logger.info("MeekYoloV11 启动成功")
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 检查是否在Docker环境中运行
        self.is_docker = os.environ.get('DOCKER_ENV', '').lower() == 'true'
        if self.is_docker:
            # 在Docker环境中强制禁用GUI
            self.config['environment']['is_docker'] = True
            self.config['environment']['enable_gui'] = False
            self.config['display']['show_window'] = False
            
        # 初始化模型
        self.initialize_model()
        
        # 初始化输入源
        self.initialize_source()
        
        # 从模型配置文件加载类别名称
        self.load_class_names()
        
        # 加载中文字体
        self.font_path = '/System/Library/Fonts/PingFang.ttc'  # MacOS系统字体路径
        # 如果是Windows系统，可以使用类似路径：
        # self.font_path = 'C:/Windows/Fonts/simhei.ttf'
        
        # 加载模型
        self.model = YOLO(self.config['model']['path'])
        # 根据配置决定是否启用跟踪
        if self.config['tracking']['enabled']:
            self.tracker = self.model.track
        else:
            self.tracker = self.model.predict
        self.model.to(self.device)
        
        # 用于存储目标颜色映射
        self.id_colors = {}
        
        # 添加运行状态控制
        self.running = False
        # 添加分析线程
        self.analysis_thread = None
        # 加帧缓冲
        self.current_frame = None
        self.frame_ready = False
        self.frame_lock = threading.Lock()
        
        # 添加回调函数属性
        self.callback_func = None
        self.task_id = None  # 添加任务ID属性
        self.rtsp_proxy = RTSPProxy()
        
        # 使用全任务管理器
        self.task_manager = task_manager
        
        logger.info("YOLO检测器初始化完成")

    def initialize_model(self):
        # 设置设备
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 添加DetectionModel到安全globals列表
        add_safe_globals([DetectionModel])
        
        # 加载模型
        self.model = YOLO(self.config['model']['path'])
        # 设置verbose参数为False以禁用YOLO内部打印
        self.model.verbose = False
        self.model.to(self.device)

    def initialize_source(self):
        """初始化输入源"""
        source_type = self.config['source']['type']
        
        if source_type == 'rtsp':
            # 不在这里初始化RTSP流，而是在process_rtsp中处理
            self.process_func = None
            
        elif source_type == 'image':
            self.process_func = self.process_image
            self.image_path = self.config['source']['image']['path']
            self.save_path = self.config['source']['image']['save_path']
            # 确保保存路径存在
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
        elif source_type == 'images':
            self.process_func = self.process_images
            self.input_dir = self.config['source']['images']['input_dir']
            self.save_dir = self.config['source']['images']['save_dir']
            self.image_formats = self.config['source']['images']['formats']
            # 确保保存目录存在
            os.makedirs(self.save_dir, exist_ok=True)
            # 获取所有图片文件
            self.image_files = []
            for fmt in self.image_formats:
                self.image_files.extend(glob.glob(os.path.join(self.input_dir, f"*{fmt}")))
            
        elif source_type == 'video':
            self.process_func = self.process_video
            self.video_path = self.config['source']['video']['path']
            self.save_path = self.config['source']['video']['save_path']
            self.output_fps = self.config['source']['video']['fps']
            # 确保保存路径存在
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
        else:
            raise ValueError(f"不支持的输入类型: {source_type}")

    async def process_rtsp(self, rtsp_url: str):
        """处理RTSP视频流
        
        Args:
            rtsp_url (str): RTSP流地址，格式如：
                - rtsp://username:password@ip:port/path
                - rtsp://ip:port/path
        
        Raises:
            ValueError: URL格式错误
            Exception: 视频流打开失败或处理过程中的其他错误
        """
        cap = None
        try:
            # 确保URL是字符串
            if not isinstance(rtsp_url, str):
                raise ValueError("RTSP URL must be a string")
            
            # 设置FFMPEG参数以优化RTSP流接收
            rtsp_url += "?rtsp_transport=tcp"  # 强制使用TCP
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            
            # 设置缓冲区参数
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 设置缓冲区大小
            
            # 设置FFMPEG特定参数
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            cap.set(cv2.CAP_PROP_FPS, 25)  # 设置期望的帧率
            
            if not cap.isOpened():
                raise Exception(f"无法打开RTSP流: {rtsp_url}")
            
            self.running = True
            if self.config['print']['enabled']:
                logger.info(f"开始处理RTSP流: {rtsp_url}")
            self.last_callback = time.time()
            
            consecutive_errors = 0  # 连续错误计数
            max_consecutive_errors = 5  # 最大连续错误次数
            
            while self.running:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_errors += 1
                        logger.warning(f"读取帧失败 ({consecutive_errors}/{max_consecutive_errors})")
                        if consecutive_errors >= max_consecutive_errors:
                            raise Exception("连续多次读取帧失败")
                        await asyncio.sleep(0.1)
                        continue
                    
                    consecutive_errors = 0  # 重置错误计数
                    
                    # 处理帧
                    results = self.process_frame(frame)
                    
                    # 格式化结果
                    formatted_results = self.format_detections(results)
                    
                    # 只有检测到目标时才进行回调
                    if formatted_results and self.callback_func and time.time() - self.last_callback >= self.callback_interval:
                        try:
                            # 生成时间戳
                            timestamp = datetime.now()
                            
                            # 绘制结果
                            result_frame = self.draw_results(frame.copy(), results)
                            # 生成文件名
                            image_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                            image_path = os.path.join("results", "images", self.task_id, image_filename)
                            os.makedirs(os.path.dirname(image_path), exist_ok=True)
                            
                            # 保存图片
                            cv2.imwrite(image_path, result_frame)
                            
                            # 转换为base64
                            _, buffer = cv2.imencode('.jpg', result_frame)
                            image_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # 准备回调数据
                            callback_data = {
                                "task_id": self.task_id,
                                "timestamp": timestamp.isoformat(),
                                "detections": formatted_results,
                                "image_base64": image_base64,
                                "image_path": image_path,
                                "has_detections": True
                            }
                            
                            # 只在配置启用时记录日志
                            if self.config['print']['enabled']:
                                logger.info(f"检测到 {len(formatted_results)} 个目标，已保存图片: {image_path}")
                                logger.info(f"目标详情: {formatted_results}")
                                logger.info(f"准备发送回调: {callback_data}")
                            
                            # 执行回调
                            await self.callback_func(callback_data)
                            
                            # 更新最后回调时间
                            self.last_callback = time.time()
                            
                        except Exception as callback_error:
                            logger.error(f"回调执行异常: {str(callback_error)}")
                
                    # 控制理速度
                    await asyncio.sleep(0.01)  # 避免CPU占用过高
                    
                except cv2.error as e:
                    logger.error(f"OpenCV错误: {str(e)}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise
                    await asyncio.sleep(0.1)
                    continue
                
        except Exception as e:
            logger.error(f"处理RTSP流异常: {str(e)}")
            raise
            
        finally:
            self.running = False
            if cap is not None:
                cap.release()

    def process_image(self):
        """处理单张图片"""
        # 读取图片
        frame = cv2.imread(self.image_path)
        if frame is None:
            raise Exception(f"无法读取图片: {self.image_path}")
        
        # 设置最大分析次数和最小目标数变化阈值
        max_attempts = 5
        min_change_threshold = 0.1  # 目标数量变化阈值（10%）
        prev_num_objects = 0
        stable_count = 0
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            # 处理图片
            results = self.process_frame(frame)
            current_num_objects = len(results)
            
            # 计算目标数量的变化率
            if prev_num_objects > 0:
                change_rate = abs(current_num_objects - prev_num_objects) / prev_num_objects
            else:
                change_rate = 1.0 if current_num_objects > 0 else 0.0
            
            # 如果目标数量变化小于阈值，增加稳定计数
            if change_rate <= min_change_threshold:
                stable_count += 1
            else:
                stable_count = 0
            
            # 如果连续两次检测结果稳定认为检测到目标
            if stable_count >= 2:
                break
            
            prev_num_objects = current_num_objects
            
            if self.config['print']['enabled']:
                print(f"\r分析进度: {attempt}/{max_attempts}, 当前检测到 {current_num_objects} 个目标", 
                      end="", flush=True)
            
            # 短暂等待以确保模型有足够时间处理
            time.sleep(0.1)
        
        # 处理图片
        results = self.process_frame(frame)
        frame = self.draw_results(frame, results)
        
        # 保存结果
        cv2.imwrite(self.save_path, frame)
        if self.config['print']['enabled']:
            print(f"\n分析完成，共检测到 {len(results)} 个目标")
            print(f"结果已保存至: {self.save_path}")
        
        self.running = False

    def process_images(self):
        """处理多张图片"""
        total = len(self.image_files)
        self.running = True
        for i, image_path in enumerate(self.image_files, 1):
            # 取图片
            frame = cv2.imread(image_path)
            if frame is None:
                if self.config['print']['enabled']:
                    print(f"无法读取图片: {image_path}")
                continue
            
            # 处理图片
            results = self.process_frame(frame)
            frame = self.draw_results(frame, results)
            
            # 生成��存路径
            save_name = os.path.basename(image_path)
            save_path = os.path.join(self.save_dir, save_name)
            
            # 保存结果
            cv2.imwrite(save_path, frame)
            if self.config['print']['enabled']:
                print(f"处理进度: {i}/{total}, 保存: {save_path}")
            
            if not self.running:
                break
        
        self.running = False

    def process_video(self, video_path: str, save_path: str, progress_callback=None):
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径或URL
            save_path: 结果保存路径
            progress_callback: 进度回调函数，接收一个float参数(0-1)
        """
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("无法打开视频文件")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 建频写入
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 处理帧
                results = self.process_frame(frame)
                frame = self.draw_results(frame, results)
                
                # 写入结果
                out.write(frame)
                
                # 回进
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress)
                
                # 释放内存
                del results
                
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # 确保最终进度为100%
            if progress_callback:
                progress_callback(1.0)

    def get_color(self, track_id):
        """为每个track_id生成唯一颜色"""
        if track_id not in self.id_colors:
            # 生成随机颜色,排近黑色和白色的颜色
            color = tuple(map(int, np.random.randint(50, 200, 3)))
            self.id_colors[track_id] = color
        return self.id_colors[track_id]

    def process_frame(self, frame: np.ndarray) -> list:
        """处理单帧图像"""
        # 根据配置决定是否使用跟踪
        if self.config['tracking']['enabled']:
            results = self.tracker(frame, 
                                 conf=self.config['model']['conf_thres'],
                                 persist=self.config['tracking']['persist'],
                                 verbose=False)
        else:
            results = self.tracker(frame, 
                                 conf=self.config['model']['conf_thres'],
                                 verbose=False)
        
        processed_results = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            # 只在配置启用且检测到目标时记录日志
            if len(boxes) > 0 and self.config['print']['enabled']:
                logger.info(f"检测到 {len(boxes)} 个目标")
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                cls_idx = int(box.cls.item())
                cls_name = self.names.get(cls_idx, '未知')
                # 根据配定是否获取跟踪ID
                track_id = int(box.id[0]) if (self.config['tracking']['enabled'] and box.id is not None) else -1
                
                processed_results.append(([x1, y1, x2, y2], conf, cls_name, track_id))
        
        return processed_results

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def cv2AddChineseText(self, img, text, position, textColor=(255, 255, 255), textSize=20):
        """
        在图片上添加文本
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 创建字体对象
        font = ImageFont.truetype(self.font_path, textSize)
        
        # 创建绘制对象
        draw = ImageDraw.Draw(img)
        
        # 绘制文本
        draw.text(position, text, textColor, font=font)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def draw_results(self, frame, results):
        # 式配置
        style = self.config['visualization']['style']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = style['font_scale']
        thickness = style['thickness']
        chinese_text_size = style['chinese_text_size']
        margin = style['margin']
        
        # 获取颜色配置
        text_color = tuple(style['colors']['text'])
        bg_color = tuple(style['colors']['background'])
        
        # 存储所有信息框的位置
        info_boxes = []
        
        # 首先绘制所有目标框
        if self.config['visualization']['show_box']:
            for box, score, cls_name, track_id in results:
                x1, y1, x2, y2 = map(int, box)
                box_color = self.get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                
                # 根据配置决是否显示锚
                if self.config['visualization']['show_anchor']:
                    anchor_size = 4
                    cv2.drawMarker(frame, (x1, y1), box_color, cv2.MARKER_CROSS, anchor_size, thickness)
                    cv2.drawMarker(frame, (x2, y2), box_color, cv2.MARKER_CROSS, anchor_size, thickness)
        
        # 然后绘制信息框，处理重叠
        for box, score, cls_name, track_id in results:
            x1, y1, x2, y2 = map(int, box)
            box_color = self.get_color(track_id)
            
            # 准备显示的信息
            info_list = []
            if self.config['visualization']['show_track_id']:
                info_list.append(f"ID: {track_id}")
            if self.config['visualization']['show_confidence']:
                info_list.append(f"Conf: {score:.2f}")
            if self.config['visualization']['show_position']:
                info_list.append(f"Pos: ({x1},{y1})-({x2},{y2})")
            if self.config['visualization']['show_size']:
                info_list.append(f"Size: {x2-x1}x{y2-y1}")
            
            if not info_list and not self.config['visualization']['show_class']:
                continue  # 如果有任何信息需要显示，直接跳过
            
            # 计本大小
            text_heights = []
            text_widths = []
            for text in info_list:
                (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_heights.append(h)
                text_widths.append(w)
            
            if text_heights:
                text_height = max(text_heights)
                text_width = max(text_widths)
            else:
                text_height = 20  # 默认高度
                text_width = 100  # 默认宽度
            
            # 计信息框的位置和大小 - 放置在目标左侧
            info_height = (len(info_list) + 1) * (text_height + margin)  # +1 是为了类别信息
            text_bg_x2 = max(0, x1 - margin)  # 确保不会超出画面左边界
            text_bg_x1 = max(0, text_bg_x2 - text_width - 2 * margin)
            text_bg_y1 = max(0, y1 + (y2 - y1 - info_height) // 2)  # 直中
            text_bg_y2 = min(frame.shape[0], text_bg_y1 + info_height)
            
            # 检查重并调整位置
            while any(self.check_overlap(
                (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2),
                existing_box
            ) for existing_box in info_boxes):
                text_bg_x1 -= text_width + margin  # 向左移动
                text_bg_x2 -= text_width + margin
                # 确保不会超出画面左边界
                if text_bg_x1 < 0:
                    text_bg_x1 = max(0, x1 - margin - text_width - 2 * margin)
                    text_bg_x2 = max(0, x1 - margin)
                    text_bg_y1 = text_bg_y2 + margin  # 如果左边放不，就到下面
                    text_bg_y2 = text_bg_y1 + info_height
            
            info_boxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))
            
            # 绘制信框背景
            cv2.rectangle(frame, 
                         (text_bg_x1, text_bg_y1), 
                         (text_bg_x2, text_bg_y2), 
                         bg_color, -1)
            
            # 绘制连接线
            if self.config['visualization']['show_line']:
                line_start = (text_bg_x2, text_bg_y1 + info_height // 2)
                line_end = (x1, y1 + (y2 - y1) // 2)
                cv2.line(frame, line_start, line_end, box_color, 1)
            
            # 制文本信息
            current_y = text_bg_y1 + margin
            
            # 绘类别信息
            if self.config['visualization']['show_class']:
                frame = self.cv2AddChineseText(frame, 
                                             f"类型: {cls_name}",
                                             (text_bg_x1 + margin, current_y),
                                             textColor=text_color,
                                             textSize=chinese_text_size)
                current_y += text_height + margin
            
            # 绘制其他信息
            for text in info_list:
                cv2.putText(frame, text,
                           (text_bg_x1 + margin, current_y + text_height),
                           font, font_scale, 
                           box_color if "ID:" in text else text_color,
                           thickness)
                current_y += text_height + margin

        return frame

    def check_overlap(self, box1, box2):
        """检查两矩形是否重叠"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 如果一个矩形在另一个形的上或方，不重叠
        if y2_1 < y1_2 or y2_2 < y1_1:
            return False
        
        # 如果一个矩形在另一个矩形左侧或右侧，则不重叠
        if x2_1 < x1_2 or x2_2 < x1_1:
            return False
        
        return True

    def load_class_names(self):
        """从模型配置文件加载类别名称"""
        try:
            # 获取模型目录路径
            model_dir = os.path.dirname(self.config['model']['path'])
            data_yaml_path = os.path.join(model_dir, 'data.yaml')
            
            # 读取data.yaml文件
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            # 获取类别名称列表
            names_dict = data_config.get('names', {})
            
            # 转换为字典格式 {index: name}
            if isinstance(names_dict, list):
                self.names = {i: name for i, name in enumerate(names_dict)}
            elif isinstance(names_dict, dict):
                self.names = {int(k): v for k, v in names_dict.items()}
            else:
                raise ValueError("不支持的类别名称格式")
            
            if not self.names:
                raise ValueError("未找到类别名称配置")
            
        except Exception as e:
            logger.error(f"加载类别名称失败: {str(e)}")
            self.names = {0: '未知类别'}

    def set_task_info(self, task_id: str, callback_func=None, callback_interval: float = 1.0):
        """设置任务信息和回调参数
        
        Args:
            task_id (str): 任务唯一标识符，用于区分不同的分析任务
            callback_func (Callable, optional): 回调函数，用于接收检测结果
                回调函数格式: async def callback(data: dict)
                data 参数包含:
                    - task_id (str): 任务ID
                    - timestamp (str): 检测时间，ISO格式
                    - detections (List[dict]): 检测结果列表，每个检测包含:
                        - track_id (int): 目标跟踪ID
                        - class (str): 目标类别
                        - confidence (float): 置信度
                        - bbox (dict): 边界框信息
                            - x1, y1 (int): 左上角坐标
                            - x2, y2 (int): 右下角坐标
                            - width, height (int): 宽度和高度
                    - image_base64 (str): 检测结果图片的base64编码
                    - image_path (str): 检测结果图片的保存路径
                    - has_detections (bool): 是否检测到目标
            callback_interval (float, optional): 回调间隔时间(秒)，默认1.0秒
                用于控制回调频率，避免回调过于频繁
        """
        self.task_id = task_id
        self.callback_func = callback_func
        self.callback_interval = callback_interval

    def stop(self):
        """停止处理"""
        if self.config['print']['enabled']:
            logger.info(f"停止任务: {self.task_id}")
        self.running = False

    def format_detections(self, results: list) -> list:
        """格式化检测结果为标准字典格式
        
        Args:
            results (list): process_frame 返回的原始检测结果列表
        
        Returns:
            list: 格式化后的检测结果列表，每个元素为字典：
                {
                    "track_id": int,      # 目标跟踪ID
                    "class": str,         # 目标类别
                    "confidence": float,  # 置信度分数
                    "bbox": {             # 边界框信息
                        "x1": int,        # 左上角x坐标
                        "y1": int,        # 左上角y坐标
                        "x2": int,        # 右下角x坐标
                        "y2": int,        # 右下角y坐标
                        "width": int,     # 宽度
                        "height": int     # 高度
                    }
                }
        """
        formatted = []
        for box, score, cls_name, track_id in results:
            x1, y1, x2, y2 = map(int, box)
            formatted.append({
                "track_id": track_id,
                "class": cls_name,
                "confidence": float(score),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })
        return formatted

    @staticmethod
    def print_help():
        """打印帮助信息"""
        print("""\n可用命令:
start       - 开始分析
stop        - 停止分析
quit/exit   - 退出程序
help        - 显示帮助信息
status      - 显示前状态
config      - 显示当前配置
""")