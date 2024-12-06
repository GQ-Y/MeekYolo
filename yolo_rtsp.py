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

class MeekYolo:
    def __init__(self, config_path='config/config.yaml'):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 初始化模型
        self.initialize_model()
        
        # 初始化输入源
        self.initialize_source()
        
        # 添加类别名称映射
        self.names = {
            0: '渣土车',
            1: '挖掘机', 
            2: '吊车'
        }
        
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

    def initialize_model(self):
        # 设置设备
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if self.config['print']['enabled']:
            print(f"使用设备: {self.device}")
        
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
            self.process_func = self.process_rtsp
            rtsp_url = self.config['source']['rtsp']['url']
            options = self.config['source']['rtsp']['ffmpeg_options']
            if options:
                rtsp_url += ''.join(options)
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise Exception("无法打开RTSP流")
                
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
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception("无法打开视频文件")
        else:
            raise ValueError(f"不支持的输入类型: {source_type}")

    def process_rtsp(self):
        """处理RTSP流"""
        fps = 0
        prev_time = time.time()
        retry_count = 0
        max_retries = 3
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if self.config['print']['enabled']:
                        print(f"无法读取视频帧,尝试重新连 ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    if retry_count >= max_retries:
                        if self.config['print']['enabled']:
                            print("重试次数超过上限,退出程序")
                        break
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.config['source']['rtsp']['url'], cv2.CAP_FFMPEG)
                    continue
                
                retry_count = 0

                # 处理帧
                results = self.process_frame(frame)
                
                # 只有在总开关和console开关都打开的情况下才打印检测结果
                if self.config['print']['enabled'] and self.config['console']['enabled']:
                    # 打印分隔线
                    if self.config['console']['show_separator']:
                        print("\n" + "="*50)
                    
                    # 打印时间
                    if self.config['console']['show_time']:
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(f"时间: {current_time}")
                    
                    # 打印总目标数
                    if self.config['console']['show_total']:
                        print(f"检测到 {len(results)} 个目标")
                    
                    # 打印详细信息
                    if self.config['console']['show_details'] and results:
                        print("\n目标详情:")
                        for i, (box, score, cls_name, track_id) in enumerate(results, 1):
                            x1, y1, x2, y2 = map(int, box)
                            width = x2 - x1
                            height = y2 - y1
                            print(f"\n目标 {i}:")
                            print(f"  跟踪ID: {track_id}")
                            print(f"  类型: {cls_name}")
                            print(f"  位置: ({x1}, {y1}) - ({x2}, {y2})")
                            print(f"  尺寸: {width}x{height} 像素")
                            print(f"  置信度: {score:.4f}")
                
                # 绘制结果
                frame = self.draw_results(frame, results)
                
                # 计算和显示FPS
                if self.config['display']['show_fps']:
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time
                    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 显示结果
                cv2.imshow(self.config['display']['window_name'], frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def process_image(self):
        """处理单张图片"""
        # 读取图片
        frame = cv2.imread(self.image_path)
        if frame is None:
            raise Exception(f"无法读取图片: {self.image_path}")
        
        # 处理图片
        results = self.process_frame(frame)
        frame = self.draw_results(frame, results)
        
        # 保存结果
        cv2.imwrite(self.save_path, frame)
        if self.config['print']['enabled']:
            print(f"结果已保存至: {self.save_path}")

    def process_images(self):
        """处理多张图片"""
        total = len(self.image_files)
        for i, image_path in enumerate(self.image_files, 1):
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                if self.config['print']['enabled']:
                    print(f"无法读取图片: {image_path}")
                continue
            
            # 处理图片
            results = self.process_frame(frame)
            frame = self.draw_results(frame, results)
            
            # 生成保存路径
            save_name = os.path.basename(image_path)
            save_path = os.path.join(self.save_dir, save_name)
            
            # 保存结果
            cv2.imwrite(save_path, frame)
            if self.config['print']['enabled']:
                print(f"处理进度: {i}/{total}, 保存: {save_path}")

    def process_video(self):
        """处理视频文件"""
        # 获取视频信息
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.save_path, fourcc, self.output_fps, (width, height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if self.config['print']['enabled']:
                    print(f"\r处理进度: {frame_count}/{total_frames}", end="")
                
                # 处理帧
                results = self.process_frame(frame)
                frame = self.draw_results(frame, results)
                
                # 写入结果
                out.write(frame)
                
                # 显示结果
                cv2.imshow(self.config['display']['window_name'], frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            if self.config['print']['enabled']:
                print(f"\n视频处理完成，结果保存至: {self.save_path}")
            self.cap.release()
            out.release()
            cv2.destroyAllWindows()

    def get_color(self, track_id):
        """为每个track_id生成唯一的颜色"""
        if track_id not in self.id_colors:
            # 生成随机颜色,但排除接近黑色和白色的颜色
            color = tuple(map(int, np.random.randint(50, 200, 3)))
            self.id_colors[track_id] = color
        return self.id_colors[track_id]

    def process_frame(self, frame):
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
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                cls_idx = int(box.cls.item())
                cls_name = self.names.get(cls_idx, '未知')
                # 根据配置决定是否获取跟踪ID
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
        在图片上添加文文本
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 创建字体对象
        font = ImageFont.truetype(self.font_path, textSize)
        
        # 创建绘制对
        draw = ImageDraw.Draw(img)
        
        # 绘制文本
        draw.text(position, text, textColor, font=font)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def draw_results(self, frame, results):
        # 获取样式配置
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
                
                # 根据配置决定是否显示锚
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
                continue  # 如果没有任何信息需要显示，直接跳过
            
            # 计算文本大小
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
            
            # 计算信息框的位置和大小 - 放置在目标框左侧
            info_height = (len(info_list) + 1) * (text_height + margin)  # +1 是为了类别信息
            text_bg_x2 = max(0, x1 - margin)  # 确保不会超出画面左边界
            text_bg_x1 = max(0, text_bg_x2 - text_width - 2 * margin)
            text_bg_y1 = max(0, y1 + (y2 - y1 - info_height) // 2)  # 垂直居中
            text_bg_y2 = min(frame.shape[0], text_bg_y1 + info_height)
            
            # 检查重叠并调整位置
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
                    text_bg_y1 = text_bg_y2 + margin  # 如果左边放不下，就放到下面
                    text_bg_y2 = text_bg_y1 + info_height
            
            info_boxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))
            
            # 绘制信息框背景
            cv2.rectangle(frame, 
                         (text_bg_x1, text_bg_y1), 
                         (text_bg_x2, text_bg_y2), 
                         bg_color, -1)
            
            # 绘制连接线
            if self.config['visualization']['show_line']:
                line_start = (text_bg_x2, text_bg_y1 + info_height // 2)
                line_end = (x1, y1 + (y2 - y1) // 2)
                cv2.line(frame, line_start, line_end, box_color, 1)
            
            # 绘制文本信息
            current_y = text_bg_y1 + margin
            
            # 绘制类别信息
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
        """检查两个矩形是否重叠"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 如果一个矩形在另一个矩形的上方或���方，则不重叠
        if y2_1 < y1_2 or y2_2 < y1_1:
            return False
        
        # 如果一个矩形在另一个矩形的左侧或右侧，则不重叠
        if x2_1 < x1_2 or x2_2 < x1_1:
            return False
        
        return True

    def run(self):
        """启动处理"""
        self.process_func() 

if __name__ == "__main__":
    detector = MeekYolo()
    detector.run() 