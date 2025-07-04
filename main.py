from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict
import asyncio
import base64
from typing import Optional
from datetime import datetime
import subprocess
import os
import signal
from functools import reduce
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="H-D18检测", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


# 数据模型
class StartDetectionRequest(BaseModel):
    userId: int
    channel: int


class StopDetectionRequest(BaseModel):
    userId: int
    channel: int


class DetectionResult(BaseModel):
    timestamp: str
    status_id: int
    helipad_status: str
    aircraft_count: int
    aircraft_total_time: float
    authorized_personnel: int
    unauthorized_personnel: int
    warning_active: bool
    frame_base64: Optional[str] = None


class AsyncChannelDetection:
    def __init__(self, channel: str):
        self.channel = channel
        self.is_running = False
        self.users = set()
        self.models = {}
        self.latest_result = None

        # 双缓冲队列
        self.frame_queue = asyncio.Queue(maxsize=2)
        self.result_queue = asyncio.Queue(maxsize=1)

        # 状态跟踪 - 简化为字典
        self.state = {
            'field_occupied': False, 'field_start_time': 0, 'total_occupied_time': 0,
            'hourly_usage': defaultdict(float), 'last_status': 0, 'last_time': None,
            'warning_active': False, 'pre_restart_occupied': False
        }

        # 资源管理
        self.cap = None
        self.ffmpeg_process = None
        self.tasks = []
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 配置 - 使用字典简化
        self.configs = {
            1: {"width": 1920, "height": 1080,
                "rtsp_url": "rtsp://your_username:your_password@123.123.123.123:8554/Streaming/Channels/1801"},
            2: {"width": 640, "height": 360,
                "rtsp_url": "rtsp://your_username:your_password@123.123.123.123:8554/Streaming/Channels/1802"}
        }

        # 多边形点 - 预计算缩放
        self.ellipse_points_1920 = np.array([
            (739, 389), (694, 400), (646, 411), (591, 423), (566, 433), (543, 442),
            (514, 452), (467, 469), (431, 487), (397, 504), (375, 519), (345, 542),
            (323, 561), (308, 590), (303, 619), (309, 640), (326, 661), (345, 676),
            (370, 690), (391, 702), (429, 715), (461, 720), (499, 733), (547, 740),
            (584, 744), (645, 754), (697, 754), (748, 754), (781, 757), (833, 758),
            (893, 752), (955, 744), (1018, 742), (1085, 734), (1125, 724), (1157, 724),
            (1228, 712), (1262, 703), (1300, 694), (1341, 681), (1380, 670), (1420, 660),
            (1463, 647), (1497, 632), (1540, 612), (1575, 595), (1626, 567), (1652, 537),
            (1677, 506), (1683, 477), (1676, 446), (1653, 420), (1629, 404), (1596, 389),
            (1555, 374), (1505, 361), (1451, 350), (1411, 347), (1347, 341), (1290, 336),
            (1224, 338), (1157, 338), (1097, 336), (1024, 343), (959, 350), (914, 352),
            (855, 366), (799, 375)
        ], dtype=np.int32)

        # 预计算640分辨率的点
        self.ellipse_points_640 = (self.ellipse_points_1920 * np.array([640 / 1920, 360 / 1080])).astype(np.int32)
        self.ellipse_mask = None

    def add_user(self, user_id: int):
        self.users.add(user_id)

    def remove_user(self, user_id: int):
        self.users.discard(user_id)
        return len(self.users) == 0

    def get_config(self):
        return self.configs.get(int(self.channel[-2:]), self.configs[2])

    async def start_ffmpeg_stream(self):
        """使用GPU加速的FFmpeg推流"""
        stream_key = f"rtsp_{self.channel}"
        rtmp_url = f"rtmp://localhost:1935/live/{stream_key}"
        config = self.get_config()

        # 优先尝试GPU加速，失败则使用CPU

        gpu_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f"{config['width']}x{config['height']}", '-r', '20', '-i', '-',
            '-c:v', 'h264_nvenc','-preset', 'p4',
            '-rc', 'cbr', '-maxrate', '1500k', '-bufsize', '3000k',
            '-pix_fmt', 'yuv420p', '-g', '40', '-f', 'flv', rtmp_url
        ]
        cpu_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f"{config['width']}x{config['height']}", '-r', '20', '-i', '-',
            '-c:v', 'libx264', '-preset', 'superfast', '-tune', 'zerolatency',
            '-crf', '30', '-maxrate', '1000k', '-bufsize', '2000k',
            '-pix_fmt', 'yuv420p', '-g', '40', '-f', 'flv', rtmp_url
        ]

        try:
            await self.stop_ffmpeg_stream()

            # 先尝试GPU加速
            try:
                self.ffmpeg_process = await asyncio.create_subprocess_exec(
                    *gpu_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                await asyncio.sleep(0.3)
                if self.ffmpeg_process.returncode is None:
                    print(f"通道 {self.channel} 使用GPU加速FFmpeg")
                    return True
            except Exception:
                print(f"通道 {self.channel} GPU加速失败，尝试CPU模式")

            # 回退到CPU模式
            self.ffmpeg_process = await asyncio.create_subprocess_exec(
                *cpu_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            await asyncio.sleep(0.3)
            if self.ffmpeg_process.returncode is None:
                print(f"通道 {self.channel} 使用CPU FFmpeg")
                return True
            else:
                self.ffmpeg_process = None
                return False

        except Exception as e:
            print(f"FFmpeg启动失败: {e}")
            self.ffmpeg_process = None
            return False

    async def stop_ffmpeg_stream(self):
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                await asyncio.wait_for(self.ffmpeg_process.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.ffmpeg_process.kill()
            finally:
                self.ffmpeg_process = None

    def create_ellipse_mask(self, shape, points):
        """向量化创建多边形遮罩"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return mask

    def vectorized_point_in_region(self, points):
        """向量化的区域检测"""
        if self.ellipse_mask is None:
            return np.array([False] * len(points))

        # 向量化坐标检查
        valid_mask = ((points[:, 0] >= 0) & (points[:, 0] < self.ellipse_mask.shape[1]) &
                      (points[:, 1] >= 0) & (points[:, 1] < self.ellipse_mask.shape[0]))

        result = np.zeros(len(points), dtype=bool)
        if valid_mask.any():
            valid_points = points[valid_mask].astype(int)
            result[valid_mask] = self.ellipse_mask[valid_points[:, 1], valid_points[:, 0]] > 0

        return result

    def vectorized_iou(self, boxes1, boxes2):
        """向量化的IoU计算"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))

        boxes1 = np.array(boxes1).reshape(-1, 4)
        boxes2 = np.array(boxes2).reshape(-1, 4)

        # 广播计算所有组合的IoU
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - intersection

        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)

    async def reconnect_rtsp(self):
        """异步RTSP重连"""
        try:
            if self.cap:
                self.cap.release()
                await asyncio.sleep(0.2)

            config = self.get_config()
            print(f"通道 {self.channel} 连接RTSP: {config['rtsp_url']}")

            # 使用标准连接方式，移除GPU特定设置避免兼容性问题
            self.cap = cv2.VideoCapture(config["rtsp_url"], cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 20)

            if self.cap.isOpened():
                # 选择预计算的多边形点
                w, h = config["width"], config["height"]
                points = self.ellipse_points_640 if w == 640 else self.ellipse_points_1920
                self.ellipse_mask = self.create_ellipse_mask((h, w), points)

                # 测试读取几帧
                for i in range(5):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"通道 {self.channel} RTSP连接成功")
                        return True
                    await asyncio.sleep(0.1)

                print(f"通道 {self.channel} 无法读取有效帧")
                return False
            else:
                print(f"通道 {self.channel} RTSP连接失败")
                return False

        except Exception as e:
            print(f"通道 {self.channel} RTSP重连异常: {e}")
            return False

    async def frame_capture_loop(self):
        """异步帧捕获循环 - 生产者"""
        consecutive_failures = 0

        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    consecutive_failures += 1
                    print(f"通道 {self.channel} 连接失败 {consecutive_failures}/10")
                    if consecutive_failures > 10:
                        print(f"通道 {self.channel} 尝试重连RTSP...")
                        await self.reconnect_rtsp()
                        consecutive_failures = 0
                    await asyncio.sleep(1)
                    continue

                # 在线程池中读取帧，避免阻塞事件循环
                loop = asyncio.get_event_loop()
                success, frame = await loop.run_in_executor(self.executor, self.cap.read)

                if not success or frame is None:
                    consecutive_failures += 1
                    print(f"通道 {self.channel} 读取帧失败 {consecutive_failures}/10")
                    if consecutive_failures > 10:
                        print(f"通道 {self.channel} 尝试重连RTSP...")
                        await self.reconnect_rtsp()
                        consecutive_failures = 0
                    await asyncio.sleep(0.1)
                    continue

                consecutive_failures = 0

                # 双缓冲 - 非阻塞放入队列
                try:
                    self.frame_queue.put_nowait((frame.copy(), time.time()))
                except asyncio.QueueFull:
                    try:
                        self.frame_queue.get_nowait()  # 丢弃旧帧
                        self.frame_queue.put_nowait((frame.copy(), time.time()))
                    except asyncio.QueueEmpty:
                        pass

            except Exception as e:
                print(f"通道 {self.channel} 帧捕获异常: {e}")
                consecutive_failures += 1
                await asyncio.sleep(0.1)

            await asyncio.sleep(0.05)  # 控制帧率

    async def frame_process_loop(self):
        """异步帧处理循环 - 消费者"""
        while self.is_running:
            try:
                frame, timestamp = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)

                # 在线程池中处理帧
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.process_frame_optimized, frame, timestamp
                )

                # 更新最新结果
                try:
                    self.result_queue.put_nowait(result)
                except asyncio.QueueFull:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except asyncio.QueueEmpty:
                        pass

                self.latest_result = result

                # 异步推流
                if self.ffmpeg_process and self.ffmpeg_process.stdin:
                    try:
                        self.ffmpeg_process.stdin.write(frame.tobytes())
                        await self.ffmpeg_process.stdin.drain()
                    except:
                        pass

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"帧处理异常: {e}")
                await asyncio.sleep(0.1)

    def process_frame_optimized(self, frame, current_time):
        """优化的帧处理 - 使用向量化操作"""
        h, w = frame.shape[:2]

        # 绘制多边形边界
        points = self.ellipse_points_640 if w == 640 else self.ellipse_points_1920
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)

        # 并行检测
        aircraft_results = self.models['aircraft'].track(frame, persist=True, verbose=False)
        person_results = self.models['person'].track(frame, persist=True, verbose=False)

        # 向量化处理检测结果
        aircrafts_in_area, vest_boxes = self.process_aircraft_detections(frame, aircraft_results)
        authorized_personnel, unauthorized_personnel = self.process_person_detections(
            frame, person_results, vest_boxes
        )

        # 状态计算和更新
        status_info = self.calculate_status(aircrafts_in_area, authorized_personnel, unauthorized_personnel)
        self.update_state(status_info['status_id'], aircrafts_in_area, current_time)

        # 绘制UI
        self.draw_ui(frame, status_info, authorized_personnel, unauthorized_personnel)

        # 编码
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return DetectionResult(
            timestamp=datetime.now().isoformat(),
            status_id=status_info['status_id'],
            helipad_status=status_info['helipad_status'],
            aircraft_count=aircrafts_in_area,
            aircraft_total_time=self.get_current_total_time(current_time),
            authorized_personnel=authorized_personnel,
            unauthorized_personnel=unauthorized_personnel,
            warning_active=self.state['warning_active'],
            frame_base64=frame_base64
        )

    def process_aircraft_detections(self, frame, aircraft_results):
        """向量化处理飞机检测"""
        if not aircraft_results or aircraft_results[0].boxes is None:
            return 0, []

        boxes = aircraft_results[0].boxes.xyxy.cpu().numpy()
        classes = aircraft_results[0].boxes.cls.cpu().numpy()

        # 分离飞机和背心
        aircraft_mask = classes == 0
        vest_mask = classes == 1

        aircraft_boxes = boxes[aircraft_mask] if aircraft_mask.any() else np.array([])
        vest_boxes = boxes[vest_mask] if vest_mask.any() else np.array([])

        aircrafts_in_area = 0

        # 向量化处理飞机
        if len(aircraft_boxes) > 0:
            centers = (aircraft_boxes[:, :2] + aircraft_boxes[:, 2:]) / 2
            in_area_mask = self.vectorized_point_in_region(centers)
            aircrafts_in_area = in_area_mask.sum()

            # 批量绘制
            for box, in_area in zip(aircraft_boxes, in_area_mask):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if in_area else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if in_area:
                    cv2.putText(frame, "EH216-s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制背心
        for box in vest_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "安全背心", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return aircrafts_in_area, vest_boxes.tolist()

    def process_person_detections(self, frame, person_results, vest_boxes):
        """向量化处理人员检测"""
        if not person_results or person_results[0].boxes is None:
            return 0, 0

        person_boxes = person_results[0].boxes.xyxy.cpu().numpy()
        if len(person_boxes) == 0:
            return 0, 0

        # 向量化区域检测
        centers = (person_boxes[:, :2] + person_boxes[:, 2:]) / 2
        in_area_mask = self.vectorized_point_in_region(centers)

        authorized_personnel = 0
        unauthorized_personnel = 0

        # 向量化IoU计算
        if len(vest_boxes) > 0 and in_area_mask.any():
            area_person_boxes = person_boxes[in_area_mask]
            iou_matrix = self.vectorized_iou(area_person_boxes, vest_boxes)
            has_vest_mask = (iou_matrix > 0.1).any(axis=1)

            authorized_personnel = has_vest_mask.sum()
            unauthorized_personnel = len(area_person_boxes) - authorized_personnel
        else:
            unauthorized_personnel = in_area_mask.sum()

        # 批量绘制
        for box, in_area in zip(person_boxes, in_area_mask):
            x1, y1, x2, y2 = map(int, box)
            if in_area:
                is_authorized = False
                if len(vest_boxes) > 0:
                    ious = self.vectorized_iou([box], vest_boxes)
                    is_authorized = (ious > 0.1).any()

                color = (0, 255, 0) if is_authorized else (0, 0, 255)
                text = "工作人员" if is_authorized else "非工作人员!"
                thickness = 2 if is_authorized else 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "人员", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return authorized_personnel, unauthorized_personnel

    def calculate_status(self, aircrafts_in_area, authorized_personnel, unauthorized_personnel):
        """计算状态信息"""
        status_map = {
            (True, False, False): (1, "使用中", (0, 255, 0)),
            (False, True, False): (2, "占用", (0, 165, 255)),
            (False, False, True): (2, "占用", (0, 165, 255)),
            (False, False, False): (0, "空闲", (255, 255, 255))
        }

        key = (aircrafts_in_area > 0, authorized_personnel > 0, unauthorized_personnel > 0)
        # 查找最匹配的状态
        for pattern, status_info in status_map.items():
            if all(k <= p for k, p in zip(pattern, key)):
                status_id, helipad_status, status_color = status_info
                break
        else:
            status_id, helipad_status, status_color = status_map[(False, False, False)]

        # 更新告警状态
        should_warn = aircrafts_in_area > 0 and unauthorized_personnel > 0
        self.state['warning_active'] = should_warn

        return {
            'status_id': status_id,
            'helipad_status': helipad_status,
            'status_color': status_color
        }

    def update_state(self, status_id, aircrafts_in_area, current_time):
        """更新状态统计"""
        # 更新使用统计
        now = datetime.fromtimestamp(current_time)
        if self.state['last_time'] and status_id == 1 and 9 <= now.hour < 18:
            duration = (now - self.state['last_time']).total_seconds()
            self.state['hourly_usage'][now.hour] += duration

        # 更新占用时间
        if aircrafts_in_area > 0 and not self.state['field_occupied']:
            self.state['field_occupied'] = True
            self.state['field_start_time'] = current_time
        elif aircrafts_in_area == 0 and self.state['field_occupied']:
            self.state['total_occupied_time'] += current_time - self.state['field_start_time']
            self.state['field_occupied'] = False

        self.state['last_status'] = status_id
        self.state['last_time'] = now

    def get_current_total_time(self, current_time):
        """获取当前总占用时间"""
        total = self.state['total_occupied_time']
        if self.state['field_occupied']:
            total += current_time - self.state['field_start_time']
        return total

    def draw_ui(self, frame, status_info, authorized_personnel, unauthorized_personnel):
        """绘制UI界面 - 根据分辨率自适应调整"""
        h, w = frame.shape[:2]

        # 基于1920*1080的缩放因子
        scale_factor = min(w / 1920, h / 1080)

        # 动态计算尺寸
        alert_height = int(100 * scale_factor)
        info_width = int(320 * scale_factor)
        info_height = int(100 * scale_factor)
        margin = int(10 * scale_factor)

        # 动态字体大小
        alert_font_scale = max(0.5, 1.0 * scale_factor)
        info_font_scale = max(0.4, 0.7 * scale_factor)
        small_font_scale = max(0.3, 0.6 * scale_factor)

        # 动态文字位置偏移
        alert_text_x = int(20 * scale_factor)
        alert_text_y = int(alert_height * 0.6)
        info_text_offset = int(20 * scale_factor)
        line_spacing = int(30 * scale_factor)

        # 告警显示
        if self.state['warning_active']:
            cv2.rectangle(frame, (0, 0), (w, alert_height), (0, 0, 255), -1)
            cv2.putText(frame, "危险：非工作人员进入使用中的停机坪!", (alert_text_x, alert_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, alert_font_scale, (255, 255, 255), max(1, int(3 * scale_factor)))

        # 状态信息
        info_y_start = alert_height + int(50 * scale_factor) if self.state['warning_active'] else int(50 * scale_factor)
        cv2.rectangle(frame, (w - info_width - margin, info_y_start - int(30 * scale_factor)),
                      (w - margin, info_y_start + info_height), (0, 0, 0), -1)

        texts = [
            f"停机坪状态: {status_info['helipad_status']}",
            f"占用时间: {self.get_current_total_time(time.time()):.1f}s",
            f"工作人员: {authorized_personnel} | 非工作人员: {unauthorized_personnel}"
        ]

        colors = [status_info['status_color'], (255, 255, 255), (255, 255, 255)]
        scales = [info_font_scale, info_font_scale, small_font_scale]

        for i, (text, color, scale) in enumerate(zip(texts, colors, scales)):
            cv2.putText(frame, text, (w - info_width + info_text_offset, info_y_start + int(10 * scale_factor) + i * line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, int(2 * scale_factor)))

    async def start_detection(self, aircraft_model_path: str, person_model_path: str):
        """启动检测"""
        if self.is_running:
            print(f"通道 {self.channel} 已在运行")
            return True

        try:
            print(f"通道 {self.channel} 开始启动检测...")

            # 检查模型文件
            if not os.path.exists(aircraft_model_path):
                print(f"飞机模型文件不存在: {aircraft_model_path}")
                return False
            if not os.path.exists(person_model_path):
                print(f"人员模型文件不存在: {person_model_path}")
                return False

            # 加载模型
            print(f"通道 {self.channel} 加载模型...")
            self.models = {
                'aircraft': YOLO(aircraft_model_path),
                'person': YOLO(person_model_path)
            }
            print(f"通道 {self.channel} 模型加载成功")

            # 连接RTSP
            print(f"通道 {self.channel} 连接RTSP...")
            if not await self.reconnect_rtsp():
                print(f"通道 {self.channel} RTSP连接失败")
                return False

            # 启动FFmpeg
            print(f"通道 {self.channel} 启动FFmpeg...")
            if not await self.start_ffmpeg_stream():
                print(f"通道 {self.channel} FFmpeg启动失败")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False

            self.is_running = True
            print(f"通道 {self.channel} 启动异步任务...")

            # 启动异步任务
            self.tasks = [
                asyncio.create_task(self.frame_capture_loop()),
                asyncio.create_task(self.frame_process_loop()),
                asyncio.create_task(self.health_monitor())
            ]

            print(f"通道 {self.channel} 检测启动成功")
            return True

        except Exception as e:
            print(f"通道 {self.channel} 启动检测失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def health_monitor(self):
        """健康监控"""
        last_restart = time.time()

        while self.is_running:
            try:
                current_time = time.time()

                # 每3分钟重启RTSP
                if current_time - last_restart > 180:
                    print(f"通道 {self.channel} 定期重启RTSP")
                    await self.reconnect_rtsp()
                    last_restart = current_time

                await asyncio.sleep(30)  # 30秒检查一次

            except Exception as e:
                print(f"健康监控异常: {e}")
                await asyncio.sleep(10)

    async def stop_detection(self):
        """停止检测"""
        print(f"正在停止通道 {self.channel}...")

        self.is_running = False

        # 取消所有任务
        for task in self.tasks:
            task.cancel()

        # 等待任务结束
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # 清理资源
        await self.stop_ffmpeg_stream()

        if self.cap:
            self.cap.release()
            self.cap = None

        # 重置状态
        self.state.update({
            'warning_active': False, 'field_occupied': False, 'field_start_time': 0
        })
        self.latest_result = None
        self.ellipse_mask = None

        print(f"通道 {self.channel} 已停止")

    def get_latest_result(self):
        return self.latest_result

    def get_hourly_usage(self):
        current_hour = min(datetime.now().hour, 17)
        return [self.state['hourly_usage'].get(h, 0.0) for h in range(9, current_hour)]


class OptimizedDetectionSystem:
    def __init__(self):
        self.channels = {}
        self.lock = asyncio.Lock()

    async def start_detection(self, user_id: int, channel: int, aircraft_model_path: str, person_model_path: str):
        async with self.lock:
            if channel not in self.channels:
                self.channels[channel] = AsyncChannelDetection(str(channel))

            channel_system = self.channels[channel]
            channel_system.add_user(user_id)

            if not channel_system.is_running:
                success = await channel_system.start_detection(aircraft_model_path, person_model_path)
                if not success:
                    channel_system.remove_user(user_id)
                    if len(channel_system.users) == 0:
                        del self.channels[channel]
                    return False, "通道启动失败"
                return True, "新通道推流已启动"
            else:
                return True, "已加入现有通道"

    async def stop_detection(self, user_id: int, channel: int):
        async with self.lock:
            if channel not in self.channels:
                return False, "通道不存在"

            channel_system = self.channels[channel]
            should_stop = channel_system.remove_user(user_id)

            if should_stop:
                await channel_system.stop_detection()
                del self.channels[channel]
                return True, "推流已停止"
            else:
                return False, "还有其他用户在使用通道"

    def get_channel_result(self, channel: int):
        if channel in self.channels:
            return self.channels[channel].get_latest_result()
        return None

    def is_channel_running(self, channel: int):
        return channel in self.channels and self.channels[channel].is_running


# 全局实例
detection_system = OptimizedDetectionSystem()


# API端点
@app.post("/start_detection")
async def start_detection(request: StartDetectionRequest):
    success, message = await detection_system.start_detection(
        request.userId, request.channel, "H-D18plane_vest.pt", "person.pt"
    )

    if success:
        stream_key = f"rtsp_{request.channel}"
        flv_url = f"http://222.186.32.142:18080/live/{stream_key}.flv"
        return {
            "msg": "推流已启动", "code": 200,
            "data": {"code": "h264", "flvUrl": flv_url, "message": message}
        }
    else:
        return {"msg": "启动失败", "code": 400, "data": {"message": message}}


@app.post("/stop_detection")
async def stop_detection(request: StopDetectionRequest):
    success, message = await detection_system.stop_detection(request.userId, request.channel)
    return {
        "msg": "推流已停止" if success else "还有其他用户在使用通道",
        "code": 200, "data": None
    }


@app.get("/detection_result/{channel}")
async def get_detection_result(channel: int):
    result = detection_system.get_channel_result(channel)
    if result:
        return result
    else:
        raise HTTPException(status_code=404, detail="暂无检测结果")


@app.get("/detection_status/{channel}")
async def get_detection_status(channel: int):
    is_running = detection_system.is_channel_running(channel)
    return {"is_running": is_running, "timestamp": datetime.now().isoformat()}


async def generate_video_stream(channel: int):
    consecutive_failures = 0
    while detection_system.is_channel_running(channel):
        try:
            result = detection_system.get_channel_result(channel)
            if result and result.frame_base64:
                img_data = base64.b64decode(result.frame_base64)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img_data + b'\r\n')
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures > 100:
                    break
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures > 100:
                break
        await asyncio.sleep(0.05)


@app.get("/video_stream/{channel}")
async def video_stream(channel: int):
    if not detection_system.is_channel_running(channel):
        raise HTTPException(status_code=400, detail="通道检测未启动")
    return StreamingResponse(generate_video_stream(channel), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/usage_statistics/{channel}")
async def get_usage_statistics(channel: int):
    if channel not in detection_system.channels:
        raise HTTPException(status_code=404, detail="通道不存在")

    channel_system = detection_system.channels[channel]
    hourly_usage = channel_system.get_hourly_usage()
    daily_rate = sum(hourly_usage) / (8 * 3600) if hourly_usage else 0

    return {
        "hourly_usage_seconds": hourly_usage,
        "daily_usage_rate": daily_rate,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    return {"message": "H-D18检测API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9952)
