# H-D18 检测系统

一个基于FastAPI和YOLO的智能停机坪检测系统，用于实时监控飞机停机坪的使用状态，识别飞机、工作人员和非授权人员。

## 🌟 主要功能

- **实时检测**: 使用YOLO模型检测飞机、人员和安全背心
- **多通道支持**: 支持多个RTSP视频流同时检测
- **状态监控**: 实时监控停机坪使用状态（空闲/使用中/占用）
- **人员管理**: 区分工作人员和非授权人员
- **安全告警**: 非授权人员进入使用中停机坪时发出告警
- **视频推流**: 支持RTMP推流和HTTP视频流
- **使用统计**: 记录停机坪使用时间统计

## 🏗️ 系统架构

```
├── 视频输入 (RTSP)
├── YOLO检测模型
│   ├── 飞机检测 (H-D18plane_vest.pt)
│   └── 人员检测 (person.pt)
├── 异步处理引擎
│   ├── 帧捕获循环
│   ├── 帧处理循环
│   └── 健康监控
├── 视频输出
│   ├── RTMP推流
│   └── HTTP视频流
└── REST API接口
```

## 📋 系统要求

### 硬件要求
- CPU: Intel i5 或 AMD Ryzen 5 以上
- 内存: 8GB RAM 最小，16GB 推荐
- GPU: NVIDIA GPU（可选，用于加速）
- 存储: 20GB 可用空间

### 软件依赖
- Python 3.8+
- OpenCV 4.5+
- FFmpeg（支持RTMP推流）
- CUDA（可选，GPU加速）

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/RuiLing-Su/RTSP-track.git
cd h-d18-detection

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 2. 依赖包安装

```bash
pip install fastapi uvicorn
pip install opencv-python
pip install ultralytics
pip install numpy
pip install pydantic
```

### 3. 模型文件准备

确保以下模型文件存在于项目根目录：
- `H-D18plane_vest.pt` - 飞机和安全背心检测模型
- `person.pt` - 人员检测模型

### 4. FFmpeg配置

安装FFmpeg并确保可执行：
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg

# Windows
# 下载FFmpeg并添加到PATH环境变量
```

### 5. RTMP服务器配置

项目默认推流到本地RTMP服务器，需要配置nginx-rtmp或使用其他RTMP服务器：

```nginx
# nginx.conf 示例配置
rtmp {
    server {
        listen 1935;
        chunk_size 4096;
        
        application live {
            live on;
            record off;
            
            # 允许推流
            allow publish all;
            
            # HTTP-FLV配置
            play http://localhost:18080/live/;
        }
    }
}
```

### 6. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:9952` 启动

## 📡 API接口

### 启动检测

**POST** `/start_detection`

```json
{
    "userId": 1,
    "channel": 1
}
```

**响应:**
```json
{
    "msg": "推流已启动",
    "code": 200,
    "data": {
        "code": "h264",
        "flvUrl": "http://222.186.32.142:18080/live/rtsp_1.flv",
        "message": "新通道推流已启动"
    }
}
```

### 停止检测

**POST** `/stop_detection`

```json
{
    "userId": 1,
    "channel": 1
}
```

### 获取检测结果

**GET** `/detection_result/{channel}`

**响应:**
```json
{
    "timestamp": "2024-01-01T12:00:00",
    "status_id": 1,
    "helipad_status": "使用中",
    "aircraft_count": 1,
    "aircraft_total_time": 120.5,
    "authorized_personnel": 2,
    "unauthorized_personnel": 0,
    "warning_active": false,
    "frame_base64": "base64编码的图片数据"
}
```

### 获取检测状态

**GET** `/detection_status/{channel}`

### 视频流

**GET** `/video_stream/{channel}`

返回MJPEG视频流，可直接在浏览器中查看

### 使用统计

**GET** `/usage_statistics/{channel}`

## ⚙️ 配置说明

### 通道配置

系统支持多个通道，每个通道对应一个RTSP视频源：

```python
configs = {
    1: {
        "width": 1920, 
        "height": 1080,
        "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1"
    },
    2: {
        "width": 640, 
        "height": 360,
        "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream2"
    }
}
```

### 检测区域配置

系统使用预定义的椭圆形区域作为停机坪检测范围，支持1920x1080和640x360两种分辨率。

### 状态映射

- **状态0 (空闲)**: 无飞机、无人员
- **状态1 (使用中)**: 有飞机在停机坪
- **状态2 (占用)**: 有人员但无飞机

## 🔧 高级配置

### GPU加速

系统自动检测GPU并优先使用NVIDIA硬件加速：

```python
# FFmpeg GPU加速命令
gpu_cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
    '-c:v', 'h264_nvenc', '-preset', 'p4',
    # ... 其他参数
]
```

### 性能优化

1. **双缓冲队列**: 避免帧丢失
2. **向量化计算**: 提高检测效率
3. **异步处理**: 分离帧捕获和处理
4. **线程池**: 避免阻塞事件循环

### 错误处理

- 自动RTSP重连
- FFmpeg进程监控
- 健康状态检查
- 优雅的资源清理

## 📊 监控和日志

### 系统监控

- 帧率监控
- 连接状态检查
- 资源使用统计
- 错误计数

### 日志输出

系统提供详细的控制台日志输出：

```
通道 1 连接RTSP: rtsp://admin:qaz12345@112.28.137.127:8554/Streaming/Channels/1801
通道 1 RTSP连接成功
通道 1 使用GPU加速FFmpeg
通道 1 检测启动成功
```

## 🛠️ 故障排除

### 常见问题

1. **RTSP连接失败**
   - 检查网络连接
   - 验证RTSP URL和凭据
   - 确认摄像头支持的编码格式

2. **FFmpeg启动失败**
   - 检查FFmpeg安装
   - 验证RTMP服务器配置
   - 检查端口占用情况

3. **模型加载失败**
   - 确认模型文件存在
   - 检查模型文件完整性
   - 验证YOLO版本兼容性

4. **GPU加速不可用**
   - 安装NVIDIA驱动
   - 安装CUDA toolkit
   - 检查FFmpeg NVENC支持

### 性能调优

1. **降低分辨率**: 使用640x360替代1920x1080
2. **调整帧率**: 降低处理帧率减少CPU负载
3. **优化检测参数**: 调整YOLO推理参数
4. **内存管理**: 监控内存使用避免泄漏

## 🔒 安全考虑

1. **网络安全**: 使用VPN或专网部署
2. **访问控制**: 实现用户认证和授权
3. **数据加密**: RTSP和API通信加密
4. **日志审计**: 记录关键操作日志

## 📈 扩展性

### 水平扩展

- 支持多服务器部署
- 负载均衡配置
- 分布式处理

### 功能扩展

- 支持更多检测对象
- 添加数据库存储
- 集成报警系统
- Web管理界面

## 📞 支持与维护

### 系统要求检查

运行前请确保：
- [ ] Python 3.8+ 已安装
- [ ] 所有依赖包已安装
- [ ] 模型文件已准备
- [ ] FFmpeg 已配置
- [ ] RTMP服务器已启动
- [ ] 网络连接正常

### 联系方式

如有问题或建议，请联系技术支持团队。

---

**版本**: 1.0.0  
**最后更新**: 2024年1月  
**许可证**: 请联系相关方获取许可信息
