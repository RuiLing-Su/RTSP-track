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

## 📋 系统要求

### 硬件要求
- CPU: Intel i5 或 AMD Ryzen 5 以上
- 内存: 8GB RAM 最小，16GB 推荐
- GPU: NVIDIA GPU（可选，用于加速）
- 存储: 20GB 可用空间

### 软件依赖
- Python 3.8+
- OpenCV 5.0
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

### 3. 模型文件准备

### 4. FFmpeg配置

### 5. SRS服务器配置

项目默认推流到本地SRS服务器

### 6. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:9952` 启动

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

---
