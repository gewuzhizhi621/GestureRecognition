<div align="center">
🖐️ GestureRecognition
基于 YOLOv12 的实时手势识别系统
一个面向课程设计 / 毕业项目 / 计算机视觉入门实践的手势识别项目，支持摄像头实时检测、模型加载、置信度调节、检测结果显示与历史记录展示。
<br>
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv12-Object%20Detection-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Real--time%20Camera-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-FFB000?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
</div>
---
📌 项目简介
`GestureRecognition` 是一个基于 YOLOv12 的实时手势识别系统。项目使用自定义手势数据集训练目标检测模型，并通过 Python + OpenCV + Tkinter 构建可视化检测界面，实现从摄像头画面中实时识别手势类别。
本项目目前支持以下 5 类手势识别：
类别编号	英文类别	中文说明
0	`rock`	石头
1	`paper`	布
2	`scissors`	剪刀
3	`OK`	OK 手势
4	`good`	点赞手势
---
✨ 项目亮点
🎯 基于 YOLOv12：使用 YOLO 系列目标检测模型完成手势检测任务
📷 摄像头实时识别：支持调用本地摄像头进行实时手势识别
🖥️ 图形化操作界面：基于 Tkinter 构建桌面端可视化检测系统
🎚️ 参数可调节：支持调整置信度阈值与 NMS 阈值
📊 检测结果展示：实时显示当前识别手势与置信度
🕒 历史记录显示：保存最近的检测结果，方便观察识别过程
🗂️ 工程结构清晰：包含数据集处理、模型评估、摄像头检测等独立脚本
🚀 适合扩展：可继续扩展更多手势类别、语音反馈、动作控制、人机交互等功能
---
🧠 系统功能
功能模块	功能说明
模型加载	加载训练完成的 YOLO 手势识别模型
实时检测	调用摄像头获取视频流并进行实时推理
结果显示	显示当前识别到的手势类别和置信度
历史记录	记录最近多次检测结果
参数调节	支持调整 Confidence 和 NMS 阈值
数据整理	支持图像、标签文件复制与数据集划分
模型评估	支持对测试集进行模型验证与性能评估
---
🧩 技术栈
技术	作用
Python	项目主要开发语言
YOLOv12 / Ultralytics	手势目标检测模型训练与推理
OpenCV	摄像头调用、图像读取与视频帧处理
Tkinter	桌面端 GUI 界面开发
Pillow	图像格式转换与界面图像显示
PyTorch	深度学习模型运行环境
tqdm	数据集处理进度显示
---
📁 项目结构
```text
GestureRecognition/
├── configs/
│   └── gesture_dataset.yaml          # YOLO 数据集配置文件
│
├── datasets/
│   ├── raw/                          # 原始图像和标签数据
│   │   ├── images/
│   │   └── labels/
│   └── gesture/                      # 划分后的 YOLO 数据集
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
│
├── models/
│   └── yolov12n.pt                   # YOLOv12 预训练模型
│
├── runs/
│   └── training/
│       └── train5/
│           └── weights/
│               └── best.pt           # 训练完成后的最佳模型权重
│
├── scripts/
│   ├── copy_images.py                # 复制图像文件到 raw/images
│   ├── copy_labels.py                # 复制标签文件到 raw/labels
│   ├── split_dataset.py              # 划分训练集和验证集
│   ├── detect_camera.py              # OpenCV 摄像头检测脚本
│   └── evaluate.py                   # 模型评估脚本
│
├── src/
│   └── main.py                       # 图形化手势识别系统主程序
│
├── third_party/
│   └── yolov12/                      # YOLOv12 第三方源码或本地依赖
│
├── requirements.txt                  # Python 依赖列表
├── .gitignore                        # Git 忽略文件配置
└── README.md                         # 项目说明文档
```
---
🚀 快速开始
1. 克隆项目
```bash
git clone https://github.com/gewuzhizhi621/GestureRecognition.git
cd GestureRecognition
```
2. 创建虚拟环境
Windows：
```bash
python -m venv .venv
.venv\Scripts\activate
```
macOS / Linux：
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. 安装依赖
```bash
pip install -r requirements.txt
```
如果安装 PyTorch 速度较慢，建议根据自己的 CUDA / CPU 环境到 PyTorch 官网选择对应安装命令。
---
▶️ 运行项目
方式一：运行图形化界面
```bash
python src/main.py
```
运行后点击界面中的：
```text
Load Model → Start Detection
```
即可启动摄像头实时识别。
---
方式二：运行简单摄像头检测脚本
```bash
python scripts/detect_camera.py
```
按下键盘 `q` 可以退出检测窗口。
---
🧪 数据集配置
数据集配置文件位于：
```text
configs/gesture_dataset.yaml
```
内容示例：
```yaml
path: datasets/gesture
train: train/images
val: val/images
test: test/images
nc: 5
names: ["rock", "paper", "scissors", "OK", "good"]
```
YOLO 标签文件格式如下：
```text
class_id x_center y_center width height
```
其中坐标均为归一化后的相对坐标，取值范围为 `0 ~ 1`。
---
🗃️ 数据集整理流程
1. 准备原始数据
将原始图片放入：
```text
datasets/raw/images/
```
将 YOLO 格式标签文件放入：
```text
datasets/raw/labels/
```
图片文件和标签文件需要同名，例如：
```text
0001.jpg
0001.txt
```
---
2. 划分训练集和验证集
```bash
python scripts/split_dataset.py
```
默认按照 `8:2` 划分训练集和验证集。
划分完成后会生成：
```text
datasets/gesture/train/
datasets/gesture/val/
```
---
🏋️ 模型训练
可以使用如下命令训练模型：
```bash
yolo detect train model=models/yolov12n.pt data=configs/gesture_dataset.yaml imgsz=640 epochs=100 batch=16 project=runs/training name=train5
```
训练完成后，最佳模型权重通常保存在：
```text
runs/training/train5/weights/best.pt
```
本项目图形化界面默认加载的模型路径也是：
```text
runs/training/train5/weights/best.pt
```
---
📈 模型评估
运行评估脚本：
```bash
python scripts/evaluate.py
```
评估结果会输出到：
```text
runs/evaluation/
```
常见评估结果包括：
文件	说明
`confusion_matrix.png`	混淆矩阵
`F1_curve.png`	F1 曲线
`P_curve.png`	Precision 曲线
`R_curve.png`	Recall 曲线
`PR_curve.png`	Precision-Recall 曲线
`results.png`	训练结果汇总图
`results.csv`	训练指标数据
---
🖥️ 界面说明
图形化界面主要包含以下区域：
区域	说明
左侧控制栏	加载模型、启动检测、选择摄像头、调节阈值
实时画面区	显示摄像头实时画面
当前结果区	显示识别到的手势类别和置信度
历史记录区	显示最近的检测记录
---
📷 运行效果展示
建议在项目中新增以下目录用于保存截图和演示动图：
```text
docs/images/
docs/demo.gif
```
推荐截图命名：
```text
docs/images/main_window.png
docs/images/camera_detection.png
docs/images/training_result.png
```
如果你已经添加截图，可以在 README 中加入：
```markdown
![Main Window](docs/images/main_window.png)
```
---
⚙️ 主要文件说明
文件 / 文件夹	说明
`src/main.py`	项目主程序，包含 Tkinter 图形界面和实时识别逻辑
`scripts/detect_camera.py`	简单摄像头实时检测脚本
`scripts/evaluate.py`	模型评估脚本
`scripts/split_dataset.py`	数据集划分脚本
`scripts/copy_images.py`	批量复制图像文件
`scripts/copy_labels.py`	批量复制标签文件
`configs/gesture_dataset.yaml`	YOLO 数据集配置文件
`models/yolov12n.pt`	YOLOv12 预训练模型
`runs/training/train5/weights/best.pt`	项目训练得到的最佳模型
`requirements.txt`	Python 依赖文件
---
🧱 核心流程
```text
采集手势图片
      ↓
标注 YOLO 格式标签
      ↓
整理 images / labels
      ↓
划分 train / val / test 数据集
      ↓
使用 YOLOv12 训练模型
      ↓
生成 best.pt 权重文件
      ↓
加载模型并进行摄像头实时识别
      ↓
在 GUI 界面中显示检测结果
```
---
📝 GitHub 上传建议
推荐上传
```text
README.md
requirements.txt
.gitignore
configs/
src/
scripts/
models/yolov12n.pt
runs/training/train5/weights/best.pt
```
可选上传
```text
docs/images/
docs/demo.gif
third_party/yolov12/
```
不建议上传
```text
.venv/
.vscode/
.git/
__pycache__/
datasets/raw/
datasets/gesture/
runs/training/train/
runs/training/train6/
runs/training/train9/
runs/training/train10/
runs/evaluation/
third_party/yolov12/.git/
```
说明：数据集和训练过程文件通常较大，建议只上传核心代码、配置文件和最终模型权重。
---
❗ 常见问题
1. 提示找不到模型文件怎么办？
请确认模型文件是否存在：
```text
runs/training/train5/weights/best.pt
```
如果你的模型保存在其他位置，需要修改 `src/main.py` 中的：
```python
MODEL_PATH = PROJECT_ROOT / "runs" / "training" / "train5" / "weights" / "best.pt"
```
---
2. 摄像头打不开怎么办？
可以尝试：
确认摄像头没有被其他软件占用
将摄像头编号从 `0` 改为 `1` 或 `2`
检查系统是否允许 Python 访问摄像头
---
3. 识别类别显示不正确怎么办？
请保证以下三个地方的类别顺序完全一致：
```text
数据集标注类别顺序
configs/gesture_dataset.yaml
src/main.py 中的 GESTURE_CLASSES
```
推荐统一为：
```python
GESTURE_CLASSES = ["rock", "paper", "scissors", "OK", "good"]
```
---
4. 没有 GPU 可以运行吗？
可以。本项目默认使用 CPU 推理：
```python
device="cpu"
```
如果电脑支持 CUDA，可以根据环境修改为 GPU 推理。
---
🔮 后续可扩展方向
增加更多手势类别
增加手势控制电脑功能
增加语音播报识别结果
增加识别结果统计图表
将 Tkinter 界面升级为 PyQt / Web 界面
部署到树莓派、Jetson Nano 等边缘设备
将模型导出为 ONNX，用于更轻量化推理
---
👤 作者
项目	信息
Author	gewuzhizhi621
Repository	`GestureRecognition`
Direction	Computer Vision / Object Detection
---
📄 License
本项目仅用于学习、课程设计和计算机视觉实验研究。  
如需用于商业用途，请自行确认所使用数据集、模型权重和第三方依赖的授权协议。
---
<div align="center">
⭐ 如果这个项目对你有帮助，欢迎 Star！
GestureRecognition · YOLOv12 Real-time Hand Gesture Detection System
</div>
