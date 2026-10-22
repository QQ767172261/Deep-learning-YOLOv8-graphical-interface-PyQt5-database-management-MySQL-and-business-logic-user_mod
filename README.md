深度学习-基于YOLOv8的手势识别系统v2版本，新增了四个核心功能模块：角色分配管理、模型版本管理、核心推理参数配置、历史记录管理。
包含完整数据集+训练日志+可视化图+混淆矩阵+ui界面。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/927baff2683b4b659b05bef9772cdf47.png)

1、功能特性：
手势识别：支持8种常见手势的实时识别（比耶、竖3手指、竖5手指、竖4手指、OK、竖食指、握拳、竖大拇指）![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bb8a6fa9e40149878a89a3d226ac97b1.png)

多输入模式：支持图片文件、视频文件和实时摄像头三种输入方式
用户与角色管理：支持注册、登录、密码修改，提供管理员 / 普通用户角色划分
管理员后台：管理员可通过可视化后台对用户进行增删改查和密码重置
模型版本管理：支持多版本 YOLO 模型上传、版本列表查看、模型一键切换
推理参数在线配置：图形化界面实时调整置信度阈值、NMS 阈值、最大检测数、图像尺寸等关键参数
历史记录与统计：将手势识别结果保存到 MySQL，支持历史记录查看、目标详情查看与简单统计
数据导出：支持将历史识别数据导出为 CSV、JSON 等格式
结果展示：实时显示检测结果、置信度、坐标位置等详细信息
现代化 UI：独特的赛博朋克风格界面，提供优秀的视觉体验
2、 技术栈
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dcd0939ad95243199ed49abc49187cac.png)

主要开发语言：Python
GUI 图形界面框架：PyQt5
深度学习检测模型：YOLOv8
图像处理库：OpenCV
数据库连接：MySQL
深度学习框架：PyTorch
数值计算：NumPy
图像处理：Pillow
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/84104a25448943a08ef25ce0840ae9d5.png)
这是一个非常庞大且功能完善的系统，涵盖了**深度学习（YOLOv8）**、**图形界面（PyQt5）**、**数据库管理（MySQL）**以及**业务逻辑（用户/模型管理）**。

同学，以下作为参考

### 1. 项目目录结构建议

在开始写代码前，建议按照以下结构组织文件：

```text
GestureSystem/
├── main.py                 # 程序入口
├── database.py             # MySQL数据库连接与操作
├── ui_main.py              # PyQt5生成的界面文件 (由QtDesigner设计)
├── detector.py             # YOLOv8推理核心
├── models/                 # 存放 .pt 权重文件
│   └── gesture_v1.pt
├── resources/              # 存放图片、qss样式表
└── utils.py                # 工具类（导出CSV/JSON等）
```

### 2. 核心代码实现

#### 2.1 数据库模块 (`database.py`)
负责处理用户登录、注册以及历史记录的存储。

```python
import mysql.connector
from mysql.connector import Error

class Database:
    def __init__(self, host, user, password, database):
        self.connection = None
        try:
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                passwd=password,
                database=database
            )
            if self.connection.is_connected():
                print("MySQL数据库连接成功")
        except Error as e:
            print(f"数据库连接错误: {e}")

    def execute_query(self, query, values=None):
        cursor = self.connection.cursor()
        try:
            if values:
                cursor.execute(query, values)
            else:
                cursor.execute(query)
            self.connection.commit()
            return cursor
        except Error as e:
            print(f"执行错误: {e}")
            return None

    def fetch_data(self, query, values=None):
        cursor = self.execute_query(query, values)
        if cursor:
            return cursor.fetchall()
        return []

    # 示例：保存识别记录
    def save_record(self, user_id, gesture, confidence, coords):
        query = """
        INSERT INTO history (user_id, gesture_name, confidence, coordinates) 
        VALUES (%s, %s, %s, %s)
        """
        self.execute_query(query, (user_id, gesture, confidence, coords))
```

#### 2.2 YOLOv8 推理模块 (`detector.py`)
负责加载模型、参数配置和图像推理。

```python
from ultralytics import YOLO
import cv2
import numpy as np

class GestureDetector:
    def __init__(self, model_path='models/gesture_v1.pt'):
        # 加载YOLOv8模型
        self.model = YOLO(model_path)
        self.conf_threshold = 0.5
        self.iou_threshold = 0.5

    def update_params(self, conf, iou):
        self.conf_threshold = conf
        self.iou_threshold = iou

    def detect(self, frame):
        # 推理
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
        result = results[0]
        
        # 解析结果
        detections = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = result.names[cls]
            
            detections.append({
                'box': xyxy,
                'confidence': conf,
                'label': label,
                'class_id': cls
            })
            
            # 在帧上绘制边界框 (可选，也可在UI线程绘制)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return frame, detections
```

#### 2.3 主界面与逻辑 (`main.py`)
这是系统的核心，整合了PyQt5界面、视频流处理、多线程和数据库交互。

```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from detector import GestureDetector
from database import Database
# 假设你使用 pyuic5 将 .ui 文件转为了 ui_main.py
from ui_main import Ui_MainWindow 

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, list) # 发送帧和检测结果

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.capture = cv2.VideoCapture(0)
        self.detector = GestureDetector()

    def run(self):
        while self._run_flag:
            ret, cv_img = self.capture.read()
            if ret:
                # 执行检测
                processed_img, detections = self.detector.detect(cv_img)
                self.change_pixmap_signal.emit(processed_img, detections)
        self.capture.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 初始化组件
        self.db = Database('localhost', 'root', 'password', 'gesture_db')
        self.thread = VideoThread()
        
        # 绑定信号槽
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn.clicked.connect(self.stop_video)
        
        # 参数配置联动
        self.conf_slider.valueChanged.connect(self.update_conf)

    def start_video(self):
        self.thread.start()

    def stop_video(self):
        self.thread.stop()

    def update_image(self, cv_img, detections):
        """将OpenCV图像转换为Qt图像并显示"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))
        
        # 更新右侧信息栏 (示例：只显示第一个检测结果)
        if detections:
            det = detections[0]
            self.label_result.setText(f"识别结果: {det['label']}")
            self.progress_conf.setValue(int(det['confidence'] * 100))
            # 保存到数据库逻辑可以在这里触发，或者通过单独的保存按钮触发

    def update_conf(self, val):
        # 动态调整置信度阈值 (假设滑块范围0-100)
        self.thread.detector.update_params(conf=val/100.0, iou=0.5)
        self.conf_value_label.setText(f"{val/100:.2f}")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

### 3. 关键功能实现思路

为了实现你提到的所有高级功能，你需要补充以下逻辑：

1.  **赛博朋克风格 UI**
    -   使用 Qt Designer 设计界面。
    -   编写 `.qss` 样式表文件，设置背景色为深蓝/黑 (`#1a1a2e`)，按钮使用霓虹色 (`#0f3460`, `#e94560`)，字体使用科技感字体。
    -   在主窗口中加载样式：`app.setStyleSheet(open("style.qss", "r").read())`

2.  **多输入模式**
    -   在 `VideoThread` 中增加逻辑，根据用户选择（摄像头/视频文件/图片），初始化不同的 `cv2.VideoCapture` 源（0代表摄像头，或者文件路径）。

3.  **模型版本管理**
    -   在界面添加一个下拉框 (`QComboBox`) 列出 `models/` 文件夹下的所有 `.pt` 文件。
    -   当用户切换时，调用 `self.thread.detector.model = YOLO(new_path)` 重新加载模型。

4.  **历史记录与导出**
    -   利用 `Database` 类查询 `history` 表，将数据填充到 `QTableWidget` 中。
    -   使用 Python 的 `csv` 或 `json` 库读取查询结果并写入文件，实现导出功能。

### 4. 运行前准备

1.  **数据库**：你需要先在 MySQL 中创建数据库和表（用户表、记录表）。
2.  **模型**：将训练好的 `best.pt` 放入 `models` 文件夹。
3.  **UI文件**：使用 Qt Designer 画出界面并转换为 Python 代码。

这套代码框架为你提供了系统运行的骨架，你可以在此基础上填充具体的业务逻辑。
