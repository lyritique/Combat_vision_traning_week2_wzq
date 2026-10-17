# 视觉第二周作业：装甲板识别与位置解算

同一份素材用两种方案各做一遍：

| 方案 | 目录 | 识别方式 | 数字结果 |
|---|---|---|---|
| 传统视觉 | `conventional_approach/` | 颜色阈值 → 灯条轮廓 → 灯条配对 → `mlp.onnx` 确认 | 数字网络输出 1~5 / outpost / guard / base / negative |
| 神经网络 | `YOLO_approach/` | YOLOv8-pose 装甲板检测网络，直接出 4 个关键点 | 类别 + 颜色 + 置信度 |

两个方案共用同一套 **PnP 位置解算**（`conventional_approach/src/PnP_solver.cc`）和 **图片读入**（`conventional_approach/include/input_image.cc`），神经方案通过相对路径引用，不复制代码。

## 环境

- g++ 11 或更新（C++17）
- CMake ≥ 3.16
- OpenCV ≥ 4.5.4，需要 `core imgproc imgcodecs calib3d dnn videoio highgui`
- Python 3（只用于模型下载/转换、真值解析、误差评测，不参与识别与解算）

```bash
sudo apt install build-essential cmake libopencv-dev
```

## 目录

```
.
├── conventional_approach/        传统视觉方案（图片 + 视频两个可执行程序）
│   ├── include/                  func.h / func.cc / input_image.cc / PnP_solver.h
│   ├── src/                      main.cc / main_video.cc / PnP_solver.cc
│   └── models/                   mlp.onnx + label.txt（数字识别，不随仓库分发）
├── YOLO_approach/                神经网络方案（图片 + 视频两个可执行程序）
│   ├── include/                  func.h / preprocess.cc / model_use.cc / draw.cc
│   ├── src/                      yoloArmorDetector.cc / main_video.cc
│   └── models/                   Infantry.onnx / Infantry_fp32.onnx
├── data/truth.csv                从 图N.md 解析出的真值（脚本生成）
├── scripts/                      模型转换、真值解析、误差评测
└── Project_materials/            作业素材（不随仓库提交）
```

## 构建

```bash
# 传统方案
cd conventional_approach
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build

# 神经网络方案
cd ../YOLO_approach
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

两个工程都会把模型目录的**绝对路径**编进程序，所以在 `build/` 里运行也能找到模型。

## 模型准备

大模型不随仓库分发（`models/` 已在 `.gitignore` 里）。如果 clone 下来 `models/` 是空的，
按下面两条命令自行下载；仓库里已有模型时可以跳过。

**1. 数字识别网络**（约 314KB）：

```bash
mkdir -p conventional_approach/models && cd conventional_approach/models
BASE="https://gh-proxy.com/https://api.github.com/repos/chenjunnn/rm_auto_aim/contents/armor_detector/model"
curl -sSfL "$BASE/mlp.onnx"  | python3 -c "import json,sys,base64;open('mlp.onnx','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
curl -sSfL "$BASE/label.txt" | python3 -c "import json,sys,base64;open('label.txt','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
ls -l    # mlp.onnx 应为 314058 字节
```

**2. 装甲板检测网络**（约 6.4MB），并转成 OpenCV 能读的 FP32：

```bash
mkdir -p YOLO_approach/models && cd YOLO_approach/models
BASE="https://gh-proxy.com/https://api.github.com/repos/SZURPVision/26_NNDeployment_Lib_and_Detection_Models"
sha=$(curl -sSf "$BASE/git/trees/HEAD?recursive=1" \
      | python3 -c "import json,sys;print([t['sha'] for t in json.load(sys.stdin)['tree'] if 'Infantry' in t['path'] and t['path'].endswith('.onnx')][0])")
curl -sSf "$BASE/git/blobs/$sha" \
      | python3 -c "import json,sys,base64;open('Infantry.onnx','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
ls -l    # Infantry.onnx 应为 6429469 字节

cd ../..
python3 scripts/convert_onnx_fp16_to_fp32.py \
    YOLO_approach/models/Infantry.onnx YOLO_approach/models/Infantry_fp32.onnx
```

**为什么要转换**：OpenCV 4.5.4 的 dnn 不支持 FP16 常量（`Unsupported data type: FLOAT16`）。
转换脚本会把权重转成 FP32、删掉两个空 Cast、并把检测头里的广播常量展开
（后两步分别修掉 `Conv inputs.size() != 0` 和 DFL 里 Mul 的形状断言）。

## 运行

```bash
# 传统方案：图片（阈值 / 灯条颜色）
cd conventional_approach
M=../Project_materials/含真值图像
./build/armor_conventional_detector "$M/图1/Image_1791695492312_183.jpg" 25 blue
./build/armor_conventional_detector "$M/图3/Image_1791695502223_335.jpg" 50 blue

# 传统方案：视频（531 帧 -> 标注视频 + CSV）
./build/armor_video

# 神经网络方案：图片
cd ../YOLO_approach
./build/armor_yolo_detector "$M/图1/Image_1791695492312_183.jpg"

# 神经网络方案：视频
./build/armor_yolo_video
```

输出都在各自工程的 `output/` 下（已 gitignore）：

| 文件 | 内容 |
|---|---|
| `conventional_approach/output/traditional_video.avi` | 传统方案标注视频 |
| `conventional_approach/output/debug/detections_traditional.csv` | 传统方案逐块检测结果 |
| `YOLO_approach/output/yolo_video.avi` | 神经网络标注视频 |
| `YOLO_approach/output/debug/detections_yolo.csv` | 神经网络逐块检测结果 |

## 误差评测

```bash
cd ~/project/Task2

# 1) 从 图N.md 生成真值表
python3 scripts/parse_truth_md.py

# 2) 跑两种方案的图片模式（单文件模式，结果写入各自的 detections CSV）

# 3) 生成误差表（Markdown + CSV）
python3 scripts/evaluate_images.py \
    conventional_approach/output/debug/detections_traditional.csv \
    YOLO_approach/output/debug/detections_yolo.csv
```

输出 `output/eval_images.md` 和 `output/eval_images.csv`，误差定义 **Δ = 计算值 − 真值**。

## 约定

- 相机光学系：**x 右 / y 下 / z 前**；`tvec` 单位 **m**；参考点 = **装甲板几何中心**
- 两套内参不能混：静态图 960×720（`fx=fy=869.1169, cx=480, cy=360`，零畸变）；视频 1440×1080（`fx=2362.2796, fy=2336.1916, cx=679.1751, cy=529.0686` + 5 个畸变系数）
- PnP 物体点 `(0, ±W/2, ±H/2) mm`（小装甲板 W=135，大装甲板 W=225，灯条长 H=55）
- 传统方案角点顺序：**左灯条下 → 左灯条上 → 右灯条上 → 右灯条下**
- 神经网络关键点顺序：**左上 → 左下 → 右下 → 右上**，喂 PnP 前必须经过 `cornersFromNNKeypoints()` 重排

## 已知问题

- **图2 素材与真值配对待确认**：图2 目录下三张 jpg 的检测中心与 `图2.md` 的投影差 40~80px，其中 `Image_1791695496483_858.jpg` 实测对应 `图3.md`。`data/truth.csv` 暂把 `Image_1791732303841_547.jpg` 配到 `图2.md`，但该图检测中心统一偏左 44~60px（两种方案一致），像素对齐存疑。
- 传统方案对阈值敏感：图3 在 th=45 时无候选、th=50 时误差 0.375m；图1 在 th=25 时误差 0.868m（光晕把灯条尺寸撑大，距离被算近）。
- 神经网络方案在视频里会把 4~5m 外的背景图案误检成装甲板，用 2.5m 距离门限过滤（见 `YOLO_approach/src/main_video.cc`）。
