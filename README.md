# 视觉第二周作业：装甲板识别与位置解算

同一份素材，用**两种方案**（传统视觉 / 神经网络）各处理**图片**和**视频**，
解算装甲板在相机光学系下的位置，并与 `图N.md` 的真值比较。

| 方案 | 目录 | 识别方式 | 输出 |
|---|---|---|---|
| 传统视觉 | `conventional_approach/` | 颜色差分 → 灯条轮廓 → 灯条配对 → `mlp.onnx` 确认号码 → PnP | 号码 + 位姿 |
| 神经网络 | `YOLO_approach/` | YOLOv8-pose 直接回归 4 个关键点 → PnP | 类别 + 颜色 + 置信度 + 位姿 |

---

## 0. 全流程一览

```
① clone ─► ② 放素材 ─► ③ 构建 ─► ④ 跑 4 类程序 ─► ⑤ 收 CSV/视频/图
                                            └─► ⑥ 真值表 ─► ⑦ 误差表 ─► ⑧ 分析
```

* **②③ 是一次性准备**（换机器才重做）；
* **④～⑧ 是每次实验的循环**。

| 步骤 | 产出 | 大概耗时 |
|---|---|---|
| ②③ 准备 | 素材就位、两个 `build/` | 3～6 分钟（编译占大头） |
| ④⑤ 四种输入 | 2 个标注视频 + 2 个 detections CSV + 若干标注图 | 1～2 分钟（视频解码占大头） |
| ⑥⑦ 评测 | `data/truth.csv`、`output/eval_images.md` | 秒级 |
| ⑧ 分析 | 一张对比表 + 三条结论 | — |

---

## 目录结构

```
.
├── config/
│   ├── default.yaml               所有可调参数（内参、阈值、路径、门限）
│   ├── config.yaml                另一份参数快照（调参时用）
│   └── image_truth_pairs.csv      图片 ↔ 真值 的唯一对应表（误差表与自检共用）
├── conventional_approach/         传统视觉方案
│   ├── include/                   config.hpp / func.h / func.cc / input_image.cc / PnP_solver.h
│   ├── src/                       main.cc（图片）/ main_video.cc（视频）/ PnP_solver.cc / config.cc
│   └── models/                    mlp.onnx + label.txt
├── YOLO_approach/                 神经网络方案
│   ├── include/                   func.h / preprocess.cc / model_use.cc / draw.cc
│   ├── src/                       yoloArmorDetector.cc（图片）/ main_video.cc（视频）
│   └── models/                    Infantry.onnx（FP16 原始）/ Infantry_fp32.onnx（可直接加载）
├── data/truth.csv                 真值（脚本生成，勿手改）
├── scripts/                       真值解析 / 误差评测 / 配对自检 / 模型转换 / 视频统计
├── Project_materials/             素材（不随仓库提交，需自己放）
└── output/                        评测结果（默认被 .gitignore 忽略）
```

> 报告 `视觉第二周作业_姓名.md` 与两个标注视频**单独提交**，不放进仓库。

## 1. 环境

| 依赖 | 版本 | 用途 |
|---|---|---|
| g++ | **≥ 11**（默认 C++17，代码用了 `std::filesystem`） | 编译 |
| CMake | ≥ 3.16 | 构建 |
| OpenCV | ≥ 4.5.4（需要 `core imgproc imgcodecs calib3d dnn videoio highgui`） | 全部视觉模块 |
| Python 3 | 3.8+ | 真值解析、误差评测、视频统计 |
| Python 包 | `numpy`、`pyyaml`（`check_pairs.py` 画核对图另需 `pillow`） | 脚本 |

```bash
sudo apt install build-essential cmake libopencv-dev python3 python3-numpy python3-yaml
```

> `scripts/evaluate_images.py` 顶部 `import yaml`，**没装 PyYAML 会直接报错**。

自检一下：

```bash
pkg-config --modversion opencv4
g++ --version | head -1
python3 -c "import numpy, yaml; print('py ok')"
```

---

## 2. 拿代码

```bash
git clone https://github.com/lyritique/Combat_vision_traning_week2_wzq.git
cd Combat_vision_traning_week2_wzq
```

主线 `main` 就是最终版本，包含配置层、`data/`、`scripts/` 和两个模型。

---

## 3. 放素材（仓库里没有，必须自己放）

`.gitignore` 里忽略了 `Project_materials/`（体积原因），所以要按下面的结构放到**仓库根目录**：

```
Project_materials/
├── 含真值图像/
│   ├── 图片相机内参.md
│   ├── 图1/  Image_1791695492312_183.jpg   +  图1.md
│   ├── 图2/  Image_1791732303841_547.jpg   +  图2.md  (+ 内录图像_标注_第2次.jpg)
│   └── 图3/  Image_1791695496483_858.jpg   +  图3.md  (+ 内录图像_标注_第3次.jpg)
└── 相机内参与视频/
    ├── Video_1791696012009_214.avi          (531 帧, 1440×1080, 30fps)
    └── 相机内参.md
```

检查：

```bash
ls Project_materials/含真值图像/图1 Project_materials/含真值图像/图2 Project_materials/含真值图像/图3
ls -lh Project_materials/相机内参与视频/
```

> **文件名以素材包为准。** 不同批次的包里 `图3` 的原图可能叫 `Image_1791695502223_335.jpg`
> 或 `Image_1791695496483_858.jpg`。改**一处**即可：`config/image_truth_pairs.csv` 的 `image` 列。
> 报告与 `data/truth.csv` 用的是 `Image_1791695496483_858.jpg`。

---

## 4. 模型（已随仓库提交）

| 文件 | 状态 |
|---|---|
| `conventional_approach/models/mlp.onnx`（314 KB） | 直接可用 |
| `YOLO_approach/models/Infantry_fp32.onnx`（13 MB） | **直接可用**，`config/default.yaml` 指向它 |
| `YOLO_approach/models/Infantry.onnx`（6.4 MB） | FP16 原始权重，OpenCV 4.5.4 读不了，只作转换输入 |

也就是说：**clone 后不需要任何模型准备步骤**。只有在想从 FP16 重新生成 FP32 时才跑：

```bash
python3 scripts/convert_onnx_fp16_to_fp32.py \
    YOLO_approach/models/Infantry.onnx YOLO_approach/models/Infantry_fp32.onnx
```

转换脚本做三件事：FP16→FP32、删掉两个空 Cast、展开检测头的广播常量。
（后两步分别修掉 `Conv inputs.size() != 0` 和 DFL 里的 Mul 形状断言失败。）

> 如果 `models/` 整目录是空的（模型没跟着仓库走），按文末「附录 A」下载。

---

## 5. 构建

两个工程各自独立构建，共享同一份配置和 PnP 模块：

```bash
cd conventional_approach && cmake -S . -B build && cmake --build build -j && cd ..
cd YOLO_approach         && cmake -S . -B build && cmake --build build -j && cd ..
```

得到 4 个可执行程序：

```
conventional_approach/build/armor_conventional_detector
conventional_approach/build/armor_video
YOLO_approach/build/armor_yolo_detector
YOLO_approach/build/armor_yolo_video
```

CMake 会把**仓库根目录**编进程序（`PROJECT_ROOT`），配置文件、模型、素材的路径都相对仓库根解析，
所以**在哪个目录运行都能找到配置**。

---

## 6. 跑四种输入

四个程序 = **2 方案 × 2 输入形态**：

| # | 方案 | 输入 | 程序 | 主要产出 |
|---|---|---|---|---|
| ① | 传统 | 静态图（图1 / 图3） | `armor_conventional_detector` | `detections_traditional.csv`、`result_*.png`、`mask_*.png` |
| ② | 传统 | 视频（531 帧） | `armor_video` | `traditional_video.avi`、`traditional_video.csv` |
| ③ | 神经 | 静态图（图1 / 图3） | `armor_yolo_detector` | `detections_yolo.csv`、`yolo_*.png` |
| ④ | 神经 | 视频（531 帧） | `armor_yolo_video` | `yolo_video.avi`、`yolo_video.csv` |

**统一在仓库根目录运行**：

```bash
mkdir -p output/debug          # YOLO 图片程序的标注图用相对路径写的，先备好
M=Project_materials/含真值图像

# ---------- ① 传统 + 图片（图1 阈值 25，图3 阈值 50，都是蓝色灯条） ----------
./conventional_approach/build/armor_conventional_detector "$M/图1/Image_1791695492312_183.jpg" 25 blue
./conventional_approach/build/armor_conventional_detector "$M/图3/Image_1791695496483_858.jpg" 50 blue

# ---------- ② 传统 + 视频 ----------
./conventional_approach/build/armor_video

# ---------- ③ 神经 + 图片 ----------
./YOLO_approach/build/armor_yolo_detector "$M/图1/Image_1791695492312_183.jpg"
./YOLO_approach/build/armor_yolo_detector "$M/图3/Image_1791695496483_858.jpg"

# ---------- ④ 神经 + 视频 ----------
./YOLO_approach/build/armor_yolo_video
```

> **跑之前先清 CSV**：图片程序的 detections CSV 是**追加写**的，重复跑会累积旧行。
> ```bash
> rm -f conventional_approach/output/debug/detections_traditional.csv \
>       YOLO_approach/output/debug/detections_yolo.csv
> ```

**参数覆盖规则**（三处优先级从高到低）：

1. 命令行位置参数：图片程序 `<输入> <阈值> <blue|red>`；视频程序 `<视频> <输出视频> <输出CSV>`；
2. `--config path/to.yaml` 指定的配置文件；
3. `config/default.yaml` 里的默认值。

**跑对了的标志**（传统 + 图1，控制台应出现类似）：

```
读到 1 张图（.../图1/Image_1791695492312_183.jpg）
第 0 张（960x720） 灯条=2 候选=1
  块#0 长=18/15 数字=negative(0.999427) tvec=(0.0823986, 0.144074, 2.89603)m 重投影=0.610903px
  剔除后剩 1/1 块
```

`tvec` 约 `(0.08, 0.14, 2.90) m`、重投影 < 1 px 说明这条链路是通的（图1 是 3.8 m 远的小目标，误差大是正常的）。

---

## 7. 数据在哪、长什么样

### 7.1 图片检测结果 `detections_*.csv`

| 列 | 含义 | 坐标系 / 单位 |
|---|---|---|
| `image` | 图片文件名 | — |
| `method` | `traditional` / `yolo` | — |
| `det` | 该图内第几个候选（剔除后的序号） | — |
| `label` | 号码（传统）或类别（神经） | — |
| `conf` | 置信度 | 0~1 |
| `x_m, y_m, z_m` | 装甲板**几何中心**在相机光学系下的位置 | **相机光学系，m**（x 右 / y 下 / z 前） |
| `reproj_px` | 四角点重投影的平均绝对误差 | 像素 |
| `px, py` | 检测框中心像素（四角点均值） | 图像像素系 |

### 7.2 视频结果 CSV

```bash
head -3 conventional_approach/output/traditional_video.csv   # frame,bars,candidates,kept,number,conf,x_m,y_m,z_m,reproj_px
head -3 YOLO_approach/output/yolo_video.csv                  # frame,dets,class,color,conf,pose_ok,x_m,y_m,z_m,reproj_px
```

**每块装甲板一行**，没有检测到的帧不写行（用 `frame` 的缺口就能看出漏检）。

### 7.3 标注视频与标注图

| 文件 | 内容 |
|---|---|
| `conventional_approach/output/traditional_video.avi` | 传统方案标注视频（绿框=灯条，青框=装甲板，红点=板中心，文字=号码+距离） |
| `YOLO_approach/output/yolo_video.avi` | 神经方案标注视频（青框=关键点四边形，红点=4 关键点，文字=颜色/类别/置信度+距离） |
| `conventional_approach/output/debug/result_*.png` | 每张图的标注结果 |
| `conventional_approach/output/debug/mask_*.png` | 传统方案的二值图（**调阈值时看这个**） |
| `output/debug/yolo_*.png` | 神经方案的标注图（在仓库根目录的 `output/debug/`） |

---

## 8. 图片误差表

### 8.1 生成真值表

```bash
python3 scripts/parse_truth_md.py --materials Project_materials/含真值图像
# 图1 -> Image_1791695492312_183.jpg：4 块
# 图3 -> Image_1791695496483_858.jpg：4 块
# 共写入 8 行 -> data/truth.csv
```

对应关系只写在 `config/image_truth_pairs.csv` 一处（`enabled=0` 的行只参与自检、不进统计）。
真值是**相机光学系、单位 m、参考点=装甲板几何中心**，和 `x_m,y_m,z_m` 同系同单位，可直接相减。

### 8.2 生成误差表

```bash
python3 scripts/evaluate_images.py \
    conventional_approach/output/debug/detections_traditional.csv \
    YOLO_approach/output/debug/detections_yolo.csv

cat output/eval_images.md
```

脚本做三件事：

1. 用**静态图内参**把真值中心投到像素：`u = fx·X/Z + cx`；
2. 和检测的 `(px,py)` 做**一对一最近匹配**（门限 `--max-px`，默认 20 px）；
3. 对匹配上的行算 `Δ = 计算值 − 真值`、三维误差 `‖Δ‖`；像素差 > 8 px 的行会标 `[可疑]`。

### 8.3 参考值（本仓库实测）

| 图 | 方案 | 板号 | 计算 (x,y,z) m | 真值 (x,y,z) m | 三维误差 | 重投影 | 像素差 |
|---|---|---|---|---|---|---|---|
| 图1 | 传统 | armor_4 | (0.082, 0.144, 2.896) | (0.119, 0.184, 3.763) | **0.868 m** | 0.61 px | 3.2 px |
| 图1 | 神经 | armor_1 | (−0.163, 0.156, 3.525) | (−0.174, 0.170, 3.795) | **0.270 m** | 0.13 px | 0.3 px |
| 图1 | 神经 | armor_4 | (0.104, 0.181, 3.617) | (0.119, 0.184, 3.763) | **0.146 m** | 0.02 px | 2.9 px |
| 图3 | 传统 | armor_4 | (−0.112, 0.082, 1.361) | (−0.092, 0.092, 1.467) | **0.108 m** | 1.70 px | 16.0 px |
| 图3 | 神经 | armor_3 | (0.166, 0.070, 1.462) | (0.189, 0.070, 1.553) | **0.094 m** | 0.04 px | 8.5 px |
| 图3 | 神经 | armor_4 | (−0.119, 0.088, 1.430) | (−0.092, 0.092, 1.467) | **0.046 m** | 0.09 px | 17.0 px |

平均三维误差：传统 ≈ **0.49 m**（2 块），神经 ≈ **0.14 m**（4 块）。

读法：**Δz 是主导项**（Δx、Δy 通常只有几厘米），所以先看 z，再看 x/y。

### 8.4 为什么 图2 不进表

`图2/Image_1791732303841_547.jpg` 配 `图2.md` 时，两块正面板的真值投影整体偏 **44～60 px**，
而同一张图配 `图3.md` 只差 6 px —— 说明素材与 `/tf` 记录**不是同一帧**。
所以 `config/image_truth_pairs.csv` 里这两行都是 `enabled=0`，只用于复现这个结论，不进统计。

```bash
# 只看配对残差，不进误差表
python3 scripts/check_pairs.py --materials Project_materials/含真值图像
```

> 早期版本把 `--max-px` 默认设成 80 px，上面的 44～60 px 错位会被当成"正常匹配"，
> 于是产出 0.11～0.20 m 这种"看起来很好"的假精度。默认收到 20 px 后，这类行会直接变成"未匹配"。

**图3 也要打折扣看**：`Image_1791695496483_858.jpg` 与 `图3.md` 仍有 8～17 px 的整体偏移
（同样是抓图时刻与真值时刻的差），所以 图3 的绝对误差只能当"同一量级"参考，不能当精度的证据。
两张图里真正干净的是 **图1**。

---

## 9. 视频数据的分析

视频没有真值，能分析的是**稳定性**：检出率、距离一致性、误检、速度。

```bash
python3 scripts/video_stats.py \
    conventional_approach/output/traditional_video.csv \
    YOLO_approach/output/yolo_video.csv
```

| 指标 | 传统 | 神经 |
|---|---|---|
| 检测块数 | 553 | 804 |
| 有装甲板的帧 | 523 / 531（98.5%） | 530 / 531（99.8%） |
| 每帧平均块数 | 1.06 | 1.52 |
| 重投影 中位 / P95 | 1.19 / 2.77 px | 0.38 / 1.02 px |
| 距离 z 中位 / 范围 | 1.46 m / 1.36～1.64 m | 1.54 m / 1.45～2.04 m |
| 号码 / 类别 | 全部 `4` | 全部 `infantry4`（红 799 / 白 5） |
| 一帧两块板的帧 | 30 | 274 |
| 速度（本机 CPU） | ≈ 60 fps | ≈ 20 fps |

分析要点：

* **检出率**：`有装甲板的帧 / 531`，比较两种方案的稳定性；
* **距离一致性**：中位数应贴近 1.4～1.6 m，标准差小说明解算稳；
* **误检**：神经方案在 4～5.8 m 有背景图案误检，靠 `yolo.max_dist_m = 2.5` 挡掉 256 个——
  把这项改成 0 再跑一次，看 `检测总数` 的差，就能量化这项后处理的价值；
* **速度**：程序结尾会直接打印 fps。

---

## 10. 两种方案怎么比较

### 10.1 精度（图片）

| 图 | 板 | 传统三维 | 神经三维 | 谁更准 |
|---|---|---|---|---|
| 图1 | armor_4 | 0.868 m | 0.146 m | 神经 |
| 图1 | armor_1 | 漏检 | 0.270 m | 神经 |
| 图3 | armor_4 | 0.108 m | 0.046 m | 神经 |
| 图3 | armor_3 | 漏检 | 0.094 m | 神经 |

### 10.2 鲁棒性（视频）

| 指标 | 传统 | 神经 | 说明 |
|---|---|---|---|
| 检出率 | 98.5% | 99.8% | 阈值法对光晕/阈值敏感 |
| 误检来源 | 灯条配对错误 | 4～5 m 背景图案 | 一个靠数字过滤，一个靠距离门限 |
| 对参数敏感度 | 高（图1 th=30 直接无候选） | 低 | 传统方案要按素材调阈值 |

### 10.3 速度

传统 ≈ 60 fps、神经 ≈ 20 fps（本机 CPU）。原因：传统方案是像素级阈值运算，
神经方案要跑一次 640×480 前向。机器不同数值会浮动（同一份代码在另一台机器上是 48 / 23 fps）。

### 10.4 写进"问题分析"的三点

1. **传统方案的距离误差主要来自光晕。**
   图1 距离 3.8 m 时灯条理论投影 `fx·55/z ≈ 12.5 px`，二值图里实测 15～18 px；
   PnP 用固定模板反推距离，尺寸一胀，距离就算近（Δz = −0.867 m）。误差随距离放大。
2. **神经方案精度更高，但要知道它高在哪。**
   网络直接回归灯条端点，不受光晕/闭运算影响，图1 的检测中心与真值投影只差 0.3～2.9 px。
   但 `reproj_px` 很小**不等于**位置准（它只说明"这 4 个点自己自洽"）。
3. **数字网络在仿真图上不稳。**
   图1 判 `negative`（灯条只有十几像素），图3 认出 `3`（0.97）；
   视频上 553 块全部判 `4`，与场景一致。所以图片模式关掉数字过滤，视频模式保留。

---

## 11. 模型与 `.gitignore`

`.gitignore` 只管 `build/`、`Project_materials/`、`output/`、`.cache`，**不再忽略 `models/`**：
仓库里同时有 `mlp.onnx`、`Infantry.onnx`（FP16 源）和 `Infantry_fp32.onnx`（可直接加载），
所以 fresh clone 不用做任何模型转换。

---

## 12. 一键复现（可选）

在仓库根目录、素材已就位的前提下：

```bash
#!/usr/bin/env bash
set -e
M=Project_materials/含真值图像
mkdir -p output/debug

rm -f conventional_approach/output/debug/detections_traditional.csv \
      YOLO_approach/output/debug/detections_yolo.csv

# ①② 传统：图片 + 视频
./conventional_approach/build/armor_conventional_detector "$M/图1/Image_1791695492312_183.jpg" 25 blue
./conventional_approach/build/armor_conventional_detector "$M/图3/Image_1791695496483_858.jpg" 50 blue
./conventional_approach/build/armor_video
# ③④ 神经：图片 + 视频
./YOLO_approach/build/armor_yolo_detector "$M/图1/Image_1791695492312_183.jpg"
./YOLO_approach/build/armor_yolo_detector "$M/图3/Image_1791695496483_858.jpg"
./YOLO_approach/build/armor_yolo_video

# ⑥⑦ 真值表 + 误差表
python3 scripts/parse_truth_md.py --materials "$M"
python3 scripts/evaluate_images.py \
    conventional_approach/output/debug/detections_traditional.csv \
    YOLO_approach/output/debug/detections_yolo.csv
python3 scripts/video_stats.py \
    conventional_approach/output/traditional_video.csv \
    YOLO_approach/output/yolo_video.csv
echo "误差表: output/eval_images.md"
```

保存为 `scripts/run_all.sh` 后 `bash scripts/run_all.sh` 即可（本机约 1 分钟，编解码占大头）。

---

## 附录 A：模型下载（仅当 `models/` 是空的）

**数字识别网络**（314 KB）：

```bash
mkdir -p conventional_approach/models && cd conventional_approach/models
BASE="https://gh-proxy.com/https://api.github.com/repos/chenjunnn/rm_auto_aim/contents/armor_detector/model"
curl -sSfL "$BASE/mlp.onnx"  | python3 -c "import json,sys,base64;open('mlp.onnx','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
curl -sSfL "$BASE/label.txt" | python3 -c "import json,sys,base64;open('label.txt','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
cd ../..
```

**装甲板检测网络**（6.4 MB，转成 FP32）：

```bash
mkdir -p YOLO_approach/models && cd YOLO_approach/models
BASE="https://gh-proxy.com/https://api.github.com/repos/SZURPVision/26_NNDeployment_Lib_and_Detection_Models"
sha=$(curl -sSf "$BASE/git/trees/HEAD?recursive=1" \
      | python3 -c "import json,sys;print([t['sha'] for t in json.load(sys.stdin)['tree'] if 'Infantry' in t['path'] and t['path'].endswith('.onnx')][0])")
curl -sSf "$BASE/git/blobs/$sha" \
      | python3 -c "import json,sys,base64;open('Infantry.onnx','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
cd ../..
python3 scripts/convert_onnx_fp16_to_fp32.py \
    YOLO_approach/models/Infantry.onnx YOLO_approach/models/Infantry_fp32.onnx
```

---

## 附录 B：关键约定

* **坐标系**：相机光学系，**x 右 / y 下 / z 前**（OpenCV 约定）；
* **单位**：`tvec` 与真值都是 **m**（物体点内部用 mm，解算后 `/1000`）；
* **参考点**：**装甲板几何中心**（不是灯条中点、不是底盘中心）；
* **两套内参**：静态图 960×720（`fx=fy=869.1169, cx=480, cy=360`，零畸变）；
  视频 1440×1080（`fx=2362.28, fy=2336.19, cx=679.18, cy=529.07` + 5 个畸变系数）；
* **PnP 物体点**：`(0, ±W/2, ±H/2) mm`（小板 W=135，大板 W=225，灯条长 H=55）；
* **角点顺序**：传统 左灯条下→左灯条上→右灯条上→右灯条下；神经 左上→左下→右下→右上（喂 PnP 前会重排）。
