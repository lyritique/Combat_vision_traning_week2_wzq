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
① clone ─► ② 放素材 ─► ③ 转模型 ─► ④ 构建 ─► ⑤ 跑 4 类程序 ─► ⑥ 收 CSV/视频/图
                                                              └─► ⑦ 真值表 ─► ⑧ 误差表 ─► ⑨ 分析
```

* **②③④ 是一次性准备**（换机器才重做）；
* **⑤～⑨ 是每次实验的循环**。

| 步骤 | 产出 | 大概耗时 |
|---|---|---|
| ②③④ 准备 | 素材就位、`Infantry_fp32.onnx`、两个 `build/` | 5～10 分钟（编译占大头） |
| ⑤⑥ 四种输入 | 2 个标注视频 + 2 个 detections CSV + 若干标注图 | 4～6 分钟（视频解码占大头） |
| ⑦⑧ 评测 | `data/truth.csv`、`output/eval_images.md` | 秒级 |
| ⑨ 分析 | 一张对比表 + 三条结论 | — |

---

## 1. 环境

| 依赖 | 版本 | 用途 |
|---|---|---|
| g++ | **≥ 11**（默认 C++17，代码用了 `std::filesystem`） | 编译 |
| CMake | ≥ 3.16 | 构建 |
| OpenCV | ≥ 4.5.4（需要 `core imgproc imgcodecs calib3d dnn videoio highgui`） | 全部视觉模块 |
| Python 3 | 3.8+ | 模型转换、真值解析、误差评测 |
| Python 包 | `numpy`、`pyyaml` | 转换脚本 / 评测脚本读配置 |

```bash
sudo apt install build-essential cmake libopencv-dev python3 python3-numpy python3-yaml
```

> `scripts/evaluate_images.py` 顶部 `import yaml`，**没装 PyYAML 会直接报错**。
> 如果你的 g++ 是 9 或 10，构建时加 `-DCMAKE_CXX_STANDARD=17`。

自检一下：

```bash
opencv_version --version 2>/dev/null || pkg-config --modversion opencv4
g++ --version | head -1
python3 -c "import numpy, yaml; print('py ok')"
```

---

## 2. 拿代码（注意分支）

```bash
git clone -b report/week2-final https://github.com/lyritique/Combat_vision_traning_week2_wzq.git
cd Combat_vision_traning_week2_wzq
```

> **必须用 `report/week2-final` 分支。** `main` 是早期版本，没有配置层、没有报告、没有 `data/` 和 `scripts/`。

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

> **文件名以你手上的素材包为准。** 不同批次的包里，`图3` 的原图可能叫 `Image_1791695502223_335.jpg`。
> 仓库里的 `data/truth.csv` 是按 **图1→`..._2312_183.jpg`、图3→`..._502223_335.jpg`、图2→`..._547.jpg`** 写的；
> 如果你的文件名不一样，两种办法：改 `scripts/parse_truth_md.py` 的 `MD_TO_IMAGE` 重新生成，或直接编辑 `data/truth.csv` 的 `image` 列。

---

## 4. 转模型（**fresh clone 必做**）

仓库里带了两个模型：

| 文件 | 状态 |
|---|---|
| `conventional_approach/models/mlp.onnx`（314 KB） | 直接可用 |
| `YOLO_approach/models/Infantry.onnx`（6.4 MB） | **是 FP16 权重，OpenCV 4.5.4 读不了** |
| `YOLO_approach/models/Infantry_fp32.onnx` | **不在仓库里**，但 `config/default.yaml` 指向它 |

所以 **YOLO 两个程序在转换之前一定跑不起来**（会报 `Unsupported data type: FLOAT16`）。跑一次：

```bash
python3 scripts/convert_onnx_fp16_to_fp32.py \
    YOLO_approach/models/Infantry.onnx YOLO_approach/models/Infantry_fp32.onnx
ls -l YOLO_approach/models/          # 应看到 Infantry_fp32.onnx（约 13 MB）
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

只跑默认（图1 / 25 / blue / 蓝灯条）时，直接 `./conventional_approach/build/armor_conventional_detector` 就行。

**跑对了的标志**（传统 + 图1，控制台应出现类似）：

```
读到 1 张图（.../图1/Image_1791695492312_183.jpg）
第 0 张（960x720） 灯条=… 候选=…
  块#0  数字=…  tvec=(0.082, 0.144, 2.896)m 重投影=0.61px
  剔除后剩 1/1 块
```

`tvec` 约 `(0.08, 0.14, 2.90) m`、重投影 < 1 px 说明这条链路是通的（图1 是 3.8 m 远的小目标，误差大是正常的）。

---

## 7. 数据在哪、长什么样

### 7.1 图片检测结果 `detections_*.csv`

```bash
head -3 conventional_approach/output/debug/detections_traditional.csv
head -3 YOLO_approach/output/debug/detections_yolo.csv
```

| 列 | 含义 | 坐标系 / 单位 |
|---|---|---|
| `image` | 图片文件名 | — |
| `method` | `traditional` / `yolo` | — |
| `det` | 该图内第几个候选（剔除后的序号） | — |
| `label` | 号码（传统：`4` / `negative`…）或类别（神经：`infantry4`…） | — |
| `conf` | 置信度（传统=数字网络 softmax；神经=类别 sigmoid 最大值） | 0~1 |
| `x_m, y_m, z_m` | 装甲板**几何中心**在相机光学系下的位置 | **相机光学系，m**（x 右 / y 下 / z 前） |
| `reproj_px` | 四角点重投影的平均绝对误差 | 像素 |
| `px, py` | 检测框中心像素（四角点均值） | **图像像素系**，px |

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
python3 scripts/parse_truth_md.py
# 图1: 4 块装甲板 -> Image_1791695492312_183.jpg
# 图3: 4 块装甲板 -> Image_1791695502223_335.jpg
# 图2: 4 块装甲板 -> Image_1791732303841_547.jpg
# 共写入 12 行 -> data/truth.csv
```

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
2. 和检测的 `(px,py)` 做**一对一最近匹配**（门限 `eval.max_px`，默认 80 px）；
3. 对匹配上的行算 `Δ = 计算值 − 真值`、三维误差 `‖Δ‖`。

### 8.3 读表

| 列 | 含义 |
|---|---|
| 计算 (x,y,z) | 程序解出来的板中心（相机系，m） |
| 真值 (x,y,z) | `图N.md` 的 1.1 表 |
| Δx / Δy / Δz | 计算 − 真值，**逐轴** |
| 三维误差 | `√(Δx²+Δy²+Δz²)`，作业要求的那一项 |
| 重投影 | 求解器对**自己这 4 个点**的拟合残差（不是精度指标，见 §11.4） |
| 像素差 | 检测中心与真值投影的像素距离，越大越可疑 |

**参考值**（你的机器上应接近，差太多说明素材/阈值/内参对不上）：

| 图 | 方案 | 板号 | 三维误差 |
|---|---|---|---|
| 图1 | 传统 | armor_4 | **0.868 m** |
| 图1 | 神经 | armor_1 / armor_4 | **0.270 / 0.146 m** |
| 图3 | 传统 | armor_4 | **0.375 m** |
| 图3 | 神经 | armor_3 | **0.053 m** |

平均：传统 ≈ 0.62 m，神经 ≈ 0.16 m。

### 8.4 严格门限复跑（**这一步是分析的关键**）

默认 `eval.max_px = 80` 时，图2 那三行也会出现在表里（误差 0.11~0.20 m）。但它们的**像素差是 44~60 px**——
远大于图1/图3 的 0.3~3 px。把门限收紧：

```bash
python3 scripts/evaluate_images.py --max-px 20 \
    conventional_approach/output/debug/detections_traditional.csv \
    YOLO_approach/output/debug/detections_yolo.csv
```

结果会变成：**图2 的行消失**（检测变成了"未匹配"、真值变成了"漏检"）。
这说明 80 px 的宽门限把"素材与真值不是同一帧"错配成了"精度 0.1 m"。

---

## 9. 视频数据的分析

视频没有真值，能分析的是**稳定性**：检出率、距离一致性、误检、速度。
下面的脚本直接算（在仓库根目录运行）：

```bash
python3 - <<'PY'
import csv, statistics, collections

for path in ("conventional_approach/output/traditional_video.csv",
             "YOLO_approach/output/yolo_video.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print(path, "← 空文件，检查程序是否跑成功"); continue
    frames = {int(r["frame"]) for r in rows}
    zs = [float(r["z_m"]) for r in rows]
    rp = [float(r["reproj_px"]) for r in rows]
    print(f"\n== {path} ==")
    print(f"  行数 {len(rows)}  涉及帧 {len(frames)}  平均每帧 {len(rows)/len(frames):.2f} 块")
    print(f"  z: min {min(zs):.2f} / 中位 {statistics.median(zs):.2f} / max {max(zs):.2f} m")
    print(f"  重投影中位数 {statistics.median(rp):.2f} px")
    key = "number" if "number" in rows[0] else "class"
    print(f"  {key} 分布:", dict(collections.Counter(r[key] for r in rows).most_common(5)))
PY
```

**参考值**（531 帧）：

| 指标 | 传统 | 神经 |
|---|---|---|
| 有装甲板的帧 | 523（98.5%） | 530（99.8%） |
| 候选 → 保留 | 613 → 553 | 1060 → 804 |
| 号码/类别 | 全部 `4` | `infantry4`（红 799 / 白 5） |
| 距离范围 | 1.36 ~ 1.64 m | 1.45 ~ 2.04 m |
| 速度 | ≈48 fps | ≈23 fps |

分析要点：

* **检出率**：`有装甲板的帧 / 531`，比较两种方案的稳定性；
* **距离一致性**：中位数应贴近 1.4~1.6 m，标准差小说明解算稳；
* **误检**：神经方案在 4~5.8 m 有背景图案误检，靠 `yolo.max_dist_m = 2.5` 挡掉——
  把配置里这项改成 0 再跑一次，看 `检测总数` 的差，就能量化这项后处理的价值；
* **速度**：程序结尾会直接打印 fps。

---

## 10. 两种方案怎么比较（分析模板）

把 §8.3 的表和 §9 的表填进报告，然后按三条主线写：

### 10.1 精度（图片）

| 图 | 板 | 传统 Δ(x,y,z) | 传统三维 | 神经 Δ(x,y,z) | 神经三维 | 谁更准 |
|---|---|---|---|---|---|---|
| 图1 | armor_4 | −0.037 / −0.040 / −0.867 | 0.868 | −0.016 / −0.003 / −0.145 | 0.146 | 神经 |
| 图3 | armor_4 / armor_3 | … | 0.375 | … | 0.053 | 神经 |

读法：**Δz 是主导项**（Δx、Δy 通常只有几厘米），所以先看 z，再看 x/y。

### 10.2 鲁棒性（视频）

| 指标 | 传统 | 神经 | 说明 |
|---|---|---|---|
| 检出率 | 98.5% | 99.8% | 阈值法对光晕/阈值敏感 |
| 误检来源 | 灯条配对错误 | 4~5 m 背景图案 | 一个靠数字过滤，一个靠距离门限 |
| 对参数敏感度 | 高（图3 th=45 直接无候选） | 低 | 传统方案要按素材调阈值 |

### 10.3 速度

传统 ≈48 fps、神经 ≈23 fps（本机 CPU）。原因：传统方案是像素级阈值运算，神经方案要跑一次 640×480 前向。

### 10.4 讨论（写进"问题分析"的三点）

1. **传统方案的距离误差主要来自光晕。**
   图1 距离 3.8 m 时灯条理论投影 `fx·55/z ≈ 12.5 px`，二值图里实测 15~18 px；
   PnP 用固定模板反推距离，尺寸一胀，距离就算近（Δz = −0.867 m）。误差随距离放大。
2. **神经方案精度更高，但要知道它高在哪。**
   网络直接回归灯条端点，不受光晕/闭运算影响，检测中心与真值投影只差 0.3~2.9 px。
   但 `reproj_px` 很小**不等于**位置准（见 §11.4）。
3. **数字网络在仿真图上不稳。**
   图1 判 `negative`（灯条只有十几像素），图3 在部分阈值下能认出 `3`；
   视频上 553 块全部判 `4`，与场景一致。所以图片模式关掉数字过滤，视频模式保留。

---

## 11. 已知问题 / 预期偏差（**先看这段，能省很多时间**）

### 11.1 图2 的素材与真值对不上（最需要注意）

`data/truth.csv` 现在把 `Image_1791732303841_547.jpg` 配到 `图2.md`，但实测：

| 检查项 | 结果 |
|---|---|
| `图2.md` 的 armor_1 投影 | (659, 404)，落在右板**右侧的赛道上** |
| `图2.md` 的 armor_2 投影 | (487, 426)，落在**两块板之间的空隙** |
| 两块正面板相对检测中心的偏差 | 同向偏 **44~60 px**（图1/图3 只有 0.3~3 px） |
| 同一张图换 `图3.md` 投影 | armor_4 → (425,415)，与左板中心 (427,417) 只差 **2.8 px** |

结论：**`图2.md` 对应的原始帧不在这批素材里**（很可能抓图时刻与 /tf 记录时刻不重合）。
所以：

* 默认（`max_px=80`）下表里会多出三行图2 的数据，**它们是错配出来的，不要当成精度结论**；
* 用 `--max-px 20` 复跑（§8.4），图2 的行会消失——这才是诚实的做法；
* 报告里应写成"图2 素材待确认，未计入平均值"，并附上"投影是否落在装甲板上"的核对图。

**怎么自己验证**：把 `图2.md` 的四块板中心按 `u = fx·X/Z+cx, v = fy·Y/Z+cy` 投到图上画十字，
看它是否盖住装甲板的两根灯条；或用 `图2/内录图像_标注_第2次.jpg` 对照（那张图的框本身就是按 `图2.md` 投影画的，
所以它本身也在板外，不能当独立参照）。

### 11.2 素材文件名不一定和 `data/truth.csv` 一致

`truth.csv` 是按 **图3 → `Image_1791695502223_335.jpg`** 生成的。如果你的素材包里图3 叫
`Image_1791695496483_858.jpg`，那图3 的检测行就匹配不上真值（结果里会"没有检测结果"或整行缺失）。
改 `scripts/parse_truth_md.py` 的 `MD_TO_IMAGE` 或直接改 `data/truth.csv` 的 `image` 列即可。

### 11.3 模型与 `.gitignore` 的矛盾

`.gitignore` 里写了 `models/`，但仓库里确实提交了 `mlp.onnx` 和 `Infantry.onnx`（早期强加）。
现状是：**`mlp.onnx` 有、`Infantry.onnx` 有（FP16）、`Infantry_fp32.onnx` 没有**。
所以 §4 的转换步骤不能省。

### 11.4 `reproj_px` 很小 ≠ 位置准

单目 PnP 里，**深度 z 和物体点尺寸 W 是互相吸收的**：`z ≈ fx·W/d`。
如果 `W` 写大了 10%，求解器会把 z 也放大 10%，四个角点照样严丝合缝，`reproj_px` 不会变大。
所以 `0.02 px` 只说明"形状拟合得好"，z 的精度取决于 `config` 里 `armor.small_width_mm = 135`
到底是不是"两灯条中心距"——这一项值得用实物/像素量一次确认。

### 11.5 其它小坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 反复跑之后误差表变脏 | detections CSV 是**追加写** | 跑前 `rm` 掉（§6） |
| YOLO 图片程序的标注图没生成 | 它把图写到 `output/debug/yolo_*.png`（相对**当前目录**） | 在仓库根先 `mkdir -p output/debug` |
| `config/config.yaml` 不生效 | 程序只读 `config/default.yaml`（或 `--config` 指定的） | 忽略 `config.yaml`，改 `default.yaml` |
| 视频跑完 z 永远 > 2.5 m 被全丢 | `yolo.max_dist_m` 门限 | 改成 0 关闭门限对比 |
| 结果里没有图2 | 已经用 `--max-px 20` 收紧 | 预期行为（§11.1） |

---

## 12. 一键复现（可选）

把 §6～§8 串起来（在仓库根目录，确认素材和模型已就位）：

```bash
#!/usr/bin/env bash
set -e
M=Project_materials/含真值图像
V=Project_materials/相机内参与视频/Video_1791696012009_214.avi
mkdir -p output/debug

rm -f conventional_approach/output/debug/detections_traditional.csv \
      YOLO_approach/output/debug/detections_yolo.csv

# ① 传统 + 图片
./conventional_approach/build/armor_conventional_detector "$M/图1/Image_1791695492312_183.jpg" 25 blue
./conventional_approach/build/armor_conventional_detector "$M/图3/Image_1791695496483_858.jpg" 50 blue
# ③ 神经 + 图片
./YOLO_approach/build/armor_yolo_detector "$M/图1/Image_1791695492312_183.jpg"
./YOLO_approach/build/armor_yolo_detector "$M/图3/Image_1791695496483_858.jpg"
# ② ④ 视频
./conventional_approach/build/armor_video
./YOLO_approach/build/armor_yolo_video

# ⑦ ⑧ 评测
python3 scripts/parse_truth_md.py
python3 scripts/evaluate_images.py \
    conventional_approach/output/debug/detections_traditional.csv \
    YOLO_approach/output/debug/detections_yolo.csv --max-px 20
echo "误差表: output/eval_images.md"
```

保存为 `scripts/run_all.sh` 后 `bash scripts/run_all.sh` 即可（视频那两步大约 1 分钟）。

---

## 附录 A：模型下载（仅当 `models/` 是空的）

**数字识别网络**（314 KB）：

```bash
mkdir -p conventional_approach/models && cd conventional_approach/models
BASE="https://gh-proxy.com/https://api.github.com/repos/chenjunnn/rm_auto_aim/contents/armor_detector/model"
curl -sSfL "$BASE/mlp.onnx"  | python3 -c "import json,sys,base64;open('mlp.onnx','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
curl -sSfL "$BASE/label.txt" | python3 -c "import json,sys,base64;open('label.txt','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
ls -l   # mlp.onnx 应为 314058 字节
```

**装甲板检测网络**（6.4 MB，转成 FP32）：

```bash
cd ../.. && mkdir -p YOLO_approach/models && cd YOLO_approach/models
BASE="https://gh-proxy.com/https://api.github.com/repos/SZURPVision/26_NNDeployment_Lib_and_Detection_Models"
sha=$(curl -sSf "$BASE/git/trees/HEAD?recursive=1" \
      | python3 -c "import json,sys;print([t['sha'] for t in json.load(sys.stdin)['tree'] if 'Infantry' in t['path'] and t['path'].endswith('.onnx')][0])")
curl -sSf "$BASE/git/blobs/$sha" \
      | python3 -c "import json,sys,base64;open('Infantry.onnx','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
cd ../..
python3 scripts/convert_onnx_fp16_to_fp32.py \
    YOLO_approach/models/Infantry.onnx YOLO_approach/models/Infantry_fp32.onnx
```

> GitHub 未认证 API 每小时只有 60 次请求，撞上限时把 `gh-proxy.com` 换掉或稍后再试。

---

## 附录 B：目录结构

```
.
├── config/default.yaml            所有可调参数（内参、阈值、路径、门限）
├── conventional_approach/         传统视觉方案
│   ├── include/                   config.hpp / func.h / func.cc / input_image.cc / PnP_solver.h
│   ├── src/                       main.cc（图片）/ main_video.cc（视频）/ PnP_solver.cc / config.cpp
│   └── models/                    mlp.onnx + label.txt
├── YOLO_approach/                 神经网络方案
│   ├── include/                   func.h / preprocess.cc / model_use.cc / draw.cc
│   ├── src/                       yoloArmorDetector.cc（图片）/ main_video.cc（视频）
│   └── models/                    Infantry.onnx（FP16）/ Infantry_fp32.onnx（转换后）
├── data/truth.csv                 真值（脚本生成，勿手改）
├── scripts/                       模型转换 / 真值解析 / 误差评测
├── Project_materials/             素材（不随仓库提交，需自己放）
└── 视觉第二周作业_姓名.md          报告
```

---

## 附录 C：关键约定（看数字前先确认）

* **坐标系**：相机光学系，**x 右 / y 下 / z 前**（OpenCV 约定）；
* **单位**：`tvec` 与真值都是 **m**（物体点内部用 mm，解算后 `/1000`）；
* **参考点**：**装甲板几何中心**（不是灯条中点、不是底盘中心）；
* **两套内参不能混**：静态图 960×720（`fx=fy=869.1169, cx=480, cy=360`，零畸变）；
  视频 1440×1080（`fx=2362.28, fy=2336.19, cx=679.18, cy=529.07` + 5 个畸变系数）；
* **PnP 物体点**：`(0, ±W/2, ±H/2) mm`（小板 W=135，大板 W=225，灯条长 H=55）；
* **角点顺序**：传统 左灯条下→左灯条上→右灯条上→右灯条下；神经 左上→左下→右下→右上（喂 PnP 前重排）。
