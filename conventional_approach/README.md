# 传统视觉方案

颜色阈值找灯条 → 灯条两两配对 → 数字网络确认 → PnP 解位姿。

## 模块

| 函数 | 文件 | 作用 |
|---|---|---|
| `preprocess` | `include/func.cc` | 通道差分（红 `R-B` / 蓝 `B-R`）→ 阈值 → 开运算去噪 + 闭运算补断口 |
| `findLights` | `include/func.cc` | `findContours` + `minAreaRect`，按长宽比（1.8~15）和竖直度（≤45°）筛选 |
| `matchBar` | `include/func.cc` | 两两配对：等长（≥0.75）、平行（≤10°）、中心连线与长轴垂直（偏差 ≤15°）、中心距/灯条长 ∈ [1.4, 3.6] |
| `correctCorners` | `include/func.cc` | 取每根灯条**对称轴的两个端点**，按 `左下 → 左上 → 右上 → 右下` 排列 |
| `extractNumber` | `include/func.cc` | 透视变换到 32×28（大板 54×28）→ 中间 20×28 → 灰度 → Otsu → `/255` → 行序展平成 1×560 |
| `classify` | `include/func.cc` | 跑 `mlp.onnx`。**输出是 logits，最后一层没有 Softmax**，代码里自己做 softmax；类别名读同目录 `label.txt` |
| `eraseIgnoreClasses` | `include/func.cc` | 剔除 negative、低置信度（<0.5）、位姿不可信，以及共用灯条/框重叠（IoU>0.5）的重复候选 |
| `positionSolve` | `include/func.cc` | 薄封装，转调共享模块 `src/PnP_solver.cc` 的 `solveArmorPose` |
| `drawResult` | `include/func.cc` | 绿框=灯条、青框=装甲板、红点=中心、文字=距离+号码 |

`include/input_image.cc` 和 `include/PnP_solver.h` + `src/PnP_solver.cc` 是**两个方案共用**的模块。

## 构建

```bash
cd conventional_approach
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

依赖 OpenCV（core/imgproc/imgcodecs/calib3d/dnn/videoio/highgui）。模型放在 `models/`，CMake 会把它的绝对路径编进程序。

## 运行

```bash
# 图片：<路径> <阈值> <blue|red>
./build/armor_conventional_detector "../Project_materials/含真值图像/图1/Image_1791695492312_183.jpg" 25 blue

# 视频：<视频> <输出视频> <输出CSV>（都可省略，用默认路径）
./build/armor_video
```

### 参数怎么选

| 素材 | 颜色 | 阈值 |
|---|---|---|
| 图1（3.8m，蓝灯条，偏暗） | `blue` | 25 |
| 图3（1.5m，蓝灯条） | `blue` | 50（45 时配不出候选） |
| 图2（1.3m，蓝灯条，素材待确认） | `blue` | 30 |
| 视频（1440×1080，红灯条） | `red` | 60 |

视频程序里 `use_digit_filter = true`；图片程序里是 `false`——三张仿真图的数字网络大多判 negative（图1 的"3"太小太糊；图2/图3 在不同阈值下能识别到"3"），关掉它才能出误差表。

## 输出

| 文件 | 内容 |
|---|---|
| `output/debug/result_N.png` | 标注图（原图 + 灯条框 + 装甲板框 + 距离/号码） |
| `output/debug/mask_N.png` | 预处理二值图（调试用） |
| `output/debug/detections_traditional.csv` | 每块保留装甲板一行，供误差评测脚本使用（追加写） |
| `output/traditional_video.avi` | 531 帧标注视频（MJPG） |
| `output/traditional_video.csv` | 视频逐块结果 |

## 注意

- **两套内参不能混**：图片 `cameraImages()`（960×720，零畸变）；视频 `cameraVideo()`（1440×1080 + 5 个畸变系数）。混用会让 z 系统性偏。
- 输出目录 `output/` 相对**当前工作目录**。推荐在 `conventional_approach/` 下运行，标注图会落在 `conventional_approach/output/debug/`。
- `matchBar` 会**原地排序**传入的灯条 vector（按中心 x），这样 `leftBar/rightBar` 的左右归属才稳定；`Armor` 里存的是指向该 vector 元素的指针，调用方不要再对 vector 增删。
