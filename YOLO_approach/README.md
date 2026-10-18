# 神经网络方案

YOLOv8-pose 装甲板检测网络直接输出 4 个关键点 → 重排 → 复用传统方案的 PnP 解位姿。

## 模型

模型是 `SZURPVision/26_NNDeployment_Lib_and_Detection_Models` 里的
`Infantry-v8n-fp16-20260726-D1.8w-B16.onnx`（6,429,469 字节），**权重是真正的 FP16**。

### 下载

```bash
mkdir -p YOLO_approach/models && cd YOLO_approach/models
BASE="https://gh-proxy.com/https://api.github.com/repos/SZURPVision/26_NNDeployment_Lib_and_Detection_Models"
sha=$(curl -sSf "$BASE/git/trees/HEAD?recursive=1" \
      | python3 -c "import json,sys;print([t['sha'] for t in json.load(sys.stdin)['tree'] if 'Infantry' in t['path'] and t['path'].endswith('.onnx')][0])")
curl -sSf "$BASE/git/blobs/$sha" \
      | python3 -c "import json,sys,base64;open('Infantry.onnx','wb').write(base64.b64decode(json.load(sys.stdin)['content']))"
ls -l Infantry.onnx    # 应为 6429469 字节
```

`raw.githubusercontent.com` 在本机不通（timeout / TLS reset），`gh-proxy.com` 可用。

### 转成 FP32（必须）

OpenCV 4.5.4 的 dnn **读不了 FP16 常量**，直接加载会报：

```
Unsupported data type: FLOAT16 in function 'getMatFromTensor'
```

```bash
cd ../..    # 回到仓库根目录
python3 scripts/convert_onnx_fp16_to_fp32.py \
    YOLO_approach/models/Infantry.onnx YOLO_approach/models/Infantry_fp32.onnx
```

脚本做三件事，缺一不可（都是实测踩出来的）：

1. initializer / value_info / Cast 目标里的 FLOAT16 → FLOAT；
2. **删掉两个空 Cast**（PyTorch FP16 导出在输入/输出各插了一个，转 FP32 后变成 FLOAT→FLOAT，
   OpenCV 不给它们建层，后面的 Conv 会报 `inputs.size() != 0`）；
3. **展开检测头的广播常量**（DFL 里 `1×4×2×6300` 和 `(1,6300)` 的 Mul，OpenCV 旧版 Mul 不支持
   高维广播，会断言失败）。

程序运行时优先用 `models/Infantry_fp32.onnx`，没有才退回 `Infantry.onnx`。

## 模块

| 文件 | 作用 |
|---|---|
| `include/preprocess.cc` | **letterbox**：等比缩放到 640×480 以内、灰 124 填充；BGR→RGB、`/255`、HWC→CHW，输出 1×3×480×640 |
| `include/model_use.cc` | 加载模型、前向、解码 `1×21×6300`、NMS |
| `include/draw.cc` | 青框=装甲板、红点=关键点、文字=颜色/类别/置信度/距离 |
| `src/yoloArmorDetector.cc` | 图片模式 |
| `src/main_video.cc` | 视频批处理（531 帧 → 标注视频 + CSV） |

`PnP_solver` 和 `input_image` 通过 `../conventional_approach` 的相对路径共用。

## 输出怎么解

网络输出 `[1, 21, 6300]`，**通道优先**，`element(通道 f, 候选 c) = data[c + f*6300]`：

```
0~3    颜色得分（蓝/红/白/紫）
4~12   9 类得分（哨兵/英雄/工程/三号步兵/四号步兵/五号步兵/前哨站/基地底/基地顶）
13~20  4 个关键点 (x,y)
```

流程：9 类取最大 → 阈值 0.5 → 颜色取最大 → 关键点逆 letterbox（`(x-pad_x)/scale`）→ 越界丢弃 → `NMSBoxes`（IoU 0.45）。

**关键点顺序是左上 → 左下 → 右下 → 右上**，和 PnP 需要的 `左下 → 左上 → 右上 → 右下` 方向相反，必须过 `cornersFromNNKeypoints()` 再解算；不重排时 tvec 不变但姿态会翻约 180°（实测）。

## 构建 / 运行

```bash
cd YOLO_approach
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build

# 图片（文件夹或单张图片）
./build/armor_yolo_detector "../Project_materials/含真值图像/图1"
./build/armor_yolo_detector "../Project_materials/含真值图像/图1/Image_1791695492312_183.jpg"

# 视频：<视频> <输出视频> <输出CSV>（都可省略）
./build/armor_yolo_video
```

## 配置

参数集中在仓库根目录的 `config/default.yaml`（可用 `--config` 指定别的文件）：

| YAML 段 | 含义 |
|---|---|
| `camera.stills` / `camera.video` | 两套内参（图片程序用 stills，视频用 video） |
| `armor` | 板宽 135/225、灯条长 55、重投影误差上限 |
| `yolo.model` | 检测模型路径（默认 FP32 版本） |
| `yolo.input_w / input_h` | letterbox 目标尺寸（模型契约，通常不要改） |
| `yolo.pad_value` | letterbox 填充灰度（默认 124） |
| `yolo.conf_thresh / nms_iou` | 类别阈值与 NMS IoU |
| `yolo.num_classes / num_colors` | 输出通道拆分（9 类 4 色） |
| `yolo.max_dist_m` | 视频距离门限；设 0 表示不启用 |
| `run.yolo_image` / `run.yolo_video` | 输入、输出、检测 CSV、编码 |

配置里的相对路径相对**仓库根目录**解析。

输出：

| 文件 | 内容 |
|---|---|
| `output/debug/yolo_N.png` | 标注图 |
| `output/debug/detections_yolo.csv` | 逐块检测结果（追加写），供误差评测脚本使用 |
| `output/yolo_video.avi` | 531 帧标注视频（MJPG） |
| `output/yolo_video.csv` | 视频逐块结果 |

## 注意

- 视频用配置的 `camera.video`（1440×1080 + 畸变），静态图用 `camera.stills`，**不能混**。
- 视频程序有 **2.5m 距离门限**（`yolo.max_dist_m`）：本视频真装甲板在 1.4~2m，而背景图案（货架、横幅）会被误检到 4~5.8m；换场景时改配置即可。
- letterbox 必须记录 `scale/pad` 并做逆变换，否则框会整体偏移；`swapRB=true` 不能漏，漏了颜色分支会稳定判成蓝色。
