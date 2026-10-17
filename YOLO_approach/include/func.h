# ifndef _YOLO_FUNC_
# define _YOLO_FUNC_

# include <opencv2/core/mat.hpp>
# include <opencv2/dnn.hpp>
# include "PnP_solver.h"      // PoseResult（画距离文字用）
# include <string>
# include <vector>

// 网络输入尺寸（模型声明 输入 [1,3,480,640]）
constexpr int kYoloInW = 640;
constexpr int kYoloInH = 480;

// letterbox 的逆变换参数：原图 --scale--> 缩放图 --pad--> 640x480 输入
struct LetterboxInfo
{
    float scale = 1.0f;   // 等比缩放系数
    float pad_x = 0.0f;   // 左侧填充的像素数
    float pad_y = 0.0f;   // 顶部填充的像素数
};

// 一次预处理的结果：网络输入 blob + 逆变换参数
struct Input
{
    cv::Mat       blob;   // 1x3x480x640，CV_32F，RGB 顺序，值域 0~1
    LetterboxInfo info;
};

// 预处理：letterbox（等比缩放 + 灰 124 填充）-> BGR 转 RGB -> /255 -> NCHW
Input makeBlob(const cv::Mat& bgr);

// 单个检测结果。关键点按网络原始顺序：左上、左下、右下、右上
struct YoloDetection
{
    std::vector<cv::Point2f> keypoints;   // 4 个点，已换算回原图坐标
    cv::Rect2f               box;         // 关键点外接框（NMS 用）
    int   color = -1;                     // 0蓝 1红 2白 3紫
    int   cls   = -1;                     // 0..8，见 yoloClassName
    float score = 0.0f;
};

const char* yoloColorName(int color);
const char* yoloClassName(int cls);

// 加载模型（只加载一次）。失败时把原因写进 err，返回 false。
bool loadYoloModel(const std::string& model_path, std::string& err);

// 推理 + 后处理：类别阈值 0.5、NMS IoU 0.45
std::vector<YoloDetection> detectArmorsYolo(const cv::Mat& bgr);

// 把检测结果画到图上：青色四边形 = 装甲板，红点 = 关键点，
// 文字 = 颜色/类别/置信度 + 距离（poses 非空时）
// poses 可为空指针；不为空时长度必须与 dets 一致。
void drawYoloDetections(cv::Mat& canvas,
                        const std::vector<YoloDetection>& dets,
                        const std::vector<PoseResult>* poses = nullptr);

# endif
