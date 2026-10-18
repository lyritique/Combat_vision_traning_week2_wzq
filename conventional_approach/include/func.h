# ifndef _FUNC_ 
# define _FUNC_
# include <opencv2/core/cvstd.hpp>
# include <opencv2/core/mat.hpp>
# include <opencv2/core/types.hpp>
# include <opencv2/imgcodecs.hpp>
# include "PnP_solver.h"     // CameraParams / PoseResult / solveArmorPose
# include "config.hpp"       // LightCfg / PairCfg / DigitCfg / ArmorCfg
# include <iostream>
# include <string>
# include <vector>

// 灯条颜色：真实视频里是红色灯条，仿真图（图1/图2/图3）里是蓝色灯条
enum class LightColor { RED, BLUE };

struct LightBar 
{
    cv::RotatedRect frame;
    float width {};
    float length {};
    float angle {};
};

struct Armor
{
    const LightBar* leftBar;
    const LightBar* rightBar;
    std::vector<cv::Point2f> corners;
};

struct DigitResult
{
    int cls = -1;            // 类别下标 0..8
    float conf = 0;          // softmax 之后的置信度
    std::string label;       // "1".."5" / "outpost" / "guard" / "base" / "negative"
    bool valid = false;      // 是否拿到结果
    std::string reason;      // 失败原因（便于调试和统计）
};

struct ArmorDetection //一块板的检测结果
{
    Armor       armor;      // matchBar + correctCorners 的结果
    DigitResult digit;      // 这块候选的数字识别结果
    PoseResult  pose;       // 位姿解算结果（来自 PnP_solver）
};

// names 不为空时按同样顺序返回每张图的完整路径（做误差表要按文件名对真值）
std::vector<cv::Mat> input_image (std::string path, std::vector<std::string>* names = nullptr);
cv::Mat  preprocess(const cv::Mat& row_img, LightColor color, int diff_thresh, const LightCfg& cfg);
std::vector<LightBar> findLights(const cv::Mat& binary_img, const LightCfg& cfg);
bool isArmorPair(const LightBar& left, const LightBar& right, const PairCfg& cfg);
std::vector<Armor> matchBar (std::vector<LightBar>& bars, const PairCfg& cfg);
cv::Mat extractNumber (cv::Mat& bgr, Armor& armor, const DigitCfg& cfg, bool large_armor = false);
DigitResult classify(const cv::Mat& input, const DigitCfg& cfg);

// "red" / "blue" -> LightColor
LightColor parseColor(const std::string& name);

// 剔除无效/重复候选
//   use_digit_filter ：是否用数字识别结果过滤（negative / 识别失败）
//   min_digit_conf   ：数字置信度下限，低于它的候选当成不可信（实测视频里 <0.4 的
//                      5 帧恰好同时是距离异常帧，用 0.5 能干净地把它们挡掉）
void eraseIgnoreClasses(std::vector<ArmorDetection>& detections,
                        const DigitCfg& digit_cfg,
                        const PairCfg& pair_cfg,
                        bool use_digit_filter);

// 把检测结果画到图上（绿色细框=灯条，青色粗框=装甲板，红点=中心，文字=距离+号码）
// 图片程序和视频程序共用
void drawResult(cv::Mat& canvas,
                const std::vector<LightBar>& bars,
                const std::vector<ArmorDetection>& detections);

// 传统方案的薄封装：从 Armor（两根灯条 + 四角点）解位姿，核心算法在 PnP_solver.cc
PoseResult positionSolve(const Armor& armor, const CameraParams& cam, const ArmorCfg& cfg);
# endif
