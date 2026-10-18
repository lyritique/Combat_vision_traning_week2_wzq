# ifndef ARMOR_CONFIG_H_
# define ARMOR_CONFIG_H_

// 统一配置：所有会随素材/机器/场景变化的参数都从 config/default.yaml 读，
// 代码里不再散落硬编码。
//
// 注意："模型/算法契约"不进配置：
//   21 通道布局、9 类 4 色的顺序、角点顺序、4 点 PnP、输出已做 sigmoid 等。

# include <string>
# include <vector>

# include "PnP_solver.h"

// 一套相机内参
struct CamCfg
{
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::vector<double> dist;
};

// 装甲板物理模型（PnP 的"尺子"）
struct ArmorCfg
{
    float small_w = 135.0f;
    float large_w = 225.0f;
    float bar_len = 55.0f;
    float max_reproj = 5.0f;
    bool  use_large = false;
};

// 灯条提取
struct LightCfg
{
    int   morph = 3;
    int   min_pts = 10;
    float min_w = 1.0f;
    float min_aspect = 1.8f;
    float max_aspect = 15.0f;
    float max_tilt = 45.0f;
};

// 灯条配对 / 去重打分
struct PairCfg
{
    float min_len = 0.75f;
    float max_ang = 10.0f;
    float max_perp = 15.0f;
    float min_gap = 1.4f;
    float max_gap = 3.6f;
    float nominal = 2.4545f;
};

// 数字识别
struct DigitCfg
{
    std::string model = "conventional_approach/models/mlp.onnx";
    std::string labels = "conventional_approach/models/label.txt";
    float conf_min = 0.5f;
    bool  filter_image = false;
    bool  filter_video = true;
    int   w_small = 32;
    int   w_large = 54;
    int   h = 28;
    int   roi_w = 20;
    int   top_y = 7;
    int   bot_y = 19;
};

// 神经网络方案
struct YoloCfg
{
    std::string model = "YOLO_approach/models/Infantry_fp32.onnx";
    int   in_w = 640;
    int   in_h = 480;
    int   pad = 124;
    int   num_cls = 9;
    int   num_color = 4;
    float conf = 0.5f;
    float nms = 0.45f;
    float max_dist = 2.5f;
};

// 单个程序的运行参数
struct RunCfg
{
    std::string input, out_dir, out_video, out_csv, det_csv;
    std::string color = "blue";
    std::string codec = "MJPG";
    int thresh = 60;
};

struct EvalCfg
{
    float max_px = 80.0f;
};

struct Config
{
    CamCfg   stills, video;
    ArmorCfg armor;
    LightCfg light;
    PairCfg  pair;
    DigitCfg digit;
    YoloCfg  yolo;
    RunCfg   conv_image, conv_video, yolo_image, yolo_video;
    EvalCfg  eval;
    std::string path;

    // 读取失败抛 std::runtime_error
    static Config load(const std::string& path);

    // 按素材选一套内参
    CameraParams camera(bool use_video) const;

    float armorWidthMm() const { return armor.use_large ? armor.large_w : armor.small_w; }
};

// --config 优先，否则用 PROJECT_ROOT/config/default.yaml
std::string findConfigPath(int argc, char** argv);

// 取出位置参数（跳过 --config <path>），供 main 解析 input/thresh/color
std::vector<std::string> positionalArgs(int argc, char** argv);

// 相对路径 -> 仓库根目录下（保证从任意工作目录运行都能找到文件）
std::string resolvePath(const std::string& p);

# endif
