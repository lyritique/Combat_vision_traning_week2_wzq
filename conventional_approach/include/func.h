# ifndef _FUNC_ 
# define _FUNC_
# include <opencv2/core/cvstd.hpp>
# include <opencv2/core/mat.hpp>
# include <opencv2/core/types.hpp>
# include <opencv2/imgcodecs.hpp>
# include <iostream>
# include <string>
# include <vector>

// 灯条颜色：真实视频里是红色灯条，仿真图（图1/图2/图3）里是蓝色灯条
enum class LightColor { RED, BLUE };

// 一根灯条：findLights 的输出，也是 matchLights 的输入
struct LightBar
{
    cv::RotatedRect rect;      // 拟合出的旋转矩形（配对、取角点都用它）
    cv::Point2f center;        // 灯条中心（等于 rect.center）
    float length = 0;          // 长边（像素）
    float width  = 0;          // 短边（像素）
    float angle  = 0;          // 长轴方向角，归一到 (-90, 90]，竖直 ≈ ±90
    float area   = 0;          // 轮廓面积（像素）
    float fill   = 0;          // 填充率 = 轮廓面积 / 外接矩形面积
};

// findLights 的筛选参数，先给经验值，之后可以外置到 config 文件
//
// 只用"形状"作判据，不用绝对像素量：
//   - 面积：绝对下限/上限会随分辨率、距离变化（近距离大装甲板、远距离小装甲板
//     面积能差两个数量级），属于对素材的过拟合，所以不参与筛选，只在结果里记录。
//   - 填充率：灯条高光/过曝时中心会打白（R≈G≈B），通道差分反而接近 0，
//     轮廓变成空心，填充率会明显偏低，所以也不作判据。
struct LightParams
{
    float min_aspect = 1.8f;      // 长宽比下限：灯条是细长的
    float max_aspect = 15.0f;     // 长宽比上限：过细的多半是边缘/划痕
    float max_tilt   = 45.0f;     // 长轴与竖直方向的最大夹角（度）
};

std::vector<cv::Mat> input_image (std::string );
cv::Mat  preprocess(const cv::Mat& row_img, LightColor color = LightColor::RED, int diff_thresh = 60);
std::vector<LightBar> findLights(const cv::Mat& mask, const LightParams& params = LightParams{});

// 一块装甲板候选 = 配对成功的一对灯条
struct ArmorCandidate
{
    int left  = -1;                     // 左灯条在 bars 里的下标
    int right = -1;                     // 右灯条
    std::vector<cv::Point2f> corners;   // 4 个角点，顺序固定：
                                        // 左灯条下 → 左灯条上 → 右灯条上 → 右灯条下
    cv::Point2f center;                 // 两根灯条中心的中点（≈ 装甲板中心）
    float ratio        = 0;             // 中心距 / 平均灯条长（装甲板比例，无量纲）
    float length_ratio = 0;             // 两根灯条的长度比
    float angle_diff   = 0;             // 两根灯条的倾角差（度）
    float perp_dev     = 0;             // 中心连线与灯条长轴的垂直偏差（度）
    float cost         = 0;             // 配对代价，去重时取小
    bool  large        = false;         // 比值更像大装甲板（225×55）
};

// matchLights 的成对约束参数。全部是比值/角度，与分辨率、距离无关；
// 默认值来自实测（视频真灯条对：长度比 0.91/0.98、倾角差 2.0/1.5、
// 垂直偏差 3.7/2.0、中心距/灯条长 2.21/2.34）
struct MatchParams
{
    float min_length_ratio = 0.75f;   // 长度比下限：同一块板的两根灯条应等长
    float max_angle_diff   = 10.0f;   // 倾角差上限：两根应平行
    float max_perp_dev     = 15.0f;   // 垂直偏差上限
    float min_dist_ratio   = 1.4f;    // 中心距/灯条长 下限
    // 上限：小装甲板标称 135/55≈2.45，但光晕会把实测值压到 1.6 左右（图1 实测 1.64）。
    // 不能放到 5 附近去"顺便兼容大装甲板"——只找到部分灯条时，那样会把跨装甲板的
    // 两根灯条配成一对（实测出现过 70px 中心距、解出 z=1.66m 的离谱结果）。
    // 大装甲板 225mm 的比例（标称 4.09）等确认了板型再单独放开。
    float max_dist_ratio   = 3.6f;
    float large_threshold  = 3.2f;    // 比值超过它先按大装甲板标记
};

std::vector<ArmorCandidate> matchLights(const std::vector<LightBar>& bars,
                                       const MatchParams& params = MatchParams{});

// 一套相机内参。作业给了两套，分辨率和畸变都不同，绝对不能混用
struct CameraParams
{
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::vector<double> dist;          // k1,k2,p1,p2,k3；无畸变时全 0
    cv::Mat K() const;                 // 3x3 内参矩阵
    cv::Mat D() const;                 // 畸变系数（1xN）
};

// 静态图（960×720）：fx=fy=869.1169, cx=480, cy=360, 畸变全 0
CameraParams cameraImages();
// 视频（1440×1080）：fx=2362.2796, fy=2336.1916, cx=679.1751, cy=529.0686 + 5 个畸变系数
CameraParams cameraVideo();

// 一块装甲板的位姿解算结果
struct ArmorPose
{
    cv::Vec3d tvec;                    // 装甲板中心在相机系下的位置，单位 m
    cv::Vec3d rvec;                    // 姿态（Rodrigues 向量）
    double reproj_err = 0;             // 重投影误差（像素）
    bool valid = false;                // 是否可信
    std::string reason;                // 不可信的原因（便于统计和调试）
};

/**
 * @brief 位置解算：由装甲板四角点求它在相机坐标系下的位姿
 *
 * 坐标系约定：相机光学系 x 右 / y 下 / z 前；tvec 是装甲板几何中心；
 * 物体点用 mm，算完转成 m（真值也是 m）。
 */
ArmorPose positionSolve(const ArmorCandidate& armor, const CameraParams& cam,
                        float armor_width_mm = 135.0f, float bar_length_mm = 55.0f,
                        double max_reproj_err = 5.0);

// 数字识别的结果
struct DigitResult
{
    int cls = -1;            // 类别下标 0..8
    float conf = 0;          // softmax 之后的置信度
    std::string label;       // 类别名："1".."5"/"outpost"/"guard"/"base"/"negative"
    bool valid = false;      // 是否拿到结果
    std::string reason;      // 失败原因
};

/**
 * @brief 数字识别第一步：把装甲板拉正、截出数字区域，做成网络输入
 *
 * 四角点 → 规范图（小装甲板 32×28，大装甲板 54×28，两根灯条落在左右两条边上、
 * 竖直方向占中间 12 像素）→ 取中间 20×28 作为数字区域 → 灰度 → /255。
 * 返回 CV_32F 的 1×560（行优先展平），可直接 setInput。
 *
 * @param bgr         原始 BGR 图
 * @param armor       装甲板候选（要用它的 corners）
 * @param large_armor 是否大装甲板（225mm），影响规范图宽度
 */
cv::Mat extractNumber(const cv::Mat& bgr, const ArmorCandidate& armor, bool large_armor = false);

/**
 * @brief 数字识别第二步：跑 mlp.onnx，出号码和置信度
 *
 * 网络输出是 9 维 logits（最后一层没有 Softmax），这里自己做 softmax 再取最大。
 * 类别顺序读同目录的 label.txt：1,2,3,4,5,outpost,guard,base,negative
 */
DigitResult classify(const cv::Mat& input, const std::string& model_path = "models/mlp.onnx");

void correctCorners();
void eraseIgnoreClasses();
void positionSolve();
# endif
