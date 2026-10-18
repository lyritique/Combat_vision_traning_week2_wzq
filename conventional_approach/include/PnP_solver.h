# ifndef PNP_SOLVER_H_
# define PNP_SOLVER_H_

// PnP 位置解算模块（传统视觉方案和神经网络方案共用）
//
// 两个方案的区别只在"四个图像点从哪来"：
//   传统方案：correctCorners 从两根灯条的对称轴端点得到
//   神经网络方案：装甲板检测网络直接输出 4 个关键点（需要重排，见 cornersFromNNKeypoints）
// 从四个点到位姿的算法完全一样，所以抽成这个模块。

# include <opencv2/core.hpp>
# include <string>
# include <vector>

// 一套相机内参。作业给了两套，分辨率和畸变都不同，绝对不能混用
struct CameraParams
{
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::vector<double> dist;      // k1,k2,p1,p2,k3；无畸变时全 0

    cv::Mat K() const;             // 3x3 内参矩阵
    cv::Mat D() const;             // 畸变系数（1xN；无畸变时空 Mat）
};

// 静态图：960x720，fx=fy=869.1168824543144, cx=480, cy=360，畸变全 0
CameraParams cameraImages();

// 视频：1440x1080，fx=2362.2796, fy=2336.1916, cx=679.1751, cy=529.0686
//      畸变 [-0.098562, 0.111731, -0.003835, -0.002042, 0]
CameraParams cameraVideo();

// 位置解算结果
struct PoseResult
{
    cv::Vec3d tvec;                  // 装甲板中心在相机系下的位置，单位 m（x 右 / y 下 / z 前）
    cv::Vec3d rvec;                  // 姿态（Rodrigues 向量）
    double reproj_err = 0;           // 重投影误差（像素）
    bool valid = false;              // 是否可信
    std::string reason;              // 不可信的原因（便于统计和调试）
};

/**
 * @brief 由装甲板四角点解算位姿
 *
 * 角点顺序必须严格是从"左下"开始：
 *     左灯条下 → 左灯条上 → 右灯条上 → 右灯条下
 * 顺序错了解出来的位姿会是镜像的，tvec 的模长可能还对，但姿态翻 180°。
 *
 * 物体点用 mm：(0, ±armor_width_mm/2, ±bar_length_mm/2)
 *   x = 装甲板法线方向（板面为 x=0 的平面）
 *   y = 宽度方向（两根灯条中心的连线）
 *   z = 高度方向（灯条方向）
 * 输出 tvec 的单位是 m（真值也是 m）。
 *
 * @param corners        4 个图像点（像素）
 * @param cam            该素材对应的内参
 * @param armor_width_mm 两根灯条的中心距：小装甲板 135，大装甲板 225
 * @param bar_length_mm  灯条长度：55
 * @param max_reproj_err 重投影误差上限（像素），超了判为无效
 */
PoseResult solveArmorPose(const std::vector<cv::Point2f>& corners,
                          const CameraParams& cam,
                          float armor_width_mm = 135.0f,
                          float bar_length_mm = 55.0f,
                          double max_reproj_err = 5.0);

/**
 * @brief 把神经网络方案的关键点顺序重排成本模块要求的顺序
 *
 * 装甲板检测网络的关键点顺序（作业资料）：左上、左下、右下、右上（顺时针一圈）
 * 本模块要求：左灯条下、左灯条上、右灯条上、右灯条下（逆时针一圈）
 * 两者正好是反方向，直接喂会把位姿镜像掉。
 */
std::vector<cv::Point2f> cornersFromNNKeypoints(const std::vector<cv::Point2f>& kpts);

# endif
