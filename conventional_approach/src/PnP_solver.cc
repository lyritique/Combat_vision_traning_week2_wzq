// PnP 位置解算模块的实现（传统方案和神经网络方案共用）

#include "PnP_solver.h"

#include <opencv2/calib3d.hpp>   // solvePnPGeneric / projectPoints
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>

// ---------------- 内参 ----------------

cv::Mat CameraParams::K() const
{
    return (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                      0.0, fy, cy,
                                      0.0, 0.0, 1.0);
}

cv::Mat CameraParams::D() const
{
    if (dist.empty()) return cv::Mat();     // 无畸变：传空矩阵
    cv::Mat d(dist);                        // vector -> Nx1
    return d.reshape(1, 1);                 // solvePnP 要 1xN 行向量
}

// ---------------- 物体点 ----------------

// 装甲板模型的 4 个物体点（单位 mm），顺序与 solveArmorPose 要求的角点顺序一一对应
static std::vector<cv::Point3f> armorObjectPoints(float armor_width_mm, float bar_length_mm)
{
    const float hw = armor_width_mm * 0.5f;   // 半宽（两灯条中心距的一半）
    const float hh = bar_length_mm * 0.5f;    // 半高（灯条长度的一半）
    return {
        cv::Point3f(0.0f, -hw,  hh),          // 左灯条下
        cv::Point3f(0.0f, -hw, -hh),          // 左灯条上
        cv::Point3f(0.0f,  hw, -hh),          // 右灯条上
        cv::Point3f(0.0f,  hw,  hh),          // 右灯条下
    };
}

// ---------------- 解算 ----------------

PoseResult solveArmorPose(const std::vector<cv::Point2f>& corners,
                          const CameraParams& cam,
                          float armor_width_mm,
                          float bar_length_mm,
                          double max_reproj_err)
{
    PoseResult pose;

    if (corners.size() != 4)
    {
        pose.reason = "角点数不是 4，无法解算";
        return pose;
    }

    const std::vector<cv::Point3f> object_points = armorObjectPoints(armor_width_mm, bar_length_mm);
    const cv::Mat K = cam.K();
    const cv::Mat D = cam.D();

    // 平面目标用 IPPE；它会给出两个解（平面正反两个歧义方向），全部取出来比一比
    std::vector<cv::Mat> rvecs, tvecs;
    const int nsol = cv::solvePnPGeneric(object_points, corners, K, D, rvecs, tvecs,
                                        false, cv::SOLVEPNP_IPPE);
    if (nsol <= 0 || rvecs.empty())
    {
        pose.reason = "solvePnP 没有解";
        return pose;
    }

    // 择优标准：把物体点投回图像，看哪个解跟检测到的角点更贴
    int best = -1;
    double best_err = 1e18;
    for (int i = 0; i < nsol; ++i)
    {
        std::vector<cv::Point2f> reproj;
        cv::projectPoints(object_points, rvecs[i], tvecs[i], K, D, reproj);

        double err = 0.0;
        for (size_t k = 0; k < reproj.size(); ++k)
            err += cv::norm(reproj[k] - corners[k]);
        err /= static_cast<double>(reproj.size());

        if (err < best_err) { best_err = err; best = i; }
    }

    pose.rvec = cv::Vec3d(rvecs[best]);
    pose.tvec = cv::Vec3d(tvecs[best]) / 1000.0;   // 物体点是 mm，结果转成 m
    pose.reproj_err = best_err;

    // 可信度检查
    for (int k = 0; k < 3; ++k)
    {
        if (!std::isfinite(pose.tvec[k]))
        {
            pose.reason = "解出 NaN/Inf";
            return pose;
        }
    }
    if (pose.tvec[2] <= 0.0)
    {
        pose.reason = "深度 ≤ 0（目标解到相机后面）";
        return pose;
    }
    if (pose.reproj_err > max_reproj_err)
    {
        pose.reason = "重投影误差过大";
        return pose;
    }

    pose.valid = true;
    return pose;
}

// ---------------- 神经网络方案的关键点重排 ----------------

std::vector<cv::Point2f> cornersFromNNKeypoints(const std::vector<cv::Point2f>& kpts)
{
    if (kpts.size() != 4) return {};
    // 输入：左上 kpts[0]、左下 kpts[1]、右下 kpts[2]、右上 kpts[3]
    // 输出：左下、左上、右上、右下
    return {kpts[1], kpts[0], kpts[3], kpts[2]};
}
