#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
# include <opencv2/opencv.hpp>
# include <opencv2/core/saturate.hpp>
#include <vector>
#include <cmath>        // atan2
#include <algorithm>    // std::max / std::min / std::sort
#include <fstream>      // 读 label.txt
# include "func.h"  








/**
 * @brief 预处理：把 BGR 原图变成"候选灯条"的二值图（白=候选，黑=背景）
 *
 * 灯条是强光源、颜色单一：真实视频里偏红（R 明显大于 B），仿真图 图1/图2/图3
 * 里偏蓝（B 明显大于 R）。所以先用通道差分挑出"目标颜色多、相反颜色少"的像素，
 * 再阈值二值化，最后用形态学清理。颜色判据保留成参数，两种都能跑。
 *
 * @param row_img     输入的 BGR 三通道图（不会被修改）
 * @param color       要找哪种颜色的灯条：LightColor::RED / LightColor::BLUE
 * @param diff_thresh 差分阈值，超过它才算候选。实测参考：
 *                    视频（红灯条）50~60；图3（蓝灯条）40~60；图1（蓝灯条偏暗）20~30
 * @return CV_8UC1 二值图，255 = 可能是灯条，0 = 背景
 */
cv::Mat  preprocess(const cv::Mat& row_img, LightColor color, int diff_thresh)
{
    // ---- 1. 颜色差分：挑出"目标颜色多、相反颜色少"的像素 ----
    // OpenCV 里图像是 BGR 顺序：ch[0]=B, ch[1]=G, ch[2]=R
    std::vector<cv::Mat> ch;
    cv::split(row_img, ch);

    // subtract 自带饱和处理（结果夹在 0~255），不会像 uchar 直接相减那样
    // 把负数变成 206 这种回绕值
    cv::Mat diff;
    if (color == LightColor::RED)
        cv::subtract(ch[2], ch[0], diff);   // R - B：红灯条数值大，灰/蓝背景接近 0
    else
        cv::subtract(ch[0], ch[2], diff);   // B - R：蓝灯条数值大

    // ---- 2. 二值化：差分值超过阈值才算候选 ----
    cv::Mat mask;
    cv::threshold(diff, mask, diff_thresh, 255, cv::THRESH_BINARY);

    // ---- 3. 形态学：先开运算去掉小噪点，再闭运算补上被阈值切断的灯条 ----
    // 3x3 的核比灯条宽度（十几像素）小得多，不会把灯条整条抹掉
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    return mask;
}

/**
 * @brief 从预处理得到的二值图里找出所有"像灯条"的候选
 *
 * 思路：二值图里每个白色连通域都可能是灯条，也可能是背景亮块、反光或残留噪点，
 *       所以先用 minAreaRect 把每个轮廓拟合成一个旋转矩形，再用形状约束筛：
 *       长宽比 + 长轴与竖直方向的夹角。这两个都是"比值/角度"，与分辨率、距离无关。
 *       面积和填充率只记录、不筛选（原因见 LightParams 的注释）。
 *       这一步是粗筛，宁可多放过一些，配对时还有更严格的成对约束兜底。
 *
 * @param mask   预处理输出的二值图（CV_8UC1，白色 = 候选）
 * @param params 筛选阈值，默认值见 LightParams；之后要外置到 config
 * @return 通过筛选的灯条列表（matchLights 的输入）
 */
std::vector<LightBar> findLights(const cv::Mat& mask, const LightParams& params)
{
    std::vector<LightBar> bars;

    // 1) 取外轮廓。RETR_EXTERNAL 只拿最外层，避免同一根灯条的内部空洞被重复计数
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 2) 逐个轮廓拟合旋转矩形，再用几何指标筛选
    for (const auto& contour : contours)
    {
        const float area = static_cast<float>(cv::contourArea(contour));   // 仅记录
        const cv::RotatedRect light = cv::minAreaRect(contour);

        // minAreaRect 的 width/height 不保证哪个是长边，直接相除会时对时错。
        // 这里把四条边量出来，统一成 长边/短边
        cv::Point2f pt[4];
        light.points(pt);
        const float edge01 = static_cast<float>(cv::norm(pt[0] - pt[1]));
        const float edge12 = static_cast<float>(cv::norm(pt[1] - pt[2]));
        const float length = std::max(edge01, edge12);
        const float width  = std::min(edge01, edge12);
        if (width < 1.0f) continue;                       // 退化成一个点或一条线

        // 判据 1：长宽比。灯条必然是细长的，这个比值与图片分辨率、拍摄距离无关
        const float aspect = length / width;
        if (aspect < params.min_aspect || aspect > params.max_aspect) continue;

        const float fill = area / (length * width);       // 仅记录：高光时灯条可能空心

        // 长轴方向角：先算它与 x 轴的夹角，归一到 (-90, 90]，竖直灯条 ≈ ±90
        const cv::Point2f dir = (edge01 > edge12) ? (pt[1] - pt[0]) : (pt[2] - pt[1]);
        float angle = std::atan2(dir.y, dir.x) * 180.0f / static_cast<float>(CV_PI);
        while (angle <= -90.0f) angle += 180.0f;
        while (angle >   90.0f) angle -= 180.0f;

        // 判据 2：长轴与竖直方向的夹角。灯条在画面里都是接近竖直的（受车体姿态
        // 影响会偏一些），斜躺的结构不是灯条
        if (90.0f - std::abs(angle) > params.max_tilt) continue;

        LightBar bar;
        bar.rect   = light;
        bar.center = light.center;
        bar.length = length;
        bar.width  = width;
        bar.angle  = angle;
        bar.area   = area;
        bar.fill   = fill;
        bars.push_back(bar);
    }

    return bars;
}

// 把两个方向角的差归一到 [0, 90]：灯条长轴是"无向"的，差 180° 等于没差
static float angleDiff(float a, float b)
{
    float d = std::fmod(std::abs(a - b), 180.0f);
    if (d > 90.0f) d = 180.0f - d;
    return d;
}

// 取一根灯条长轴的两个端点（= 灯条上下两端）。图像坐标 y 向下，y 大的算"下"端
static void barEndpoints(const LightBar& bar, cv::Point2f& top, cv::Point2f& bottom)
{
    const float rad = bar.angle * static_cast<float>(CV_PI) / 180.0f;
    const cv::Point2f dir(std::cos(rad), std::sin(rad));      // 长轴单位向量
    const cv::Point2f e1 = bar.center + dir * (bar.length * 0.5f);
    const cv::Point2f e2 = bar.center - dir * (bar.length * 0.5f);
    if (e1.y < e2.y) { top = e1; bottom = e2; }
    else             { top = e2; bottom = e1; }
}

/**
 * @brief 把一对灯条整理成装甲板的 4 个角点
 *
 * 角点 = 两根灯条各自长轴的两个端点（不是灯条中心）。
 * 顺序必须固定，后面 positionSolve 的物体点要严格按这个顺序对应：
 *   左灯条下 → 左灯条上 → 右灯条上 → 右灯条下
 * 顺序错了解出来的位姿会是镜像的。
 */
static std::vector<cv::Point2f> makeCorners(const LightBar& L, const LightBar& R)
{
    cv::Point2f lt, lb, rt, rb;
    barEndpoints(L, lt, lb);
    barEndpoints(R, rt, rb);
    return { lb, lt, rt, rb };
}

/**
 * @brief 把所有灯条两两配对，找出"可能是一块装甲板"的组合
 *
 * 用四条成对约束（都是比值/角度，与分辨率、距离无关）：
 *   1. 两根灯条长度接近（同一块板的两根灯条物理长度相同、距离也相近）
 *   2. 两根灯条平行（倾角差小）
 *   3. 两根中心的连线与灯条长轴大致垂直（旋转不变，比"两端 y 差小"更稳）
 *   4. 中心距 / 平均灯条长 落在装甲板的比例区间内
 * 最后按代价贪心去重：一根灯条只能属于一块装甲板。
 *
 * @param bars   灯条列表（findLights 的输出）
 * @param params 成对约束参数
 * @return 装甲板候选列表
 */
std::vector<ArmorCandidate> matchLights(const std::vector<LightBar>& bars, const MatchParams& params)
{
    std::vector<ArmorCandidate> candidates;

    // ---- 1. 枚举所有两两组合，逐条套约束 ----
    for (size_t i = 0; i < bars.size(); ++i)
    {
        for (size_t j = i + 1; j < bars.size(); ++j)
        {
            const LightBar& A = bars[i];
            const LightBar& B = bars[j];

            // 分出左右：x 小的算左灯条；x 相同时用下标决定，避免同一个组合被算两次
            const bool a_is_left = (A.center.x != B.center.x)
                                 ? (A.center.x < B.center.x) : (i < j);
            const LightBar& L = a_is_left ? A : B;
            const LightBar& R = a_is_left ? B : A;
            const int li = a_is_left ? static_cast<int>(i) : static_cast<int>(j);
            const int ri = a_is_left ? static_cast<int>(j) : static_cast<int>(i);

            // 约束 1：长度接近
            const float length_ratio = std::min(L.length, R.length) / std::max(L.length, R.length);
            if (length_ratio < params.min_length_ratio) continue;

            // 约束 2：平行
            const float angle_diff = angleDiff(L.angle, R.angle);
            if (angle_diff > params.max_angle_diff) continue;

            // 约束 3：灯条长轴与"两根中心连线"大致垂直
            const cv::Point2f delta = R.center - L.center;
            const float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);
            if (dist < 1.0f) continue;
            const float link_angle = std::atan2(delta.y, delta.x) * 180.0f / static_cast<float>(CV_PI);
            const float perp_dev = std::abs(90.0f - angleDiff(L.angle, link_angle));
            if (perp_dev > params.max_perp_dev) continue;

            // 约束 4：装甲板的宽高比（135/55 或 225/55）
            const float dist_ratio = dist / (0.5f * (L.length + R.length));
            if (dist_ratio < params.min_dist_ratio || dist_ratio > params.max_dist_ratio) continue;

            ArmorCandidate c;
            c.left         = li;
            c.right        = ri;
            c.corners      = makeCorners(L, R);
            c.center       = 0.5f * (L.center + R.center);
            c.ratio        = dist_ratio;
            c.length_ratio = length_ratio;
            c.angle_diff   = angle_diff;
            c.perp_dev     = perp_dev;
            c.large        = dist_ratio > params.large_threshold;

            // 代价：越像标准装甲板（比值贴近 2.45、两根越等长越平行）代价越小
            c.cost = std::abs(dist_ratio - 2.4545f)
                   + (1.0f - length_ratio) * 2.0f
                   + angle_diff / 10.0f
                   + perp_dev / 15.0f;
            candidates.push_back(c);
        }
    }

    // ---- 2. 按代价从小到大贪心领取，一根灯条只归一块装甲板 ----
    std::sort(candidates.begin(), candidates.end(),
              [](const ArmorCandidate& a, const ArmorCandidate& b) { return a.cost < b.cost; });

    std::vector<ArmorCandidate> result;
    std::vector<char> used(bars.size(), 0);
    for (const auto& c : candidates)
    {
        if (used[c.left] || used[c.right]) continue;
        used[c.left] = used[c.right] = 1;
        result.push_back(c);
    }
    return result;
}

cv::Mat CameraParams::K() const
{
    return (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                      0.0, fy, cy,
                                      0.0, 0.0, 1.0);
}

cv::Mat CameraParams::D() const
{
    if (dist.empty()) return cv::Mat();
    cv::Mat d(dist);            // vector -> Nx1
    return d.reshape(1, 1);     // solvePnP 要 1xN 的行向量
}

CameraParams cameraImages()
{
    CameraParams c;
    c.fx = 869.1168824543144;
    c.fy = 869.1168824543144;
    c.cx = 480.0;
    c.cy = 360.0;
    c.dist = {0.0, 0.0, 0.0, 0.0, 0.0};   // plumb_bob，全 0
    return c;
}

CameraParams cameraVideo()
{
    CameraParams c;
    c.fx = 2362.2795952460974;
    c.fy = 2336.1916035376717;
    c.cx = 679.17513921636282;
    c.cy = 529.06858905837635;
    c.dist = {-0.098561592108101559, 0.11173066531932681,
              -0.0038352945867108721, -0.0020421443354436304, 0.0};
    return c;
}

/**
 * @brief 装甲板模型的 4 个物体点（单位 mm）
 *
 * 约定：装甲板平面取 x = 0；y 沿"两根灯条中心的连线"（宽度方向，左负右正）；
 * z 沿灯条方向（高度方向，上负下正）；原点在装甲板几何中心。
 * 顺序必须与 ArmorCandidate::corners 一致：
 *   左灯条下 → 左灯条上 → 右灯条上 → 右灯条下
 *
 * @param width_mm    两根灯条的中心距：小装甲板 135，大装甲板 225
 * @param bar_len_mm  灯条长度：55
 */
static std::vector<cv::Point3f> armorObjectPoints(float width_mm, float bar_len_mm)
{
    const float hw = width_mm * 0.5f;      // 半宽
    const float hh = bar_len_mm * 0.5f;    // 半高
    return {
        cv::Point3f(0.0f, -hw,  hh),       // 左灯条下
        cv::Point3f(0.0f, -hw, -hh),       // 左灯条上
        cv::Point3f(0.0f,  hw, -hh),       // 右灯条上
        cv::Point3f(0.0f,  hw,  hh),       // 右灯条下
    };
}

ArmorPose positionSolve(const ArmorCandidate& armor, const CameraParams& cam,
                        float armor_width_mm, float bar_length_mm, double max_reproj_err)
{
    ArmorPose pose;

    if (armor.corners.size() != 4)
    {
        pose.reason = "角点数不是 4，无法解算";
        return pose;
    }

    const std::vector<cv::Point3f> obj = armorObjectPoints(armor_width_mm, bar_length_mm);
    const cv::Mat K = cam.K();
    const cv::Mat D = cam.D();

    // 平面目标用 IPPE；它会给出两个解（平面正反两个歧义方向），全部取出来比一比
    std::vector<cv::Mat> rvecs, tvecs;
    const int nsol = cv::solvePnPGeneric(obj, armor.corners, K, D, rvecs, tvecs,
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
        cv::projectPoints(obj, rvecs[i], tvecs[i], K, D, reproj);
        double err = 0;
        for (size_t k = 0; k < reproj.size(); ++k)
            err += cv::norm(reproj[k] - armor.corners[k]);
        err /= static_cast<double>(reproj.size());
        if (err < best_err) { best_err = err; best = i; }
    }

    pose.rvec = cv::Vec3d(rvecs[best]);
    pose.tvec = cv::Vec3d(tvecs[best]) / 1000.0;   // 物体点是 mm，结果转成 m
    pose.reproj_err = best_err;

    // 可信度检查：数值健全、目标在相机前方、重投影误差不超标
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

// 数字区域的规范尺寸与位置（与模型训练时的约定一致）
static constexpr int kDigitRoiW    = 20;   // 数字区域宽（两根灯条之间）
static constexpr int kDigitRoiH    = 28;   // 数字区域高
static constexpr int kLightLenWarp = 12;   // 灯条在规范图里的竖直长度

cv::Mat extractNumber(const cv::Mat& bgr, const ArmorCandidate& armor, bool large_armor)
{
    if (bgr.empty() || armor.corners.size() != 4) return cv::Mat();

    // 小装甲板 135mm 宽、大装甲板 225mm 宽，高度都是 55mm，所以规范图宽度不同
    const int warp_w = large_armor ? 54 : 32;
    const int warp_h = kDigitRoiH;
    const int top_y    = (warp_h - kLightLenWarp) / 2 - 1;   // 7
    const int bottom_y = top_y + kLightLenWarp;              // 19

    // 目标点顺序必须和 corners 一致：左灯条下、左灯条上、右灯条上、右灯条下
    const std::vector<cv::Point2f> dst = {
        {0.0f,           static_cast<float>(bottom_y)},
        {0.0f,           static_cast<float>(top_y)},
        {static_cast<float>(warp_w), static_cast<float>(top_y)},
        {static_cast<float>(warp_w), static_cast<float>(bottom_y)}};

    const cv::Mat M = cv::getPerspectiveTransform(armor.corners, dst);
    cv::Mat warped;
    cv::warpPerspective(bgr, warped, M, cv::Size(warp_w, warp_h));

    // 数字夹在两根灯条中间，所以取中间的 20 列
    cv::Mat gray;
    cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);
    const cv::Mat roi = gray(cv::Rect((warp_w - kDigitRoiW) / 2, 0, kDigitRoiW, kDigitRoiH)).clone();

    // /255 归一化；展平顺序是"行优先"（28 行 × 20 列），不能用 blobFromImage 默认的 1x1x28x20
    cv::Mat input;
    roi.convertTo(input, CV_32F, 1.0 / 255.0);
    return input.reshape(1, 1);
}

// 读 label.txt（和模型放在同一目录）；读不到就用内置的类别顺序
static std::vector<std::string> loadLabels(const std::string& model_path)
{
    static const char* kFallback[] = {"1", "2", "3", "4", "5", "outpost", "guard", "base", "negative"};
    std::vector<std::string> labels(kFallback, kFallback + 9);

    const size_t slash = model_path.find_last_of("/\\");
    const std::string dir = (slash == std::string::npos) ? std::string(".") : model_path.substr(0, slash);
    std::ifstream fin(dir + "/label.txt");
    if (!fin) return labels;                       // 没有 label.txt 也能跑

    std::vector<std::string> read;
    std::string line;
    while (std::getline(fin, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) read.push_back(line);
    }
    return read.size() == 9 ? read : labels;
}

DigitResult classify(const cv::Mat& input, const std::string& model_path)
{
    DigitResult result;
    if (input.empty())
    {
        result.reason = "输入为空（extractNumber 没拿到 ROI）";
        return result;
    }

    // 模型只加载一次；路径变了就重新加载
    static cv::dnn::Net net;
    static std::string loaded_path;
    if (net.empty() || loaded_path != model_path)
    {
        try
        {
            net = cv::dnn::readNetFromONNX(model_path);
            loaded_path = model_path;
        }
        catch (const cv::Exception& e)
        {
            result.reason = std::string("模型加载失败：") + e.what();
            return result;
        }
    }

    net.setInput(input);
    const cv::Mat logits = net.forward();          // 1x9，未过 softmax
    if (logits.total() != 9)
    {
        result.reason = "网络输出不是 9 类";
        return result;
    }

    // 手写 softmax：先减去最大值避免 exp 溢出
    cv::Mat x = logits.reshape(1, 1).clone();
    double max_val = 0.0;
    cv::minMaxLoc(x, nullptr, &max_val);
    cv::Mat expx;
    cv::exp(x - max_val, expx);
    const cv::Mat prob = expx / cv::sum(expx)[0];

    int best = 0;
    for (int i = 1; i < 9; ++i)
        if (prob.at<float>(0, i) > prob.at<float>(0, best)) best = i;

    const std::vector<std::string> labels = loadLabels(model_path);
    result.cls   = best;
    result.conf  = prob.at<float>(0, best);
    result.label = labels[best];
    result.valid = true;
    return result;
}



















