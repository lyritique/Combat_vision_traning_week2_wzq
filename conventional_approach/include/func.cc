# include <cstddef>
# include <opencv2/core/mat.hpp>
# include <opencv2/core/types.hpp>
# include <opencv2/imgproc.hpp>
# include <opencv2/opencv.hpp>
# include <opencv2/core/saturate.hpp>
# include <string>
# include <vector>
# include "func.h"  
# include <fstream>







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
cv::Mat  preprocess(const cv::Mat& row_img, LightColor color, int diff_thresh, const LightCfg& cfg)
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
    // 核比灯条宽度（十几像素）小得多，不会把灯条整条抹掉
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {cfg.morph, cfg.morph});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    return mask;
}

/* ------- 你原来手写逐像素的版本（保留在这里备查，不参与编译） -------
 * 里面有四处坑，已经在下面对应位置改好：
 *   1. row_ptr_resultImg 必须取 result_img 的行指针；原先写成 row_img，
 *      结果一直在往输入图上写，result_img 从没被赋值（实测返回全黑图）；
 *   2. 单通道结果图的下标必须是 [x]，不能是 [x * row_img.channels()]，
 *      否则每行会越界写 3 倍长度，堆被写坏；
 *   3. 是 saturate_cast，不是 staurate_cast（拼写）；
 *   4. 原版返回的是差分灰度图，没有做阈值；findContours 需要二值图。
 */
/*
cv::Mat preprocess(const cv::Mat& row_img)
{
    cv::Mat result_img (row_img.rows, row_img.cols, CV_8UC1);
    for ( size_t y=0; y<row_img.rows; y++ )
    {
        for ( size_t x=0; x<row_img.cols; x++ )
        {
            unsigned char* row_ptr_rowImg = row_img.ptr<unsigned char> ( y );
            unsigned char* data_ptr_rowImg = &row_ptr_rowImg[ x*row_img.channels() ];
            unsigned char* row_ptr_resultImg = result_img.ptr<unsigned char> ( y );
            unsigned char* data_ptr_resultImg = &row_ptr_resultImg[x];
            *data_ptr_resultImg = cv::saturate_cast<uchar>( data_ptr_rowImg[2] - data_ptr_rowImg[0] );
        }
    }
    return result_img;
}
*/

std::vector<LightBar> findLights(const cv::Mat& binary_img, const LightCfg& cfg)
{
    //第一步，构建轮廓集
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    //第二步，contour逐个筛
    std::vector<LightBar> bars ;

    for (const auto& contour :contours)
    {
        //每个contour,用minAreaRect框定可以倾斜的矩形
        const auto light_possible {cv::minAreaRect(contour)};

        //计算light相关参数
        cv::Point2f pt[4];
        light_possible.points(pt);
        const float edge01 = static_cast<float>(cv::norm(pt[0] - pt[1]));//计算两边
        const float edge12 = static_cast<float>(cv::norm(pt[1] - pt[2]));
        const float length = std::max(edge01, edge12);//选出长边
        const float width  = std::min(edge01, edge12);//选出短边

        //排除不合条件的light
        //第一，过小
        if (width < cfg.min_w || static_cast<float>(contour.size()) < cfg.min_pts) continue;
        //第二，过长过短
        //一般情况下，装甲板倾斜程度不会太高
        if (length > cfg.max_aspect * width || length < cfg.min_aspect * width) continue;
        //第三，倾斜程度
        //长边的两点相减，得到长边方向向量,再解出与竖轴夹角
        const cv::Point2f dir = (edge01 > edge12) ? (pt[1] - pt[0]) : (pt[2] - pt[1]);
        float angle = std::atan2(dir.y, dir.x) * 180.0f / static_cast<float>(CV_PI);
        while (angle <= -90.0f) angle += 180.0f;
        while (angle >   90.0f) angle -= 180.0f;
        if (90.0f - std::abs(angle) > cfg.max_tilt) continue;
        
        //最后封装成bar
        LightBar bar ;
        
        bar.frame = light_possible;
        bar.length = length;
        bar.width = width;
        bar.angle  = angle ;    
        bars.push_back(bar);
        

    }


    return bars;

}

//

static float angleDiff(float a, float b)
{
    float d = std::fmod(std::abs(a - b), 180.0f);
    return d > 90.0f ? 180.0f - d : d;
}

bool isArmorPair(const LightBar& left,const LightBar& right, const PairCfg& cfg)
{
    //同一块装甲板的两根灯条长度近似相同
    const float length_ratio = std::min(left.length, right.length) / std::max(left.length, right.length);
    if (length_ratio < cfg.min_len) return false;

    //角度接近
    if (angleDiff(left.angle, right.angle) > cfg.max_ang) return false;
    
    //中心水平应该相近（但做不到），有两中心的连线应与灯条长轴接近垂直
    const cv::Point2f delta = right.frame.center - left.frame.center;
    const float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    if (dist < 1.0f) return false;
    const float link_angle = std::atan2(delta.y, delta.x) * 180.0f / static_cast<float>(CV_PI);
    if (std::abs(90.0f - angleDiff(left.angle, link_angle)) > cfg.max_perp) return false;

    //间距合理
    const float ratio = dist / (0.5f * (left.length + right.length));
    if (ratio < cfg.min_gap || ratio > cfg.max_gap) return false;
    
    return true;
}

/*
static std::vector<cv::Point2f> correctCorners(LightBar left,LightBar right)
{
    cv::Point2f points_left [4];
    left.frame.points(points_left);
    cv::Point2f points_right [4];
    left.frame.points(points_right);
    std::vector<cv::Point2f> corners 
    {
        points_left[1],
        points_right[2],
        points_right[3],
        points_left[0]
    };
    return corners;
}
*/

//
static void barEndpoints(const LightBar& bar, cv::Point2f& top, cv::Point2f& bottom)
{
    cv::Point2f pt[4];
    bar.frame.points(pt);

    // 两条边里长的那条就是灯条方向
    const float e0 = static_cast<float>(cv::norm(pt[0] - pt[1]));
    const float e1 = static_cast<float>(cv::norm(pt[1] - pt[2]));
    cv::Point2f dir = (e0 > e1) ? (pt[1] - pt[0]) : (pt[2] - pt[1]);
    dir = dir / static_cast<float>(cv::norm(dir));      // 单位化

    // 从中心往两端各推 length/2，就得到对称轴的两个端点
    const cv::Point2f end1 = bar.frame.center + dir * (bar.length * 0.5f);
    const cv::Point2f end2 = bar.frame.center - dir * (bar.length * 0.5f);

    // 图像坐标 y 向下，y 大的是"下"端
    if (end1.y < end2.y) { top = end1; bottom = end2; }
    else                 { top = end2; bottom = end1; }
}

static std::vector<cv::Point2f> correctCorners(const LightBar& left, const LightBar& right)
{
    cv::Point2f lt, lb, rt, rb;
    barEndpoints(left,  lt, lb);
    barEndpoints(right, rt, rb);
    return { lb, lt, rt, rb };
}

std::vector<Armor> matchBar (std::vector<LightBar>& bars, const PairCfg& cfg)
{
    //排序，按照中心位置
    std::sort(bars.begin(), bars.end(), 
    [](const LightBar& a, const LightBar& b) {return a.frame.center.x < b.frame.center.x;});

    std::vector<Armor> armors;

    //把所有灯条两两配对，找出一块装甲板的组合
    for (size_t i{}; i < bars.size() ; i++)
    {
        for (size_t j = i + 1; j < bars.size(); j++)
        {
            const LightBar& left  = bars[i];
            const LightBar& right  = bars[j];


            if (isArmorPair(left, right, cfg))
            {
                Armor armor ;
                armor.leftBar = &left;
                armor.rightBar = &right;
                armor.corners = correctCorners(left,right);
                armors.emplace_back(armor);
            }            
            
        }
    }
    return armors;
}

cv::Mat extractNumber (cv::Mat& bgr, Armor& armor, const DigitCfg& cfg, bool large_armor)
{
    //
    if (bgr.empty() || armor.corners.size() != 4) return cv::Mat();

    // 规范图尺寸与灯条位置来自配置（小/大装甲板宽度不同）
    const int warp_w   = large_armor ? cfg.w_large : cfg.w_small;
    const int warp_h   = cfg.h;
    const int top_y    = cfg.top_y;
    const int bottom_y = cfg.bot_y;
    const std::vector<cv::Point2f> dst = {                   
    {0.0f, static_cast<float>(bottom_y)},                        
    {0.0f, static_cast<float>(top_y)},                         
    {static_cast<float>(warp_w), static_cast<float>(top_y)},    
    {static_cast<float>(warp_w), static_cast<float>(bottom_y)}}; 

    const cv::Mat transMat {cv::getPerspectiveTransform(armor.corners, dst)};
    cv::Mat warped;
    cv::warpPerspective(bgr, warped, transMat, cv::Size(warp_w, warp_h));

    cv::Mat gray;
    cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY); 
    const cv::Mat roi = gray(cv::Rect((warp_w - cfg.roi_w) / 2, 0, cfg.roi_w, cfg.h)).clone();

    cv::threshold(roi, roi, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat input;
    roi.convertTo(input, CV_32F, 1.0 / 255.0);
    return input.reshape(1, 1);
}


//
static std::vector<std::string> loadLabels(const DigitCfg& cfg)
{
    static const char* kFallback[] = {"1", "2", "3", "4", "5", "outpost", "guard", "base", "negative"};
    std::vector<std::string> labels(kFallback, kFallback + 9);

    // 优先用配置里的 labels 路径；没有就退回模型同目录的 label.txt；再没有用内置顺序
    std::ifstream fin(cfg.labels);
    if (!fin && !cfg.model.empty())
    {
        const size_t slash = cfg.model.find_last_of("/\\");
        const std::string dir = (slash == std::string::npos) ? std::string(".") : cfg.model.substr(0, slash);
        fin.open(dir + "/label.txt");
    }
    if (!fin) return labels;                    // 没有也能跑

    std::vector<std::string> read;
    std::string line;
    while (std::getline(fin, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) read.push_back(line);
    }
    return read.size() == 9 ? read : labels;
}

DigitResult classify(const cv::Mat& input_blob, const DigitCfg& cfg)
{
    DigitResult result ;
    const std::string& model_path = cfg.model;
    //启动model,输入
    static cv::dnn::Net net;
    static std::string loaded_path;
    if (input_blob.empty())
    {
        result.reason = "输入为空（extractNumber 没拿到 ROI）";
        return result;
    }
    // 模型只加载一次；加载失败不要崩，返回失败原因（作业要求处理这种情况）
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

    // 前向：输入形状不对也会抛异常，同样兜住
    cv::Mat logits;
    try
    {
        net.setInput(input_blob);
        logits = net.forward();
    }
    catch (const cv::Exception& e)
    {
        result.reason = std::string("网络前向失败：") + e.what();
        return result;
    }

    //CV下来的softmax
    /*float max_logit = *std::max_element(logits.begin<float>(), logits.end<float>());
    cv::Mat softmax_logits;
    cv::exp(logits - max_logit, softmax_logits);
    float sum = static_cast<float>(cv::sum(softmax_logits)[0]);
    softmax_logits /= sum;

    double confidence;
    cv::Point class_id_point;
    minMaxLoc(softmax_logits.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
    int label_id = class_id_point.x;*/

    //softmax
    //单通道单行mat
    cv::Mat x = logits.reshape(1, 1);
    //选出max
    double max_logit {};
    cv::minMaxLoc(x, nullptr, &max_logit);
    //防溢出式用指数归一到0~1
    cv::Mat expx;
    cv::exp(x - max_logit, expx);
    const cv::Mat prob = expx / cv::sum(expx)[0];

    //best为最大的下标
    int best = 0;
    const float* p = prob.ptr<float>(0);   
    for (int i = 0; i < prob.cols; ++i)
    {
        if (p[i] > p[best]) best = i;
    }

    //取label，记下置信度，封装
    const std::vector<std::string> labels = loadLabels(cfg);
    result.cls   = best;
    result.conf  = prob.at<float>(0, best);
    result.label = labels[best];
    result.valid = true;
    return result;
}


static float centerRatio(const Armor& a)
{
    const cv::Point2f d = a.rightBar->frame.center - a.leftBar->frame.center;
    const float dist = cv::norm(d); 
    return dist / (0.5f * (a.leftBar->length + a.rightBar->length));
}

// 两个四边形的交并比（四角点按顺序连成的凸四边形正好能用）
static float quadIoU(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b)
{
    std::vector<cv::Point2f> inter;
    const float ia = static_cast<float>(cv::intersectConvexConvex(a, b, inter));
    if (ia <= 0.0f) return 0.0f;
    const float uni = static_cast<float>(cv::contourArea(a)) + static_cast<float>(cv::contourArea(b)) - ia;
    return uni > 0.0f ? ia / uni : 0.0f;
}

void eraseIgnoreClasses(std::vector<ArmorDetection>& detections,
                        const DigitCfg& digit_cfg,
                        const PairCfg& pair_cfg,
                        bool use_digit_filter)
{
    //剔除数字识别失败、置信度过低、位姿解算不可信的候选
    std::vector<ArmorDetection> candidate;
    for (const auto& detection : detections)
    {
        if (use_digit_filter)
        {
            if (!detection.digit.valid) continue;
            if (detection.digit.label == "negative") continue;
            if (detection.digit.conf < digit_cfg.conf_min) continue;   // 低置信度 = 不可信
        }
        if (!detection.pose.valid) continue;          // 深度为负、重投影误差过大等，直接丢掉
        candidate.push_back(detection);
    }

    // 择优排序 有数字就按数字置信度；没数字（或关掉数字过滤）就按"比例多接近 2.45"
    auto score = [use_digit_filter, &pair_cfg](const ArmorDetection& detection)
    {
        if (use_digit_filter && detection.digit.valid) return detection.digit.conf;
        return 1.0f / (1.0f + std::abs(centerRatio(detection.armor) - pair_cfg.nominal));
    };
    std::sort(candidate.begin(), candidate.end(),
              [&](const ArmorDetection& a, const ArmorDetection& b) { return score(a) > score(b); });

    // 贪心 与已保留的候选"共用灯条"或"框重叠"就丢弃
    std::vector<ArmorDetection> out;
    for (const auto& detection : candidate)
    {
        bool conflict = false;
        for (const auto& o : out)
        {
            // 共用灯条：指针直接比，比算几何重叠干脆
            const bool share_bar = detection.armor.leftBar  == o.armor.leftBar  ||
                                   detection.armor.leftBar  == o.armor.rightBar ||
                                   detection.armor.rightBar == o.armor.leftBar  ||
                                   detection.armor.rightBar == o.armor.rightBar;
            if (share_bar || quadIoU(detection.armor.corners, o.armor.corners) > 0.5f) { conflict = true; break; }
        }
        if (!conflict) out.push_back(detection);
    }
    detections.swap(out);
}

// ---------------- 传统方案的位姿解算封装 ----------------
// 核心算法在 PnP_solver.cc（传统方案和神经网络方案共用）；
// 这里只负责把 Armor 的四角点转交给它。
PoseResult positionSolve(const Armor& armor, const CameraParams& cam, const ArmorCfg& cfg)
{
    return solveArmorPose(armor.corners, cam,
                          cfg.use_large ? cfg.large_w : cfg.small_w,
                          cfg.bar_len, cfg.max_reproj);
}

LightColor parseColor(const std::string& name)
{
    return (name == "red") ? LightColor::RED : LightColor::BLUE;
}

// ---------------- 结果绘制（图片程序与视频程序共用） ----------------
void drawResult(cv::Mat& canvas,
                const std::vector<LightBar>& bars,
                const std::vector<ArmorDetection>& detections)
{
    if (canvas.empty()) return;

    const cv::Scalar bar_color(0, 255, 0);       // 绿：灯条
    const cv::Scalar armor_color(255, 255, 0);   // 青：装甲板
    const cv::Scalar center_color(0, 0, 255);    // 红：中心

    // 灯条：旋转矩形 + 中心
    for (const auto& b : bars)
    {
        cv::Point2f p[4];
        b.frame.points(p);
        for (int k = 0; k < 4; ++k)
            cv::line(canvas, p[k], p[(k + 1) % 4], bar_color, 1, cv::LINE_AA);
        cv::circle(canvas, b.frame.center, 2, center_color, -1);
    }

    // 装甲板：四角点连成四边形 + 距离/号码
    for (const auto& d : detections)
    {
        const Armor& a = d.armor;
        if (a.corners.size() == 4)
        {
            for (size_t k = 0; k < a.corners.size(); ++k)
                cv::line(canvas, a.corners[k], a.corners[(k + 1) % a.corners.size()],
                         armor_color, 2, cv::LINE_AA);
        }

        const cv::Point2f center = (a.leftBar->frame.center + a.rightBar->frame.center) * 0.5f;
        cv::circle(canvas, center, 3, center_color, -1);

        if (d.pose.valid)
            cv::putText(canvas, cv::format("%.2fm", d.pose.tvec[2]), center + cv::Point2f(8, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, armor_color, 2, cv::LINE_AA);

        if (d.digit.valid)
            cv::putText(canvas, cv::format("%s %.2f", d.digit.label.c_str(), d.digit.conf),
                        center + cv::Point2f(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        d.digit.label == "negative" ? cv::Scalar(0, 0, 255) : bar_color,
                        2, cv::LINE_AA);
    }
}

