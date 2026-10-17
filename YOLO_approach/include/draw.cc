# include "func.h"
# include <opencv2/imgproc.hpp>

void drawYoloDetections(cv::Mat& canvas,
                        const std::vector<YoloDetection>& dets,
                        const std::vector<PoseResult>* poses)
{
    if (canvas.empty()) return;

    const cv::Scalar quad_color(255, 255, 0);    // 青
    const cv::Scalar point_color(0, 0, 255);     // 红

    for (size_t i = 0; i < dets.size(); ++i)
    {
        const YoloDetection& d = dets[i];

        // 四个关键点（网络顺序：左上、左下、右下、右上，正好绕一圈）
        std::vector<cv::Point> poly(d.keypoints.begin(), d.keypoints.end());
        cv::polylines(canvas, poly, true, quad_color, 2, cv::LINE_AA);
        for (const auto& p : d.keypoints)
            cv::circle(canvas, p, 3, point_color, -1);

        cv::Point2f center(0.0f, 0.0f);
        for (const auto& p : d.keypoints) center += p;
        center *= 0.25f;

        // 两行文字；按检测序号上下错开，避免两个框靠得近时完全叠在一起
        const float y0 = center.y - 22.0f - 34.0f * static_cast<float>(i % 2);
        cv::putText(canvas,
                    cv::format("%s %s %.2f", yoloColorName(d.color),
                               yoloClassName(d.cls), d.score),
                    cv::Point2f(center.x + 8.0f, y0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, quad_color, 2, cv::LINE_AA);

        if (poses && i < poses->size() && (*poses)[i].valid)
            cv::putText(canvas, cv::format("%.2fm", (*poses)[i].tvec[2]),
                        cv::Point2f(center.x + 8.0f, y0 + 20.0f),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, quad_color, 2, cv::LINE_AA);
    }
}
