# include "func.h"
# include <opencv2/core/types.hpp>
# include <opencv2/dnn.hpp>
# include <opencv2/imgproc.hpp>
# include <algorithm>
# include <cmath>

Input makeBlob(const cv::Mat& bgr)
{
    Input in;
    if (bgr.empty()) return in;

    // ---- 1. letterbox：等比缩放到 640x480 以内，空白用灰 124 填充 ----
    // 两种素材其实都刚好等比例（1440x1080 和 960x720 都是 4:3），不会产生填充，
    // 但代码不能假设这一点，否则换个分辨率的素材就错位。
    const double scale = std::min(static_cast<double>(kYoloInW) / bgr.cols,
                                  static_cast<double>(kYoloInH) / bgr.rows);
    const int new_w = std::max(1, static_cast<int>(std::lround(bgr.cols * scale)));
    const int new_h = std::max(1, static_cast<int>(std::lround(bgr.rows * scale)));

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    const int pad_w = kYoloInW - new_w;
    const int pad_h = kYoloInH - new_h;
    const int left  = pad_w / 2;
    const int top   = pad_h / 2;

    cv::Mat canvas;
    cv::copyMakeBorder(resized, canvas, top, pad_h - top, left, pad_w - left,
                       cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124));

    // ---- 2. BGR->RGB、/255、HWC->CHW，得到 1x3x480x640 的 CV_32F ----
    // swapRB=true 就是 BGR->RGB。⚠️ 漏了它颜色分支会稳定判成蓝色（实测过），
    // 这是这套网络最快的自查点。
    cv::dnn::blobFromImage(canvas, in.blob, 1.0 / 255.0, cv::Size(kYoloInW, kYoloInH),
                           cv::Scalar(), /*swapRB=*/true, /*crop=*/false, CV_32F);

    in.info.scale = static_cast<float>(scale);
    in.info.pad_x = static_cast<float>(left);
    in.info.pad_y = static_cast<float>(top);
    return in;
}
