# include "func.h"
# include <opencv2/imgproc.hpp>   // cv::boundingRect
# include <iostream>

// 模型只加载一次
static cv::dnn::Net g_net;
static std::string  g_loaded_path;

const char* yoloColorName(int color)
{
    static const char* names[] = {"blue", "red", "white", "purple"};
    return (color >= 0 && color < 4) ? names[color] : "?";
}

const char* yoloClassName(int cls)
{
    // 与作业资料一致：0哨兵 1英雄 2工程 3三号步兵 4四号步兵 5五号步兵
    //                  6前哨站 7基地(底) 8基地(顶)
    static const char* names[] = {"sentry", "hero", "engineer", "infantry3",
                                  "infantry4", "infantry5", "outpost",
                                  "base_bottom", "base_top"};
    return (cls >= 0 && cls < 9) ? names[cls] : "?";
}

bool loadYoloModel(const std::string& model_path, std::string& err)
{
    if (!g_net.empty() && g_loaded_path == model_path) return true;

    try
    {
        g_net = cv::dnn::readNetFromONNX(model_path);
        g_loaded_path = model_path;
        std::cout << "YOLO 模型已加载: " << model_path << "\n";
        return true;
    }
    catch (const cv::Exception& e)
    {
        err = std::string("模型加载失败：") + e.what();
        if (err.find("FLOAT16") != std::string::npos)
        {
            err += "\n提示：这个 onnx 是 FP16 权重，OpenCV 4.5.4 的 dnn 不支持。\n"
                   "     先跑：python3 scripts/convert_onnx_fp16_to_fp32.py "
                   "YOLO_approach/models/Infantry.onnx\n"
                   "     然后在 models/ 下得到 Infantry_fp32.onnx，程序会自动优先用它。";
        }
        return false;
    }
}

std::vector<YoloDetection> detectArmorsYolo(const cv::Mat& bgr, const YoloCfg& cfg)
{
    std::vector<YoloDetection> result;
    if (bgr.empty() || g_net.empty()) return result;

    const Input in = makeBlob(bgr, cfg);
    if (in.blob.empty() || in.info.scale <= 0.0f) return result;

    cv::Mat out;
    try
    {
        g_net.setInput(in.blob);
        out = g_net.forward();          // 期望 [1,21,6300]
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "YOLO 前向失败: " << e.what() << "\n";
        return result;
    }

    // 通道 = 颜色数 + 类别数 + 4x2 关键点（模型契约，不随配置乱改）
    const int expect_ch = cfg.num_color + cfg.num_cls + 8;
    if (out.dims != 3 || out.size[1] != expect_ch)
    {
        std::cerr << "YOLO 输出形状不对，期望通道数 " << expect_ch << "，实际 "
                  << (out.dims == 3 ? out.size[1] : -1) << "\n";
        return result;
    }

    const int num_ch = out.size[1];
    const int num_c  = out.size[2];     // 6300 = 80x60 + 40x30 + 20x15
    const int kp_off = cfg.num_color + cfg.num_cls; // 关键点起始通道（本模型 = 13）

    // 输出是"通道优先"布局：element(通道 f, 候选 c) = data[c + f * num_c]
    // 展平成一维视图，避免 3 维下标写错
    const cv::Mat flat = out.reshape(1, 1);
    const float* data = flat.ptr<float>(0);
    auto at = [&](int f, int c) { return data[c + f * num_c]; };

    const float kConf = cfg.conf;
    std::vector<YoloDetection> candidates;

    for (int c = 0; c < num_c; ++c)
    {
        // 9 个类别得分取最大（输出已经 sigmoid 过，没有 objectness 通道）
        float best_score = 0.0f;
        int   best_cls   = -1;
        for (int k = 0; k < cfg.num_cls; ++k)
        {
            const float s = at(4 + k, c);
            if (s > best_score) { best_score = s; best_cls = k; }
        }
        if (best_score < kConf) continue;

        // 颜色：前 4 行取最大
        int   color      = 0;
        float best_color = at(0, c);
        for (int k = 1; k < cfg.num_color; ++k)
        {
            const float s = at(k, c);
            if (s > best_color) { best_color = s; color = k; }
        }

        // 4 个关键点：letterbox 坐标 -> 原图坐标
        std::vector<cv::Point2f> kp(4);
        for (int i = 0; i < 4; ++i)
        {
            const float x = at(kp_off + 2 * i,     c);
            const float y = at(kp_off + 2 * i + 1, c);
            kp[i].x = (x - in.info.pad_x) / in.info.scale;
            kp[i].y = (y - in.info.pad_y) / in.info.scale;
        }

        const cv::Rect2f box = cv::boundingRect(kp);
        // 越界的点丢弃（说明这个候选不可靠）
        if (box.x < 0 || box.y < 0 ||
            box.x + box.width > bgr.cols || box.y + box.height > bgr.rows) continue;

        YoloDetection d;
        d.keypoints = kp;
        d.box       = box;
        d.color     = color;
        d.cls       = best_cls;
        d.score     = best_score;
        candidates.push_back(d);
    }

    // NMS：同一块装甲板经常出 2~3 个重复候选，不做会满屏重复框
    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    for (const auto& d : candidates)
    {
        boxes.push_back(cv::Rect(d.box));
        scores.push_back(d.score);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, kConf, cfg.nms, keep);

    result.reserve(keep.size());
    for (int i : keep) result.push_back(candidates[i]);
    return result;
}
