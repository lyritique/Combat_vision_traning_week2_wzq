// 视频批处理：逐帧跑传统视觉方案，输出标注视频 + 每块装甲板一行的 CSV
//
// 用法:
//   ./armor_video                          # 用下面的默认路径
//   ./armor_video <视频> <输出视频> <输出CSV>
//
// 和图片程序的区别只有三点：
//   1. 内参用 cameraVideo()（1440x1080 + 5 个畸变系数），不是 cameraImages()
//   2. 灯条是红色的，阈值 60
//   3. 打开数字过滤（negative 是有效的剔除信号）

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "func.h"
#include "PnP_solver.h"

int main(int argc, char** argv)
{
    //一堆参数，硬编码也可以满足任务需要
    const std::string video_path = (argc > 1) ? argv[1]
        : "/home/lyritique/project/Task2/Project_materials/相机内参与视频/Video_1791696012009_214.avi";
    const std::string out_video  = (argc > 2) ? argv[2] : "output/traditional_video.avi";
    const std::string out_csv    = (argc > 3) ? argv[3] : "output/traditional_video.csv";

    const CameraParams cam   = cameraVideo();      // 视频专用内参
    const LightColor   color = LightColor::RED;    // 视频里的灯条是红色
    const int          thresh = 60;
    const bool use_digit_filter = true;            // 视频里 negative 要剔除

    // 模型路径：优先用 CMake 传进来的绝对路径
    std::string model_path = "models/mlp.onnx";
#ifdef ARMOR_MODEL_DIR
    {
        const std::string abs_path = std::string(ARMOR_MODEL_DIR) + "/mlp.onnx";
        if (std::filesystem::exists(abs_path)) model_path = abs_path;
    }
#endif

    // 输出目录（路径可能只是文件名，parent 为空时跳过）
    for (const auto& p : {out_video, out_csv})
    {
        const auto parent = std::filesystem::path(p).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "打不开视频: " << video_path << "\n";
        return 1;
    }

    const int    total  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const double fps    = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    const int    width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "视频: " << width << "x" << height << "  " << total << " 帧  " << fps << " fps\n";

    // MJPG 写 avi，兼容性最稳；需要 mp4 可以最后用 ffmpeg 转
    cv::VideoWriter writer(out_video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           fps, cv::Size(width, height));
    if (!writer.isOpened())
    {
        std::cerr << "打不开输出视频: " << out_video << "\n";
        return 1;
    }

    std::ofstream csv(out_csv);
    if (!csv)
    {
        std::cerr << "打不开输出 CSV: " << out_csv << "\n";
        return 1;
    }
    // 每块留下的装甲板一行；没有检测到的帧不写行（从 frame 序号能看出来）
    csv << "frame,bars,candidates,kept,number,conf,x_m,y_m,z_m,reproj_px\n";

    int frames = 0, frames_with = 0, cand_total = 0, kept_total = 0;
    std::map<std::string, int> number_count;

    //开始计时
    const auto t0 = std::chrono::steady_clock::now();

    cv::Mat frame;
    while (cap.read(frame))
    {
        if (frame.empty()) break;
        //闪电三连鞭
        cv::Mat mask = preprocess(frame, color, thresh);
        std::vector<LightBar> bars = findLights(mask);
        std::vector<Armor> armors = matchBar(bars);

        std::vector<ArmorDetection> detections;
        detections.reserve(armors.size());
        for (auto& a : armors)
        {
            ArmorDetection d;
            d.armor = a;
            d.digit = classify(extractNumber(frame, d.armor, false), model_path);
            d.pose  = positionSolve(d.armor, cam);
            detections.push_back(d);
        }

        const int cand = static_cast<int>(detections.size());
        cand_total += cand;

        eraseIgnoreClasses(detections, use_digit_filter);

        const int kept = static_cast<int>(detections.size());
        kept_total += kept;
        if (kept > 0) ++frames_with;
        //AI写的统计很详尽啊

        for (const auto& d : detections)
        {
            ++number_count[d.digit.valid ? d.digit.label : "invalid"];
            csv << frames << ',' << bars.size() << ',' << cand << ',' << kept << ','
                << d.digit.label << ',' << d.digit.conf << ','
                << d.pose.tvec[0] << ',' << d.pose.tvec[1] << ',' << d.pose.tvec[2] << ','
                << d.pose.reproj_err << '\n';
        }

        // 画保留的候选，再写进输出视频（没检测到的帧也照写，保持时间轴完整）
        cv::Mat view = frame.clone();
        drawResult(view, bars, detections);
        writer.write(view);

        ++frames;
        if (frames % 100 == 0) std::cout << "  已处理 " << frames << "/" << total << " 帧\n";
    }

    writer.release();
    csv.close();

    //结束计时
    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const double keep_rate = cand_total > 0 ? 100.0 * kept_total / cand_total : 0.0;

    std::cout << "\n处理完成：\n"
              << "  帧数          : " << frames << "\n"
              << "  有装甲板的帧  : " << frames_with
              << "（" << (frames ? 100.0 * frames_with / frames : 0.0) << "%）\n"
              << "  候选 / 保留   : " << cand_total << " / " << kept_total
              << "（保留率 " << keep_rate << "%）\n"
              << "  号码分布      : ";
    for (const auto& kv : number_count) std::cout << kv.first << "=" << kv.second << " ";
    std::cout << "\n  耗时          : " << sec << " s（"
              << (sec > 0 ? frames / sec : 0.0) << " fps）\n"
              << "  标注视频      : " << out_video << "\n"
              << "  结果 CSV      : " << out_csv << "\n";

    return 0;
}
