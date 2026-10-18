// 神经网络方案的视频批处理：逐帧 YOLO 检测 -> 关键点重排 -> PnP -> 标注视频 + CSV
//
// 用法:
//   ./armor_yolo_video                          # 用下面的默认路径
//   ./armor_yolo_video <视频> <输出视频> <输出CSV>

#include <opencv2/opencv.hpp>
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
    // ---------- 参数（config/default.yaml，可用命令行覆盖） ----------
    const Config cfg = Config::load(findConfigPath(argc, argv));
    const std::vector<std::string> pos = positionalArgs(argc, argv);

    const std::string video_path = pos.size() > 0 ? pos[0] : cfg.yolo_video.input;
    const std::string out_video  = pos.size() > 1 ? pos[1] : cfg.yolo_video.out_video;
    const std::string out_csv    = pos.size() > 2 ? pos[2] : cfg.yolo_video.out_csv;
    const std::string model_path = cfg.yolo.model;

    std::string err;
    if (!loadYoloModel(model_path, err))
    {
        std::cerr << err << "\n请把模型放到: " << model_path << "\n";
        return 1;
    }

    // 视频专用内参（1440x1080 + 5 个畸变系数）
    const CameraParams cam = cfg.camera(true);

    // 距离门限：本视频里真装甲板在 1.4~1.6m，而背景误检集中在 4m 以外
    // （货架、横幅上的图案）。配置项 yolo.max_dist_m；换成 0 表示不启用。
    const float max_pose_dist = cfg.yolo.max_dist;
    const std::string codec_name = cfg.yolo_video.codec;
    const int codec = (codec_name == "MP4V")
        ? cv::VideoWriter::fourcc('M', 'P', '4', 'V')
        : cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

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

    cv::VideoWriter writer(out_video, codec, fps, cv::Size(width, height));
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
    // 每块检测一行；没有检测到的帧不写行
    csv << "frame,dets,class,color,conf,pose_ok,x_m,y_m,z_m,reproj_px\n";

    int frames = 0, frames_with = 0, det_total = 0, kept_total = 0, far_dropped = 0;
    std::map<std::string, int> cls_count, color_count;
    const auto t0 = std::chrono::steady_clock::now();

    cv::Mat frame;
    while (cap.read(frame))
    {
        if (frame.empty()) break;

        const std::vector<YoloDetection> dets = detectArmorsYolo(frame, cfg.yolo);

        // 关键点重排后解位姿（网络顺序 LT,LB,RB,RT -> PnP 要的 LB,LT,RT,RB）
        std::vector<PoseResult> poses;
        poses.reserve(dets.size());
        for (const auto& d : dets)
            poses.push_back(solveArmorPose(cornersFromNNKeypoints(d.keypoints), cam,
                                           cfg.armorWidthMm(), cfg.armor.bar_len,
                                           cfg.armor.max_reproj));

        det_total += static_cast<int>(dets.size());

        // 只保留位姿可信、且在距离门限内的检测
        std::vector<YoloDetection> kept_dets;
        std::vector<PoseResult>    kept_poses;
        for (size_t j = 0; j < dets.size(); ++j)
        {
            const PoseResult& p = poses[j];
            if (!p.valid) continue;
            if (max_pose_dist > 0.0f && p.tvec[2] > max_pose_dist) { ++far_dropped; continue; }
            kept_dets.push_back(dets[j]);
            kept_poses.push_back(p);
        }

        kept_total += static_cast<int>(kept_dets.size());
        if (!kept_dets.empty()) ++frames_with;

        for (size_t j = 0; j < kept_dets.size(); ++j)
        {
            const YoloDetection& d = kept_dets[j];
            const PoseResult&    p = kept_poses[j];
            ++cls_count[yoloClassName(d.cls)];
            ++color_count[yoloColorName(d.color)];

            csv << frames << ',' << dets.size() << ',' << yoloClassName(d.cls) << ','
                << yoloColorName(d.color) << ',' << d.score << ',' << (p.valid ? 1 : 0) << ','
                << p.tvec[0] << ',' << p.tvec[1] << ',' << p.tvec[2] << ','
                << p.reproj_err << '\n';
        }

        // 画框 + 号码/距离，写进输出视频（没检测到的帧也照写，保持时间轴完整）
        cv::Mat view = frame.clone();
        drawYoloDetections(view, kept_dets, &kept_poses);
        writer.write(view);

        ++frames;
        if (frames % 100 == 0) std::cout << "  已处理 " << frames << "/" << total << " 帧\n";
    }

    writer.release();
    csv.close();

    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    std::cout << "\n处理完成：\n"
              << "  帧数          : " << frames << "\n"
              << "  有装甲板的帧  : " << frames_with
              << "（" << (frames ? 100.0 * frames_with / frames : 0.0) << "%）\n"
              << "  检测总数      : " << det_total
              << "（距离门限内保留 " << kept_total << "，>"
              << max_pose_dist << "m 丢弃 " << far_dropped << "）\n"
              << "  类别分布      : ";
    for (const auto& kv : cls_count) std::cout << kv.first << "=" << kv.second << " ";
    std::cout << "\n  颜色分布      : ";
    for (const auto& kv : color_count) std::cout << kv.first << "=" << kv.second << " ";
    std::cout << "\n  耗时          : " << sec << " s（"
              << (sec > 0 ? frames / sec : 0.0) << " fps）\n"
              << "  标注视频      : " << out_video << "\n"
              << "  结果 CSV      : " << out_csv << "\n";

    return 0;
}
