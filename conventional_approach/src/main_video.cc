// 视频批处理：逐帧跑传统视觉方案（识别 + 数字 + 位置解算），输出标注视频和结果 CSV
//
// 用法:
//   ./armor_video                        # 用下面的默认路径
//   ./armor_video <视频> <输出视频> <输出CSV>
//
// 默认约定：
//   视频 = 作业材料里的展厅实拍（1440×1080、MJPG、531 帧、30fps）
//   内参 = cameraVideo()（带 5 个畸变系数的那套，和静态图不是一套）
//   颜色 = RED（视频里的装甲板是红灯条）

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include "func.h"

int main(int argc, char** argv)
{
    const std::string video_path = (argc > 1) ? argv[1]
        : "/home/lyritique/project/Task2/Project_materials/相机内参与视频/Video_1791696012009_214.avi";
    const std::string out_video  = (argc > 2) ? argv[2] : "output/traditional_video.avi";
    const std::string out_csv    = (argc > 3) ? argv[3] : "output/traditional_video.csv";
    const std::string model_path = "models/mlp.onnx";

    const CameraParams cam  = cameraVideo();      // 视频专用内参（1440×1080 + 畸变）
    const LightColor   color = LightColor::RED;   // 视频里的灯条是红色
    const int          thresh = 60;               // 实测：视频红灯条用 50~60

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "打不开视频: " << video_path << "\n";
        return 1;
    }

    const int    total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const double fps   = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    const int    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int    height= static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "视频: " << width << "x" << height << " " << total << " 帧 " << fps << "fps\n";

    // 输出视频：MJPG 编码写 avi，兼容性最好（mp4v 在这个 OpenCV 构建里不一定能编）
    cv::VideoWriter writer(out_video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           fps, cv::Size(width, height));
    if (!writer.isOpened())
    {
        std::cerr << "打不开输出视频: " << out_video << "（换编码或换路径试试）\n";
        return 1;
    }

    std::ofstream csv(out_csv);
    if (!csv)
    {
        std::cerr << "打不开输出 CSV: " << out_csv << "\n";
        return 1;
    }
    // 每一块装甲板一行；没有检测到的帧不写行，靠 frame 列体现
    csv << "frame,bars,armors,number,conf,dist_m,x,y,z,reproj_px\n";

    int frames = 0, frames_with_armor = 0, armors_total = 0, digits_ok = 0;
    const auto t0 = std::chrono::steady_clock::now();

    cv::Mat frame;
    while (cap.read(frame))
    {
        // 单帧流水线：预处理 -> 灯条 -> 配对 -> 数字识别 -> 位置解算
        FrameResult result = processFrame(frame, cam, color, thresh, model_path);

        // 剔除 negative / 数字失败 / 解算不可信的候选
        eraseIgnoreClasses(result.armors);

        if (!result.armors.empty())
        {
            ++frames_with_armor;
            armors_total += static_cast<int>(result.armors.size());
        }

        for (const auto& d : result.armors)
        {
            if (d.digit.valid && d.digit.label != "negative") ++digits_ok;
            csv << frames << ',' << result.bars.size() << ',' << result.armors.size() << ','
                << d.digit.label << ',' << d.digit.conf << ','
                << d.pose.tvec[2] << ',' << d.pose.tvec[0] << ',' << d.pose.tvec[1] << ','
                << d.pose.tvec[2] << ',' << d.pose.reproj_err << '\n';
        }

        // 画框 + 号码 + 距离，再写进输出视频（没检测到的帧也照写，保持时间轴完整）
        cv::Mat view = frame.clone();
        drawResult(view, result);
        writer.write(view);

        ++frames;
        if (frames % 100 == 0) std::cout << "  已处理 " << frames << "/" << total << " 帧\n";
    }

    writer.release();
    csv.close();

    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n处理完成：\n"
              << "  帧数        : " << frames << "\n"
              << "  有装甲板的帧: " << frames_with_armor
              << "（" << (frames ? 100.0 * frames_with_armor / frames : 0.0) << "%）\n"
              << "  识别出号码  : " << digits_ok << " 次\n"
              << "  耗时        : " << sec << " s（" << (frames ? frames / sec : 0.0) << " fps）\n"
              << "  标注视频    : " << out_video << "\n"
              << "  结果 CSV    : " << out_csv << "\n";
    return 0;
}
