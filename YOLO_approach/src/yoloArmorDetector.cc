// 神经网络方案（图片）：装甲板检测网络出 4 个关键点
//                    -> 关键点重排 -> 复用传统方案的 PnP 解位姿
//
// 用法:
//   ./armor_yolo_detector                 # 默认 图1
//   ./armor_yolo_detector <文件夹或单张图片>

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "func.h"          // YOLO 的：makeBlob / loadYoloModel / detectArmorsYolo / drawYoloDetections
#include "PnP_solver.h"    // 复用传统方案的 PnP 模块

// 复用 conventional_approach/include/input_image.cc（CMake 里已加进 target）
std::vector<cv::Mat> input_image (std::string path, std::vector<std::string>* names = nullptr);

int main(int argc, char** argv)
{
    const std::string default_path =
        "/home/lyritique/project/Task2/Project_materials/含真值图像/图1";
    const std::string path = (argc > 1) ? argv[1] : default_path;

    // 模型路径：优先 FP32 版本（OpenCV 4.5.4 的 dnn 读不了 FP16 常量），其次原版
    std::string model_path = "models/Infantry_fp32.onnx";
#ifdef YOLO_MODEL_DIR
    {
        const std::string dir = YOLO_MODEL_DIR;
        if (std::filesystem::exists(dir + "/Infantry_fp32.onnx"))
            model_path = dir + "/Infantry_fp32.onnx";
        else if (std::filesystem::exists(dir + "/Infantry.onnx"))
            model_path = dir + "/Infantry.onnx";
    }
#else
    if (!std::filesystem::exists(model_path)) model_path = "models/Infantry.onnx";
#endif

    std::string err;
    if (!loadYoloModel(model_path, err))
    {
        std::cerr << err << "\n"
                  << "请把装甲板检测模型放到: " << model_path << "\n";
        return 1;
    }

    std::vector<std::string> image_paths;
    std::vector<cv::Mat> images = input_image(path, &image_paths);
    if (images.empty())
    {
        std::cerr << "没有读到图片: " << path << "\n";
        return 1;
    }

    std::filesystem::create_directories("output/debug");

    // 检测结果 CSV：给误差表脚本用（追加写，多次运行不同图片会累积）
    const std::string det_csv_path = "output/debug/detections_yolo.csv";
    const bool csv_is_new = !std::filesystem::exists(det_csv_path) ||
                            std::filesystem::file_size(det_csv_path) == 0;
    std::ofstream det_csv(det_csv_path, std::ios::app);
    if (csv_is_new) det_csv << "image,method,det,label,conf,x_m,y_m,z_m,reproj_px,px,py\n";

    // 静态图用 cameraImages()；视频帧要换成 cameraVideo()
    const CameraParams cam = cameraImages();

    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat& img = images[i];
        const std::vector<YoloDetection> dets = detectArmorsYolo(img);

        // 每块检测都解一次位姿（关键点顺序 LT,LB,RB,RT -> 重排成 LB,LT,RT,RB）
        std::vector<PoseResult> poses;
        poses.reserve(dets.size());
        for (const auto& d : dets)
            poses.push_back(solveArmorPose(cornersFromNNKeypoints(d.keypoints), cam, 135.0f, 55.0f));

        std::cout << "\n第 " << i << " 张（" << img.cols << "x" << img.rows << "）"
                  << " 检测到 " << dets.size() << " 块装甲板\n";

        const std::string image_name = (i < image_paths.size())
            ? std::filesystem::path(image_paths[i]).filename().string()
            : std::to_string(i);

        for (size_t j = 0; j < dets.size(); ++j)
        {
            const YoloDetection& d = dets[j];
            cv::Point2f center(0.0f, 0.0f);
            for (const auto& p : d.keypoints) center += p;
            center *= 0.25f;

            std::cout << "  块#" << j
                      << " 类别=" << yoloClassName(d.cls)
                      << " 颜色=" << yoloColorName(d.color)
                      << " 置信度=" << d.score
                      << " 中心像素=(" << static_cast<int>(center.x) << ","
                      << static_cast<int>(center.y) << ")";
            if (poses[j].valid)
                std::cout << " tvec=(" << poses[j].tvec[0] << ", " << poses[j].tvec[1] << ", "
                          << poses[j].tvec[2] << ")m 重投影=" << poses[j].reproj_err << "px";
            else
                std::cout << " 位姿失败(" << poses[j].reason << ")";
            std::cout << "\n";

            if (poses[j].valid)
                det_csv << image_name << ",yolo," << j << ','
                        << yoloClassName(d.cls) << ',' << d.score << ','
                        << poses[j].tvec[0] << ',' << poses[j].tvec[1] << ','
                        << poses[j].tvec[2] << ',' << poses[j].reproj_err << ','
                        << center.x << ',' << center.y << '\n';
        }

        cv::Mat view = img.clone();
        drawYoloDetections(view, dets, &poses);

        const std::string out = "output/debug/yolo_" + std::to_string(i) + ".png";
        cv::imwrite(out, view);
        std::cout << "  标注图: " << out << "\n";

        try
        {
            cv::imshow("yolo " + std::to_string(i), view);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "  imshow 失败，直接看图：" << out << "\n";
        }
    }

    return 0;
}
