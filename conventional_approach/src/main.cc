// 图片流程：读图 -> 预处理 -> 灯条 -> 配对 -> 数字识别 -> 位姿解算 -> 剔除 -> 画图
//
// 用法:
//   ./armor_conventional_detector                          # 默认 图1 / 蓝灯条 / 阈值 25
//   ./armor_conventional_detector <文件夹> <阈值> <blue|red>
//   例: ./armor_conventional_detector ".../图3" 45 blue

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "func.h"
#include "PnP_solver.h"


int main(int argc, char** argv)
{
    // ---------- 参数（config/default.yaml，可用命令行覆盖） ----------
    const Config cfg = Config::load(findConfigPath(argc, argv));
    const std::vector<std::string> pos = positionalArgs(argc, argv);

    const std::string folder = pos.size() > 0 ? pos[0] : cfg.conv_image.input;
    const int thresh = pos.size() > 1 ? std::stoi(pos[1]) : cfg.conv_image.thresh;
    const std::string color_name = pos.size() > 2 ? pos[2] : cfg.conv_image.color;
    const LightColor color = parseColor(color_name);

    const std::string out_dir = cfg.conv_image.out_dir;
    const CameraParams cam = cfg.camera(false);           // 静态图那套内参
    const bool use_digit_filter = cfg.digit.filter_image;

    std::filesystem::create_directories(out_dir);

    // ---------- 读图 ----------
    std::vector<std::string> image_paths;
    std::vector<cv::Mat> images = input_image(folder, &image_paths);
    if (images.empty())
    {
        std::cerr << "没有从 " << folder << " 读到 jpg\n";
        return 1;
    }
    std::cout << "读到 " << images.size() << " 张图（" << folder << "）\n";

    // 检测结果 CSV：给误差表脚本用（每块保留的装甲板一行）
    // 追加写：多次运行不同图片会累积；想清空就删掉这个文件
    const std::string det_csv_path = cfg.conv_image.det_csv;
    const bool csv_is_new = !std::filesystem::exists(det_csv_path) ||
                            std::filesystem::file_size(det_csv_path) == 0;
    std::ofstream det_csv(det_csv_path, std::ios::app);
    if (csv_is_new) det_csv << "image,method,det,label,conf,x_m,y_m,z_m,reproj_px,px,py\n";

    // ---------- 逐张处理 ----------
    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat& img = images[i];                        // extractNumber 要非 const 引用

        cv::Mat mask = preprocess(img, color, thresh, cfg.light);   // 1) 二值图
        std::vector<LightBar> bars = findLights(mask, cfg.light);   // 2) 灯条
        std::vector<Armor> armors = matchBar(bars, cfg.pair);       // 3) 配对（内部已填 corners）

        // 4) 每块候选：数字识别 + 位姿解算
        std::vector<ArmorDetection> detections;
        detections.reserve(armors.size());
        for (auto& a : armors)
        {
            ArmorDetection d;
            d.armor = a;
            d.digit = classify(extractNumber(img, d.armor, cfg.digit, false), cfg.digit);
            d.pose  = positionSolve(d.armor, cam, cfg.armor);
            detections.push_back(d);
        }

        std::cout << "\n第 " << i << " 张（" << img.cols << "x" << img.rows << "）"
                  << " 灯条=" << bars.size() << " 候选=" << detections.size() << "\n";
        for (size_t j = 0; j < detections.size(); ++j)
        {
            const auto& d = detections[j];
            std::cout << "  块#" << j
                      << " 长=" << static_cast<int>(d.armor.leftBar->length)
                      << "/" << static_cast<int>(d.armor.rightBar->length);
            if (d.digit.valid) std::cout << " 数字=" << d.digit.label << "(" << d.digit.conf << ")";
            else               std::cout << " 数字=失败";
            if (d.pose.valid)  std::cout << " tvec=(" << d.pose.tvec[0] << ", " << d.pose.tvec[1]
                                         << ", " << d.pose.tvec[2] << ")m 重投影=" << d.pose.reproj_err << "px";
            else               std::cout << " 位姿=失败(" << d.pose.reason << ")";
            std::cout << "\n";
        }

        // 5) 剔除 negative 、 识别失败 、 解算不可信 、 重复
        const size_t before = detections.size();
        eraseIgnoreClasses(detections, cfg.digit, cfg.pair, use_digit_filter);
        std::cout << "  剔除后剩 " << detections.size() << "/" << before << " 块\n";

        // 写检测结果（中心像素 = 四个角点的均值）
        const std::string image_name = (i < image_paths.size())
            ? std::filesystem::path(image_paths[i]).filename().string()
            : std::to_string(i);
        for (size_t j = 0; j < detections.size(); ++j)
        {
            const ArmorDetection& d = detections[j];
            cv::Point2f c(0.0f, 0.0f);
            for (const auto& p : d.armor.corners) c += p;
            c *= 0.25f;
            //写csv
            det_csv << image_name << ",traditional," << j << ','
                    << d.digit.label << ',' << d.digit.conf << ','
                    << d.pose.tvec[0] << ',' << d.pose.tvec[1] << ',' << d.pose.tvec[2] << ','
                    << d.pose.reproj_err << ',' << c.x << ',' << c.y << '\n';
        }

        // 6) 画图存盘
        cv::Mat view = img.clone();
        drawResult(view, bars, detections);
        const std::string path = out_dir + "/result_" + std::to_string(i) + ".png";
        cv::imwrite(path, view);
        cv::imwrite(out_dir + "/mask_" + std::to_string(i) + ".png", mask);
        std::cout << "  标注图: " << path << "\n";

        // 7) 有图形界面就显示（容器/SSH 里 imshow 会抛异常，跳过即可）
        try
        {
            cv::imshow("result " + std::to_string(i), view);
            //cv::imshow("mask " + std::to_string(i), mask);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "  imshow 失败，直接看图：" << path << "\n";
        }
    }
    det_csv.close();
    return 0;
}
