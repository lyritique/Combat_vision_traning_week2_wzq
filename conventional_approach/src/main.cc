//# include <chrono>
# include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>   // imshow / waitKey
#include <opencv2/imgproc.hpp>   // line / circle（画灯条框）
#include <vector>
#include <string>
#include <filesystem>   // 用来自动创建输出目录
# include "func.h"






int main()
{
    /*std::vector<cv::Mat> images {input_image("")};
    
    for (int i {}; i < images.size(); i++)
    {
        cv::Mat processed_img {preprocess(&images[i]),};
        ;
    }




    findLights();
    matchLights();
    extractNumber();
    classify();
    correctCorners();
    eraseIgnoreClasses();
    positionSolve();
    */

    // ================= 图片：读入 -> 预处理 -> 存盘 -> imshow 显示 =================

    // 素材文件夹：图1/图2/图3 里的装甲板是蓝灯条；换成视频帧时要用红色判据
    const std::string folder  = "/home/lyritique/project/Task2/Project_materials/含真值图像/图1";
    const std::string out_dir = "output/debug";   // 中间结果目录，相对当前运行目录

    const LightColor color  = LightColor::BLUE;   // 图1/2/3 用 BLUE；视频用 RED
    const int        thresh = 25;                 // 实测参考：图1~25、图3~45、视频~60
    const std::string model_path = "models/mlp.onnx";   // 数字识别网络（相对运行目录）

    std::filesystem::create_directories(out_dir); // 第一次运行时把目录建出来
    std::cout << "中间结果目录: " << std::filesystem::absolute(out_dir) << "\n";

    // 1) 读入：文件夹里所有 jpg，逐张 cv::Mat
    std::vector<cv::Mat> images = input_image(folder);
    if (images.empty())
    {
        std::cerr << "没有从 " << folder << " 读到任何 jpg，程序结束\n";
        return 1;                       // 没有输入就不要再往下走
    }
    std::cout << "读到 " << images.size() << " 张图\n";

    // 2) 逐张：预处理 -> 找灯条 -> 画框存盘 -> imshow 显示
    for (size_t i = 0; i < images.size(); ++i)
    {
        // a) 预处理：原图 -> 候选灯条二值图
        cv::Mat mask = preprocess(images[i], color, thresh);
        cv::imwrite(out_dir + "/mask_" + std::to_string(i) + ".png", mask);

        // b) 找灯条：二值图 -> 候选灯条列表（长宽比 + 竖直度 两道筛）
        const std::vector<LightBar> bars = findLights(mask);
        std::cout << "第 " << i << " 张（" << images[i].cols << "x" << images[i].rows
                  << "）候选灯条 " << bars.size() << " 根\n";
        for (size_t j = 0; j < bars.size(); ++j)
        {
            std::cout << "    #" << j
                      << " 中心=(" << (int)bars[j].center.x << "," << (int)bars[j].center.y << ")"
                      << " 长x宽=" << (int)bars[j].length << "x" << (int)bars[j].width
                      << " 角=" << bars[j].angle
                      << " 面积=" << (int)bars[j].area
                      << " 填充率=" << bars[j].fill << "\n";
        }

        // c) 两两配对：灯条列表 -> 装甲板候选
        const std::vector<ArmorCandidate> armors = matchLights(bars);
        std::cout << "  装甲板候选 " << armors.size() << " 块\n";
        for (size_t j = 0; j < armors.size(); ++j)
        {
            const ArmorCandidate& a = armors[j];
            std::cout << "    #" << j << " 灯条[" << a.left << "," << a.right << "]"
                      << " 中心=(" << (int)a.center.x << "," << (int)a.center.y << ")"
                      << " 中心距/灯条长=" << a.ratio
                      << " 长度比=" << a.length_ratio
                      << " 倾角差=" << a.angle_diff
                      << " 垂直偏差=" << a.perp_dev
                      << " 代价=" << a.cost
                      << (a.large ? " [偏大装甲板]" : "") << "\n";
        }

        // d) 数字识别 + 位置解算，逐块打印中间量
        const CameraParams cam = cameraImages();   // 处理视频时换成 cameraVideo()
        std::vector<ArmorDetection> detections;
        detections.reserve(armors.size());
        for (size_t j = 0; j < armors.size(); ++j)
        {
            const ArmorCandidate& a = armors[j];

            // 小装甲板 135mm、大装甲板 225mm；large 只是暂定标记，最终要靠号码确认
            ArmorDetection d;
            d.armor = a;
            d.pose  = positionSolve(a, cam, a.large ? 225.0f : 135.0f, 55.0f);
            d.digit = classify(extractNumber(images[i], a, a.large), model_path);

            std::cout << "    块 #" << j << " 灯条[" << a.left << "," << a.right << "]"
                      << " 中心距/灯条长=" << a.ratio
                      << " 长度比=" << a.length_ratio
                      << " 倾角差=" << a.angle_diff
                      << " 垂直偏差=" << a.perp_dev
                      << " 代价=" << a.cost << "\n";
            if (d.pose.valid)
                std::cout << "      位姿: tvec=(" << d.pose.tvec[0] << ", " << d.pose.tvec[1]
                          << ", " << d.pose.tvec[2] << ") m  重投影误差=" << d.pose.reproj_err << " px\n";
            else
                std::cout << "      位姿: 解算失败（" << d.pose.reason << "）\n";
            if (d.digit.valid)
                std::cout << "      数字: " << d.digit.label << "  置信度=" << d.digit.conf << "\n";
            else
                std::cout << "      数字: 识别失败（" << d.digit.reason << "）\n";

            detections.push_back(d);
        }

        // e) 剔除 negative / 数字识别失败 / 解算不可信的候选
        const size_t before = detections.size();
        eraseIgnoreClasses(detections);
        std::cout << "  剔除后剩 " << detections.size() << " / " << before << " 块\n";

        // f) 画图存盘 + imshow（绿色细框=灯条，青色粗框=装甲板，红点=中心，文字=距离+号码）
        FrameResult frame_result;
        frame_result.bars   = bars;
        frame_result.armors = detections;
        cv::Mat view = images[i].clone();
        drawResult(view, frame_result);
        const std::string lpath = out_dir + "/lights_" + std::to_string(i) + ".png";
        cv::imwrite(lpath, view);
        std::cout << "  " << lpath << "\n";

        // 显示：左=原图+检测结果，右=二值图，按任意键看下一张
        try
        {
            const std::string win1 = "lights " + std::to_string(i) + "  (press any key)";
            const std::string win2 = "mask "   + std::to_string(i);
            cv::imshow(win1, view);
            cv::imshow(win2, mask);
            cv::waitKey(0);
            cv::destroyWindow(win1);
            cv::destroyWindow(win2);
        }
        catch (const cv::Exception& e)
        {
            // 没有图形界面（例如纯 SSH / 容器）时不要崩，图已经存盘了
            std::cerr << "  imshow 失败，改成看存盘的 png：" << e.what() << "\n";
        }
    }

    return 0;
}
