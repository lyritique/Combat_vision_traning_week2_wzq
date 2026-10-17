#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

// 判断是不是图片文件（按扩展名，不区分大小写）
static bool isImageFile(const std::string& path)
{
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;

    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

// 输入既可以是一个文件夹（读里面所有 jpg/png），也可以是一个具体文件。
// 返回读成功的图片；读失败的打印原因并跳过。
// names 不为空时，把每张图的完整路径按同样顺序填进去（做误差表要按文件名对真值）。
std::vector<cv::Mat> input_image (std::string path, std::vector<std::string>* names)
{
    std::vector<std::string> files;

    if (isImageFile(path))
    {
        // 单个文件：只读这一张
        files.push_back(path);
    }
    else
    {
        // 文件夹：读里面所有 jpg 和 png，按文件名排序（顺序稳定，便于对真值）
        for (const char* ext : {"*.jpg", "*.png"})
        {
            std::vector<cv::String> matched;
            cv::glob(path + "/" + ext, matched, false);
            for (const auto& f : matched) files.push_back(f);
        }
        std::sort(files.begin(), files.end());
    }

    std::vector<cv::Mat> images;
    for (const auto& file : files)
    {
        cv::Mat img = cv::imread(file);
        if (img.empty())
        {
            std::cerr << "读取失败: " << file << "\n";
            continue;
        }
        images.push_back(img);
        if (names) names->push_back(file);
    }
    return images;
}
