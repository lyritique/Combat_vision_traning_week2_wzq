# include <opencv2/core/cvstd.hpp>
# include <opencv2/core/mat.hpp>
# include <opencv2/imgcodecs.hpp>
# include <iostream>

std::vector<cv::Mat> input_image (std::string folder)
{
    std::vector <cv::String> files;
    cv::glob(folder + "/*.jpg",files ,false);

    std::vector <cv::Mat> images;
    for (const cv::String& file :files)
    {
        cv::Mat img = cv::imread(file);
        if (img.empty())
        {
            std::cerr << "读取失败" << file <<"\n";
            continue;
        }
        images.push_back(img);
    }
    return images;
}