# ifndef _FUNC_ 
# define _FUNC_
# include <opencv2/core/cvstd.hpp>
# include <opencv2/core/mat.hpp>
# include <opencv2/core/types.hpp>
# include <opencv2/imgcodecs.hpp>
# include <iostream>
# include <string>
# include <vector>

// 灯条颜色：真实视频里是红色灯条，仿真图（图1/图2/图3）里是蓝色灯条
enum class LightColor { RED, BLUE };

struct LightBar 
{
    cv::RotatedRect frame;
    float width {};
    float length {};
    float angle {};
};

struct Armor
{
    const LightBar* leftBar;
    const LightBar* rightBar;
    std::vector<cv::Point2f> corners;
};

struct DigitResult
{
    int cls = -1;            // 类别下标 0..8
    float conf = 0;          // softmax 之后的置信度
    std::string label;       // "1".."5" / "outpost" / "guard" / "base" / "negative"
    bool valid = false;      // 是否拿到结果
    std::string reason;      // 失败原因（便于调试和统计）
};

bool isArmorPair(const LightBar& left, const LightBar& right);



std::vector<cv::Mat> input_image (std::string );
cv::Mat  preprocess(const cv::Mat& row_img, LightColor color = LightColor::RED, int diff_thresh = 60);
std::vector<LightBar> findLights(const cv::Mat& binary_img);
std::vector<Armor> matchBar (std::vector<LightBar>& bars);
cv::Mat extractNumber (cv::Mat& bgr ,Armor& armor,bool large_armor);
DigitResult classify(const cv::Mat& input, const std::string& model_path);
void eraseIgnoreClasses();
void positionSolve();
# endif
