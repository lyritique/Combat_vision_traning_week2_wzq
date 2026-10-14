# ifndef _FUNC_ 
# define _FUNC_
# include <opencv2/core/cvstd.hpp>
# include <opencv2/core/mat.hpp>
# include <opencv2/imgcodecs.hpp>
# include <iostream>
# include <string>
# include <vector>

// 灯条颜色：真实视频里是红色灯条，仿真图（图1/图2/图3）里是蓝色灯条
enum class LightColor { RED, BLUE };

std::vector<cv::Mat> input_image (std::string );
cv::Mat  preprocess(const cv::Mat& row_img, LightColor color = LightColor::RED, int diff_thresh = 60);
void findLights();
void matchLights();
void extractNumber();
void classify();
void correctCorners();
void eraseIgnoreClasses();
void positionSolve();
# endif
