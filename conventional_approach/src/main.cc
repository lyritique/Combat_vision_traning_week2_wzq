# include <chrono>
# include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>
# include "func.h"






int main()
{
    std::vector<cv::Mat> images {input_image("")};
    
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

}