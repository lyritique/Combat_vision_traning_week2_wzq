# include <opencv2/opencv.hpp>
# include <opencv2/core/saturate.hpp>
# include "func.h"  








/**
 * @brief 预处理：把 BGR 原图变成"候选灯条"的二值图（白=候选，黑=背景）
 *
 * 灯条是强光源、颜色单一：真实视频里偏红（R 明显大于 B），仿真图 图1/图2/图3
 * 里偏蓝（B 明显大于 R）。所以先用通道差分挑出"目标颜色多、相反颜色少"的像素，
 * 再阈值二值化，最后用形态学清理。颜色判据保留成参数，两种都能跑。
 *
 * @param row_img     输入的 BGR 三通道图（不会被修改）
 * @param color       要找哪种颜色的灯条：LightColor::RED / LightColor::BLUE
 * @param diff_thresh 差分阈值，超过它才算候选。实测参考：
 *                    视频（红灯条）50~60；图3（蓝灯条）40~60；图1（蓝灯条偏暗）20~30
 * @return CV_8UC1 二值图，255 = 可能是灯条，0 = 背景
 */
cv::Mat  preprocess(const cv::Mat& row_img, LightColor color, int diff_thresh)
{
    // ---- 1. 颜色差分：挑出"目标颜色多、相反颜色少"的像素 ----
    // OpenCV 里图像是 BGR 顺序：ch[0]=B, ch[1]=G, ch[2]=R
    std::vector<cv::Mat> ch;
    cv::split(row_img, ch);

    // subtract 自带饱和处理（结果夹在 0~255），不会像 uchar 直接相减那样
    // 把负数变成 206 这种回绕值
    cv::Mat diff;
    if (color == LightColor::RED)
        cv::subtract(ch[2], ch[0], diff);   // R - B：红灯条数值大，灰/蓝背景接近 0
    else
        cv::subtract(ch[0], ch[2], diff);   // B - R：蓝灯条数值大

    // ---- 2. 二值化：差分值超过阈值才算候选 ----
    cv::Mat mask;
    cv::threshold(diff, mask, diff_thresh, 255, cv::THRESH_BINARY);

    // ---- 3. 形态学：先开运算去掉小噪点，再闭运算补上被阈值切断的灯条 ----
    // 3x3 的核比灯条宽度（十几像素）小得多，不会把灯条整条抹掉
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    return mask;
}

/* ------- 你原来手写逐像素的版本（保留在这里备查，不参与编译） -------
 * 里面有四处坑，已经在下面对应位置改好：
 *   1. row_ptr_resultImg 必须取 result_img 的行指针；原先写成 row_img，
 *      结果一直在往输入图上写，result_img 从没被赋值（实测返回全黑图）；
 *   2. 单通道结果图的下标必须是 [x]，不能是 [x * row_img.channels()]，
 *      否则每行会越界写 3 倍长度，堆被写坏；
 *   3. 是 saturate_cast，不是 staurate_cast（拼写）；
 *   4. 原版返回的是差分灰度图，没有做阈值；findContours 需要二值图。
 */
/*
cv::Mat preprocess(const cv::Mat& row_img)
{
    cv::Mat result_img (row_img.rows, row_img.cols, CV_8UC1);
    for ( size_t y=0; y<row_img.rows; y++ )
    {
        for ( size_t x=0; x<row_img.cols; x++ )
        {
            unsigned char* row_ptr_rowImg = row_img.ptr<unsigned char> ( y );
            unsigned char* data_ptr_rowImg = &row_ptr_rowImg[ x*row_img.channels() ];
            unsigned char* row_ptr_resultImg = result_img.ptr<unsigned char> ( y );
            unsigned char* data_ptr_resultImg = &row_ptr_resultImg[x];
            *data_ptr_resultImg = cv::saturate_cast<uchar>( data_ptr_rowImg[2] - data_ptr_rowImg[0] );
        }
    }
    return result_img;
}
*/
/*
cv::Mat preprocess(const cv::Mat& row_img)
{










}












*/