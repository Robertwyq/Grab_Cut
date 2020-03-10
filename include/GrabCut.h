//
// Created by Robert on 2019/6/5.
//
#ifndef GRAB_CUT_GRABCUT_H
#define GRAB_CUT_GRABCUT_H

#include <cv.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "GMM.h"
#include "graph.h"
#include "block.h"
using namespace cv;
enum {
    GC_WITH_RECT  = 0,
    GC_WITH_MASK  = 1,
    GC_CUT        = 2
};
class GrabCut2D{
public:
    /**
     *
     * @param _img: input color image
     * @param _mask: output result of the segmentation
     * @param rect: the rectangle frame draw by the user
     * @param _bgdModel: background model: GMM(13*n)
     * @param _fgdModel: foreground model: GMM(13*n)
     * @param iterCount: iteration of every segmentation
     * @param mode: decide the mode for GrabCut
     */
    void GrabCut( const cv::Mat &_img, cv::Mat &_mask, cv::Rect rect,
                  cv::Mat &_bgdModel,cv::Mat &_fgdModel,
                  int iterCount, int mode );

    ~GrabCut2D() = default;

private:
    void InitMask(Mat & Mask, Size size, Rect rect);
    void InitGMM(const Mat & img, const Mat & mask, GMM &fgdGMM, GMM &bgdGMM);
    void Assign_GMMID(const cv::Mat &img, const cv::Mat &Mask, const GMM &bgdGMM,  const GMM &fgdGMM, cv::Mat &compIdxs);
    void LearnGMMs( const cv::Mat& img, const cv::Mat& mask, const cv::Mat& compIdxs,
                    GMM& bgdGMM, GMM& fgdGMM);
    double ComputeBeta(const cv::Mat &img);
    void ComputeNWeights(const cv::Mat &img, cv::Mat &leftW,cv::Mat &upleftW,cv::Mat &upW,cv::Mat &uprightW,double beta,double gamma);
    void Construct_Graph(const cv::Mat &img,const cv::Mat &Mask, const GMM &bgdGMM,  const GMM &fgdGMM,double lamda,
            const cv::Mat &leftW,const cv::Mat &upleftW,const cv::Mat &upW,const cv::Mat &uprightW, Graph<double,double,double>&graph );
    void Estimate_Segmentation(Graph<double,double,double>&graph, cv::Mat & mask);
};

/**参考伪代码流程：
 * 1：load Input Image 加载输入颜色图像
 * 2：init Mask 用矩形框初始化Mask的Label值（确定背景：0，确定前景：1，可能背景：2，可能前景：3）矩形框以外是0，以内是3
 * 3：init GMM 定义并初始化GMM（GMM完成会加分）
 * 4：Sample Points 前背景颜色采样并进行聚类（建议使用kmeans）
 * 5：Learn GMM（根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
 * 6：Construct Graph（计算数据项t-weight和平滑想n-weight）
 * 7：Estimate Segmentation（调用maxflow库进行分割）
 * 8：Save Result 输出结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
 */

#endif //GRAB_CUT_GRABCUT_H
