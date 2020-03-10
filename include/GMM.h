//
// Created by Robert on 2019/6/5.
//

#ifndef GRAB_CUT_GMM_H
#define GRAB_CUT_GMM_H

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#define max(a,b) ((a)<(b)?(b):(a))
#define min(a,b) ((a)>(b)?(b):(a))
class Component{
public:
    Component();
    explicit Component(cv::Mat &Component);
    void Init_learning();
    void Add_Pixel(cv::Vec3d color);
    void Training();
    int Get_Num() const;
    cv::Mat Export() const;
    double operator()(const cv::Vec3d & color) const;

private:
    // 均值
    cv::Vec3d mean;
    // 协方差
    cv::Matx33d cov;
    // 总共像素个数
    int Totol_Pixels;
};


class GMM{
public:
    GMM( int _Dim_NUM = 3, int _Obj_NUM = 5);
    void Init_Learning();
    void Training();
    void Add_Pixel(int ComID, cv::Vec3d color);
    int Get_K();
    int Get_MostPossible_ID(const cv::Vec3d color) const;
    double operator()(const cv::Vec3d color) const;
    static GMM MattoGMM(const cv::Mat &model);
    cv::Mat GMMtoMat() const;
private:
    int Dim_NUM;
    int Obj_NUM;
    int Total_Samples;
    std::vector<Component> components;
    std::vector<double> Pi_k;
};


#endif //GRAB_CUT_GMM_H
