//
// Created by Robert on 2019/6/5.
//
#ifndef GRAB_CUT_BORDERMATTING_H
#define GRAB_CUT_BORDERMATTING_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
/**
 * information about the bgd & fgd mean and var
 */
struct Information {
    Vec3b bgd_mean;
    Vec3b fgd_mean;
    double bgd_var;
    double fgd_var;
};
/**
 * 轮廓上的点
 */
class  point {
public:
    point()= default;
    point(int x, int y, int d = 0){
        this->x = x;
        this->y = y;
        this->d = d;
    }
    double Compute_distance(point t){
        return sqrt(pow((this->x - t.x),2)+ pow((this->y - t.y),2));
    }
public:
    int x;
    int y;
    int delta, sigma;
    int d;
    double alpha;
    Information nearbyInformation;
};
/**
 * 提取轮廓上的点
 */
class Outline {
public:
    vector<point> neighbor;
    point pointInfo;
    Outline(point p) { this->pointInfo = p;};
};
/**
 * 边界优化类
 */
class BorderMatting {
public:
    BorderMatting() = default;
    ~BorderMatting() = default;
    void Initialize(const Mat& _originImage, const Mat& _mask, int threshold_1 = 1, int threshold_2 = 4);
    void computeMeanVariance(point p, Information &result);
    void computeNearestPoint();
    void Run();
    void Store_Edge();
    void Show_Edge();
    double Gaussian(double x, double delta, double sigma);
    double mean(double x, double Fmean, double Bmean);
    double var(double x, double Fvar, double Bvar);
    double Sigmoid(double _r, double _delta, double _sigma);
    double D_n(point _p, uchar _z, int _delta, int _sigma, Information &result);
    double V_n(int deltalevel,int delta,int sigmalevel,int sigma);
    uchar value_Gray(Vec3b color);
private:
    // contour point
    vector<Outline> contourVector;
    // mask
    Mat Mask;
    // Input image
    Mat Image;
    // edge point
    Mat Edge;
    // basic para
    double lamda1 = 50;
    double lamda2 = 1000;
    int MAXDELTA = 30;
    int MAXSIGMA = 10;
    bool HaveEdge = false;

};
#endif //GRAB_CUT_BORDERMATTING_H
