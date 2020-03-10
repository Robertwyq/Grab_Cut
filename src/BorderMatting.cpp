//
// Created by Robert on 2019/6/5.
//
#include "../include/BorderMatting.h"

using namespace cv;
#define PI  (3.14159)
void BorderMatting::Initialize(const Mat& originImage, const Mat& mask, int threshold1, int threshold2) {
    mask.copyTo(this->Mask);
    // 剔除确定的背景元素
    Mask = Mask & 1;
    originImage.copyTo(this->Image);
    // 利用openCV的canny算子,大于threshold2为边缘,将边缘点存入Edge
    Canny(Mask, Edge, threshold1, threshold2);
    // 将边缘点存储
    Store_Edge();
    computeNearestPoint();
    HaveEdge = true;
}
void BorderMatting::computeNearestPoint(){
    for (int i = 0; i < Image.rows; i++) {
        for (int j = 0; j < Image.cols; j++){
            if (Edge.at<uchar>(i, j) == 0 && Mask.at<uchar>(i, j) == 1){
                point p(j, i);
                double min_d = INFINITY;
                int id = 0;
                for (int k = 0; k < contourVector.size(); k++) {
                    double dis = p.Compute_distance(contourVector[k].pointInfo);
                    if(dis < min_d){
                        min_d = dis;
                        id = k;
                    }
                }
                if (min_d > 3) {
                    continue;
                }
                else {
                    p.d = int(min_d);
                    contourVector[id].neighbor.push_back(p);
                }
            }
        }
    }
}
void BorderMatting::Store_Edge(){
    for (int i = 0; i < Image.rows; i++) {
        for (int j = 0; j < Image.cols; j++) {
            if (Edge.at<uchar>(i, j)){
                // emplace 避免产生临时变量,更加高效
                contourVector.emplace_back(point(j, i));
            }
        }
    }
}
void BorderMatting::computeMeanVariance(point p, Information &res){
    const int halfL = 20;
    Vec3b bgd_mean, fgd_mean;
    double bgd_var = 0, fgd_var = 0;
    int fgd_num = 0, bgd_num = 0;
    int x = (p.x - halfL < 0) ? 0 : p.x - halfL;
    int width = (x + 2 * halfL + 1 <= Image.cols) ? halfL * 2 + 1 : Image.cols - x;
    int y = (p.y - halfL < 0) ? 0 : p.y - halfL;
    int height = (y + 2 * halfL + 1 <= Image.rows) ? halfL * 2 + 1 : Image.rows - y;
    Mat neiborPixels = Image(Rect(x,y, width, height));
    for (int i = 0; i < neiborPixels.rows; i++) {
        for (int j = 0; j < neiborPixels.cols; j++) {
            Vec3b pixelColor = neiborPixels.at<Vec3b>(i, j);
            if (Edge.at<uchar>(y + i, x + j) == 1) {
                fgd_mean += pixelColor;
                fgd_num++;
            }
            else {
                bgd_mean += pixelColor;;
                bgd_num++;
            }
        }
    }
    if (fgd_num > 0) {
        fgd_mean = fgd_mean / fgd_num;
    }
    else {
        fgd_mean = 0;
    }
    if (bgd_num > 0) {
        bgd_mean = bgd_mean / bgd_num;
    }
    else {
        bgd_mean = 0;
    }

    for (int i = 0; i < neiborPixels.rows; i++) {
        for (int j = 0; j < neiborPixels.cols; j++) {
            Vec3b pixelColor = neiborPixels.at<Vec3b>(i, j);
            if (Edge.at<uchar>(y + i, x + j) == 1)
                fgd_var += (fgd_mean - pixelColor).dot(fgd_mean - pixelColor);
            else
                bgd_var += (pixelColor - bgd_mean).dot(pixelColor - bgd_mean);
        }
    }
    if (fgd_num >0) {
        fgd_var = fgd_var / fgd_num;
    }
    else {
        fgd_var = 0;
    }
    if (bgd_num > 0) {
        bgd_var = bgd_var / bgd_num;
    }
    else {
        bgd_var = 0;
    }
    res.bgd_mean = bgd_mean;
    res.bgd_var = bgd_var;
    res.fgd_mean = fgd_mean;
    res.fgd_var = fgd_var;
}

double BorderMatting::D_n(point p, uchar z, int delta, int sigma, Information &para) {
    double alpha = Sigmoid(p.d,delta,sigma);
    double Mean = mean(alpha, value_Gray(para.fgd_mean), value_Gray(para.bgd_mean));
    double Var  = var(alpha, para.fgd_var, para.bgd_var);
    double D = -log(Gaussian(z, Mean, Var)) / log(2.0);
    return D;
}
double BorderMatting::Gaussian(double x, double mean, double sigma) {
    double p = 1.0 / (pow(sigma, 0.5)*pow(2.0*PI, 0.5))* exp(-(pow(x - mean, 2.0) / (2.0*sigma)));
    return p;
}
double BorderMatting::mean(double alpha, double Fmean, double Bmean) {
    return (1.0 - alpha)*Bmean + alpha*Fmean;
}

double BorderMatting::var(double alpha, double Fvar, double Bvar) {
    return pow((1.0 - alpha),2)*Bvar + pow(alpha,2)*Fvar;
}
double BorderMatting::V_n(int delta_, int delta, int sigma_, int sigma) {
    double V=lamda1*pow((delta_-delta),2)+lamda2 * pow((sigma - sigma_),2);
    return V;
}
double BorderMatting::Sigmoid(double d, double deltaCenter, double sigma) {
    if (d < deltaCenter - sigma / 2)
        return 0;
    if (d >= deltaCenter + sigma / 2)
        return 1;
    double res = -(d - deltaCenter) / sigma;
    res = 1.0 / (1.0 + exp(res));
    return res;
}
uchar BorderMatting::value_Gray(Vec3b color){
    // RGB图像转化成二值灰度图像,著名心理学公式：Y <- 0.299R + 0.587G + 0.114B
    return (color[2] * 299 + color[1] * 587 + color[0] * 114 ) / 1000;
}


void BorderMatting::Run(){
    clock_t start, end;
    start = clock();

    int delta = MAXDELTA / 2;
    int sigma = MAXSIGMA / 2;

    for (int i = 0; i < contourVector.size(); i++){
        Information info;
        computeMeanVariance(contourVector[i].pointInfo, info);
        contourVector[i].pointInfo.nearbyInformation = info;
        for (int j = 0; j < contourVector[i].neighbor.size(); j++) {
            point &p = contourVector[i].neighbor[j];
            computeMeanVariance(p, info);
            p.nearbyInformation = info;
        }

        double min = INFINITY;
        for (int deltalevel = 0; deltalevel < MAXDELTA; deltalevel++) {
            for (int sigmalevel = 0; sigmalevel < MAXSIGMA; sigmalevel++){
                uchar gray = value_Gray(Image.at<Vec3b>(contourVector[i].pointInfo.y, contourVector[i].pointInfo.x));
                double D = D_n(contourVector[i].pointInfo, gray, deltalevel, sigmalevel, contourVector[i].pointInfo.nearbyInformation);
                for (int j = 0; j < contourVector[i].neighbor.size(); j++){
                    point &p = contourVector[i].neighbor[j];
                    D += D_n(p, value_Gray(Image.at<Vec3b>(p.y, p.x)), deltalevel, sigmalevel, p.nearbyInformation);
                }
                double V = V_n(deltalevel,delta,sigmalevel,sigma);
                if (D + V < min){
                    min = D + V;
                    contourVector[i].pointInfo.delta = deltalevel;
                    contourVector[i].pointInfo.sigma = sigmalevel;
                }
            }
        }
        sigma = contourVector[i].pointInfo.sigma;
        delta = contourVector[i].pointInfo.delta;
        contourVector[i].pointInfo.alpha = Sigmoid(0, delta, sigma);
        for (int j = 0; j < contourVector[i].neighbor.size(); j++){
            point &p = contourVector[i].neighbor[j];
            p.alpha = Sigmoid(p.d, delta, sigma);
        }
    }
    Mat alphaMask = Mat(Mask.size(), CV_32FC1, Scalar(0));
    for (int i = 0; i < Mask.rows; i++) {
        for (int j = 0; j < Mask.cols; j++) {
            alphaMask.at<float>(i, j) = Mask.at<uchar>(i, j);
        }
    }
    for (int i = 0; i < contourVector.size(); i++) {
        alphaMask.at<float>(contourVector[i].pointInfo.y, contourVector[i].pointInfo.x) = contourVector[i].pointInfo.alpha;
        for (int j = 0; j < contourVector[i].neighbor.size(); j++) {
            point &p = contourVector[i].neighbor[j];
            alphaMask.at<float>(p.y, p.x) = p.alpha;
        }
    }
    Mat Result = Mat(Image.size(), CV_8UC4);
    for (int i = 0; i < Result.rows; i++) {
        for (int j = 0; j < Result.cols; j++) {
            if (alphaMask.at<float>(i, j)*255!=0)
                Result.at<Vec4b>(i, j) = Vec4b(Image.at<Vec3b>(i, j)[0], Image.at<Vec3b>(i, j)[1], Image.at<Vec3b>(i, j)[2], alphaMask.at<float>(i, j) * 255);
            else {
                Result.at<Vec4b>(i, j) = Vec4b(0, 0, 0, 0);
            }
        }
    }
    end = clock();
    imshow("Border matting", Result);
    cout<<"Time："<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;
    std::cout << "Border matting done!" << std::endl;
}
void BorderMatting::Show_Edge() {
    if (!HaveEdge) {
        return;
    }
//    imshow("canny",Edge);
}
