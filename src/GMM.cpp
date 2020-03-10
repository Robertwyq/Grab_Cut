//
// Created by Robert on 2019/6/5.
//
#include "../include/GMM.h"
#include "../include/KmeansSample.h"
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;
const double Pi = 3.14159;
Component::Component() {
    mean = cv::Vec3d::all(0);
    cov = cv::Matx33d::zeros();
    Totol_Pixels = 0;
}
Component::Component(cv::Mat &a) {
    mean = a(cv::Rect(0,0,3,1));
    cov = a(cv::Rect(3,0,9,1)).reshape(1,3);
}
int Component::Get_Num() const {
    return this->Totol_Pixels;
}
cv::Mat Component::Export() const {
    cv::Mat meanMat = cv::Mat(mean.t());
    cv::Mat covMat = cv::Mat(cov).reshape(1,1);
    cv::Mat result;
    cv::hconcat(meanMat,covMat,result);
    return result; // 1*12 的行向量
}
void Component::Init_learning() {
    mean = cv::Vec3d::all(0);
    cov = cv::Matx33d::zeros();
    Totol_Pixels = 0;
}
void Component::Add_Pixel(cv::Vec3d color) {
    mean+= color;
    cov+=color*color.t();
    Totol_Pixels++;
}
void Component::Training() {
    const double variance =0.01;
    mean = mean / Totol_Pixels;
    cov = (1.0/Totol_Pixels)*cov;
    cov = cov - mean*mean.t();
    const double det = cv::determinant(cov);
    if(det<std::numeric_limits<double>::epsilon()){
        cov+=variance*cv::Matx33d::eye();
    }
}
// 计算概率密度函数
double Component::operator()(const cv::Vec3d &color) const {
    cv::Vec3d Diff = color - mean;
    double p = 1.0/(pow(2*Pi,3/2)*sqrt(cv::determinant(cov)))*
            exp(-0.5*(Diff.t()*cov.inv()*Diff)(0));
    return p;
}




GMM::GMM(int _Dim_NUM, int _Obj_NUM):Pi_k(_Obj_NUM),components(_Obj_NUM){
    Dim_NUM = _Dim_NUM;
    Obj_NUM = _Obj_NUM;
    Total_Samples = 0;

}

void GMM::Init_Learning() {
    for(int i=0;i<Obj_NUM;i++){
        components[i].Init_learning();
        Pi_k[i]=0;
    }
    Total_Samples=0;
}
void GMM::Training() {
    for(int i=0;i<Obj_NUM;i++){
        int num = components[i].Get_Num();
        if(num==0){
            Pi_k[i]=0;
        }
        else{
            Pi_k[i]=(double)num/Total_Samples;
            components[i].Training();
        }
    }
}
void GMM::Add_Pixel(int ComID, cv::Vec3d color) {
    components[ComID].Add_Pixel(color);
    Total_Samples++;
}
int GMM::Get_K() {
    return Obj_NUM;
}
int GMM::Get_MostPossible_ID(const cv::Vec3d color) const {
    int ID = 0;
    double Max_p = 0;
    for(int i=0;i<Obj_NUM;i++){
        if(components[i](color)>=Max_p){
            ID = i;
            Max_p = components[i](color);
        }
    }
    return ID;
}
// 计算一个像素属于这个GMM混合高斯模型的概率
// 相当于把这个像素属于K个高斯模型的概率与对应的权值相乘后再相加
double GMM::operator()(const cv::Vec3d color) const {
    double p = 0;
    for(int i =0;i<Obj_NUM;i++){
        if(Pi_k[i]>0){
            p+=Pi_k[i]*components[i](color);
        }
    }
    return p;
}
GMM GMM::MattoGMM(const cv::Mat &model) {
    const int paraNumOfComponent = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if((model.type()!=CV_64FC1)||(model.rows!=1)||model.cols%paraNumOfComponent!=0){
        cout<<"ERROR: check the type and the rows==1 and cols==13"<<endl;
    }
    int Num_Component = model.cols/paraNumOfComponent;
    GMM result(3,Num_Component);
    for(int i = 0;i<Num_Component;i++){
        cv::Mat componentModel = model(cv::Rect(13*i, 0, paraNumOfComponent, 1));
        // 取其中的第一个参数值为第i个高斯模型的概率
        result.Pi_k[i] = componentModel.at<double>(0, 0);
        cv::Mat para;
        para=componentModel(cv::Rect(1, 0, 12, 1));
        //  取其中的平均值和协方差12个元素
        result.components[i] = Component(para);
    }
    return result;
}
cv::Mat GMM::GMMtoMat() const {
    cv::Mat result;
    for(int i=0;i<Obj_NUM;i++){
        cv::Mat Pi_kMat(1, 1, CV_64F, cv::Scalar(Pi_k[i]));
        cv::Mat ComponentMat = components[i].Export();
        cv::Mat Combined;
        cv::hconcat(Pi_kMat, ComponentMat, Combined);
        if (result.empty()) {
            result = Combined; // 13*1
        }
        // 一直合并下去，生成最终的一列行向量
        else {
            cv::hconcat(result, Combined, result);
        }
    }
    return result;
}