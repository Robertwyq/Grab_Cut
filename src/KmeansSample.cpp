//
// Created by Robert on 2019/6/5.
//
#include "../include/KmeansSample.h"
#include <iostream>
#include <cmath>

using namespace std;

Kmeans::Kmeans(int Dim_NUM, int K) {
    dim = Dim_NUM;
    k = K;
    means = new double*[k];
    for(int i=0;i<k;i++){
        means[i]=new double[dim];
        // 内存空间初始化，memset与malloc相比，可以初始化内存空间
        memset(means[i], 0, sizeof(double) *dim);
    }
}
Kmeans::~Kmeans() {
    for(int i=0;i<k;i++){
        delete[] means[i];
    }
    delete[] means;
}
void Kmeans::Init(double *data, int N) {
    int size = N;
    auto *sample = new double[dim];
    for(int i=0;i<k;i++){
        int select = i*size/k;
        for(int j=0;j<dim;j++){
            sample[j]=data[select*dim + j];
        }
        memcpy(means[i], sample, sizeof(double)*dim);
    }
    delete[] sample;
}
double Kmeans::Compute_Distance(double *x, double *u, int Dim_NUM) {
    double temp=0;
    for(int i=0;i<Dim_NUM;i++){
        temp +=pow((x[i]-u[i]),2);
    }
    return sqrt(temp);
}
double Kmeans::Get_Label(double *x, int *label) {
    double distance = -1;
    for(int i=0; i<k;i++){
        // 寻找与均值最近的一类作为它的类别归属
        double temp = Compute_Distance(x,means[i],dim);
        if(temp<distance || distance==-1){
            distance = temp;
            *label = i;
        }
    }
    return distance;
}
void Kmeans::Cluster(double *data, int N, int *Label) {
    int size = N;
    assert(size>k);
    Init(data,N);

    auto *x = new double[dim];
    int label = -1;
    double cnt = 0;
    double Last_Cost = 0;
    double Curr_Cost = 0;
    int unchanged = 0;
    // 统计每一类有多少个点
    int *counts = new int[k];
    auto **next_means = new double *[k];
    for(int i =0;i<k;i++){
        next_means[i]=new double[dim];
    }
    while(1) {
        memset(counts, 0, sizeof(int) * k);
        for (int i = 0; i < k; i++) {
            memset(next_means[i], 0, sizeof(double) * dim);
        }
        Last_Cost = Curr_Cost;
        Curr_Cost = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < dim; j++) {
                x[j] = data[i * dim + j];
            }
            Curr_Cost += Get_Label(x, &label);
            counts[label]++;
            for (int d = 0; d < dim; d++) {
                next_means[label][d] += x[d];
            }
        }
        Curr_Cost /= size;
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                for (int d = 0; d < dim; d++) {
                    next_means[i][d] /= counts[i];
                }
                memcpy(means[i], next_means[i], sizeof(double) * dim);
            }
        }
        cnt++;
        if (fabs(Last_Cost - Curr_Cost) < eps * Last_Cost) {
            unchanged++;
        }
        if (cnt >= Max_ITE || unchanged > 3) {
            break;
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < dim; j++) {
            x[j] = data[i * dim + j];
        }
        Get_Label(x, &label);
        Label[i] = label;
    }
    delete[] counts;
    delete[] x;
    for (int i = 0; i < k; i++) {
        delete[] next_means[i];
    }
}
