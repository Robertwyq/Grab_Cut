//
// Created by Robert on 2019/6/5.
//

#include "../include/GrabCut.h"
using namespace std;

void GrabCut2D::InitMask(Mat &Mask, Size size, Rect rect) {
    Mask.create(size,CV_8UC1);
    Mask.setTo(GC_BGD);
    rect.x=max(0,rect.x);
    rect.y=max(0,rect.y);
    rect.width = min(rect.width,size.width-rect.x);
    rect.height=min(rect.height,size.height-rect.y);
    (Mask(rect)).setTo(Scalar(GC_PR_FGD));
}
void GrabCut2D::InitGMM(const Mat &img, const Mat &mask, GMM &fgdGMM, GMM &bgdGMM) {
    vector<Vec3f> Pixel_Background;
    vector<Vec3f> Pixel_Frontground;
    Mat Labels_BGD;
    Mat Labels_FGD;
    // 根据mask进行采样，分别采前景背景和背景样本
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(mask.at<uchar>(i,j)==GC_BGD || mask.at<uchar>(i,j)==GC_PR_BGD){
                Pixel_Background.push_back((Vec3f)img.at<Vec3b>(i,j));
            }
            else{
                Pixel_Frontground.push_back((Vec3f)img.at<Vec3b>(i,j));
            }
        }
    }
    // Vector-> Mat N*3 的矩阵
    Mat BGD_Sample(int(Pixel_Background.size()),3,CV_32FC1, &Pixel_Background[0][0]);
    Mat FGD_Sample(int(Pixel_Frontground.size()),3,CV_32FC1, &Pixel_Frontground[0][0]);
    // Kmeans聚类初始化
    // 初始化完成后，每个点都有一个相应的label与之对应，存储在Label中
    kmeans(BGD_Sample, bgdGMM.Get_K(),Labels_BGD,TermCriteria(TermCriteria::MAX_ITER,10,0.0),0,KMEANS_PP_CENTERS);
    kmeans(FGD_Sample, fgdGMM.Get_K(),Labels_FGD,TermCriteria(TermCriteria::MAX_ITER,10,0.0),0,KMEANS_PP_CENTERS);
    // 利用Kmeans得到的初值，使用GMM进行聚类
    bgdGMM.Init_Learning();
    for(int i=0;i<Pixel_Background.size();i++){
        bgdGMM.Add_Pixel(Labels_BGD.at<int>(i,0),Pixel_Background[i]);
    }
    bgdGMM.Training();
    fgdGMM.Init_Learning();
    for(int i=0;i<Pixel_Frontground.size();i++){
        fgdGMM.Add_Pixel(Labels_FGD.at<int>(i,0),Pixel_Frontground[i]);
    }
    fgdGMM.Training();
}
void GrabCut2D::GrabCut(const cv::Mat &_img, cv::Mat &_mask, cv::Rect rect, cv::Mat &_bgdModel,cv::Mat &_fgdModel, int iterCount, int mode ){
    if(iterCount<=0){
        cout<<"ERROR: 输出错误的迭代次数！"<<endl;
        return;
    }
    clock_t start, end;
    start = clock();

    /**
     * 3: Init GMM
     */
    GMM bgdGMM, fgdGMM;
    if(mode==GC_WITH_RECT|| mode==GC_WITH_MASK){
        if(mode==GC_WITH_RECT){
            InitMask(_mask,_img.size(),rect);
        }
        InitGMM(_img,_mask,bgdGMM,fgdGMM);
    }
    else if(mode==GC_CUT){
        bgdGMM=GMM::MattoGMM(_bgdModel);
        fgdGMM=GMM::MattoGMM(_fgdModel);
    }

    /**
     * 6:Construct Graph（计算数据项t-weight和平滑想n-weight）
     */
    cv::Mat leftW, upleftW, upW, uprightW;
    const double gamma=50;
    const double lambda = 9*gamma;
    const double beta = ComputeBeta(_img);
    ComputeNWeights(_img,leftW,upleftW,upW,uprightW,beta,gamma);

    // 所有的顶点个数
    int vertex_num = _img.cols* _img.rows;
    // 所有的边的个数
    int N_num = 2*(4*vertex_num-3*(_img.cols+_img.rows)+2);
    cv::Mat GMMcompID(_img.size(),CV_32SC1);
    for(int i=0;i<iterCount;i++){
        Graph<double,double,double> graph(vertex_num,N_num);
        // 计算新的Mask对于每个像素对应的高斯分量，存入GMMcompID
        Assign_GMMID(_img,_mask,bgdGMM,fgdGMM,GMMcompID);
        // 重新学习混合高斯模型的系数
        LearnGMMs(_img,_mask,GMMcompID,bgdGMM,fgdGMM);
//        cout<<"3:"<<_mask<<endl;
        Construct_Graph(_img,_mask,bgdGMM,fgdGMM,lambda,leftW,upleftW,upW,uprightW,graph);
//        cout<<"4:"<<_mask<<endl;
        // 使用maxflow库进行分割，更新mask，最终输出
        Estimate_Segmentation(graph,_mask);
//        cout<<"5:"<<_mask<<endl;
    }
    // 更新高斯模型，将结果mask输出
    _fgdModel = fgdGMM.GMMtoMat();
    _bgdModel = bgdGMM.GMMtoMat();
    end = clock();
    cout<<"Time："<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;
}
/**
计算一个img大小的mask，存储每个像素属于哪个高斯分量
*/
void GrabCut2D::Assign_GMMID(const cv::Mat &img, const cv::Mat &Mask, const GMM &bgdGMM, const GMM &fgdGMM, cv::Mat &compIdxs) {
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            cv::Vec3d color = img.at<cv::Vec3b>(y,x);
            // 为每个像素指定属于哪个GMM的哪个分量
            if(Mask.at<uchar>(y,x)==cv::GC_BGD || Mask.at<uchar>(y,x)==cv::GC_PR_BGD){
                compIdxs.at<int>(y,x) = bgdGMM.Get_MostPossible_ID(color);
            }
            else{
                compIdxs.at<int>(y,x) = fgdGMM.Get_MostPossible_ID(color);
            }
        }
    }
}
/**
根据上述计算出的Mask，重新训练高斯模型
 */
void GrabCut2D::LearnGMMs(const cv::Mat &img, const cv::Mat &mask, const cv::Mat &compIdxs, GMM &bgdGMM, GMM &fgdGMM) {
    bgdGMM.Init_Learning();
    fgdGMM.Init_Learning();
    for (int i = 0; i < bgdGMM.Get_K(); i++) {
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                if (compIdxs.at<int>(y, x) == i) {
                    if (mask.at<uchar>(y, x) == cv::GC_BGD || mask.at<uchar>(y, x) == cv::GC_PR_BGD)
                        bgdGMM.Add_Pixel(i, img.at<cv::Vec3b>(y, x));
                    else
                        fgdGMM.Add_Pixel(i, img.at<cv::Vec3b>(y, x));
                }
            }
        }
    }
    bgdGMM.Training();
    fgdGMM.Training();
}
// 计算光滑性函数公式中的beta值
/**
Calculate beta - parameter of GrabCut algorithm.
beta = 1/(2*avg(sqrt(||color[i] - color[j]||)))
第二项平滑项中的指数项的beta，用来调整高或者低对比度时，两个邻域像素的差别的影响的，例如在低对比度时，两个邻域像素的差别会很小，因此
要乘以一个较大的beta来放大这个差别，同理，在高对比度时，则需要缩小本身就比较大的差别。
Note： 根据论文公式，只需计算四个方向
*/
double GrabCut2D::ComputeBeta(const cv::Mat &img) {
    double beta = 0;
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            cv::Vec3d color = img.at<cv::Vec3b> (y,x);
            // 1.left
            if(x>0){
                cv::Vec3d Diff=color-(cv::Vec3d)img.at<cv::Vec3b>(y,x-1);
                beta+=Diff.dot(Diff);
            }
            // 2.upleft
            if(y>0&&x>0){
                cv::Vec3d Diff =color-(cv::Vec3d)img.at<cv::Vec3b>(y-1,x-1);
                beta+=Diff.dot(Diff);
            }
            // 3.up
            if(y>0){
                cv::Vec3d Diff =color-(cv::Vec3d)img.at<cv::Vec3b>(y-1,x);
                beta+=Diff.dot(Diff);
            }
            // upright
            if(y>0&&x<img.cols-1){
                cv::Vec3d Diff =color-(cv::Vec3d)img.at<cv::Vec3b>(y-1,x+1);
                beta+=Diff.dot(Diff);
            }
        }
    }
    if(beta<std::numeric_limits<double>::epsilon()){
        beta = 0;
    }
    // 除以临接距离的数量
    else{
        beta = 1.f/(2*beta/(4*img.cols*img.rows-3*img.cols-3*img.rows+2));
    }
    return beta;
}
/**
计算边的权重,非终端顶点的权重值，计算公式依据V函数，权重结果放在W的四个Mat型矩阵中
*/
void GrabCut2D::ComputeNWeights(const cv::Mat &img, cv::Mat &leftW, cv::Mat &upleftW, cv::Mat &upW, cv::Mat &uprightW,
                                double beta, double gamma) {
    leftW.create(img.rows,img.cols,CV_64FC1);
    upleftW.create(img.rows,img.cols,CV_64FC1);
    upW.create(img.rows,img.cols,CV_64FC1);
    uprightW.create(img.rows,img.cols,CV_64FC1);
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            cv::Vec3d color=img.at<cv::Vec3b>(y,x);
            // 1.left
            if(x>0){
                cv::Vec3d Diff = color -(cv::Vec3d)img.at<cv::Vec3b>(y,x-1);
                leftW.at<double>(y,x)= gamma*exp(-beta*Diff.dot(Diff));
            }
            else{
                leftW.at<double>(y,x)=0;
            }
            // 2.upleft
            if(x>0&&y>0){
                cv::Vec3d Diff = color -(cv::Vec3d)img.at<cv::Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x)= gamma/sqrt(2.0)*exp(-beta*Diff.dot(Diff));
            }
            else{
                upleftW.at<double>(y,x)=0;
            }
            // 3.up
            if(y>0){
                cv::Vec3d Diff = color-(cv::Vec3d)img.at<cv::Vec3b>(y-1,x);
                upW.at<double>(y,x)= gamma*exp(-beta*Diff.dot(Diff));
            }
            else{
                upW.at<double>(y,x)=0;
            }
            // 4.upright
            if(y>0&&x<img.cols-1){
                cv::Vec3d Diff= color-(cv::Vec3d)img.at<cv::Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x)= gamma/sqrt(2.0)*exp(-beta*Diff.dot(Diff));
            }
            else{
                uprightW.at<double>(y,x)=0;
            }
        }
    }
}
/**：Construct Graph
 * 通过计算得到的能量项构建图，图的顶点为像素点，图的边由两部分构成
 * 边1：每个顶点与Sink汇点t（代表背景） 背景点即为汇入点
 * 边2：源点source（代表前景）连接的边    前景点即为源点
 * 根据Mask判断前景还是后景 【0，1，2，3】
 * 加点、加边、建图
*/
void GrabCut2D::Construct_Graph(const cv::Mat &img, const cv::Mat &Mask, const GMM &bgdGMM, const GMM &fgdGMM,
                                double lamda, const cv::Mat &leftW, const cv::Mat &upleftW, const cv::Mat &upW,
                                const cv::Mat &uprightW, Graph<double, double, double> &graph) {
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            // 1. 加点
            int vertexID = graph.add_node();
            cv::Vec3b color = img.at<cv::Vec3b>(y,x);
            // 2. t-weights
            double FromSource, ToSink;
            if(Mask.at<uchar>(y,x)==cv::GC_PR_BGD||Mask.at<uchar>(y,x)==cv::GC_PR_FGD){
                FromSource = -log(bgdGMM(color));
                ToSink = -log(fgdGMM(color));
            }
            else if(Mask.at<uchar>(y,x)==cv::GC_BGD){
                FromSource = 0;
                ToSink = lamda;
            }
            else{
                FromSource = lamda;
                ToSink = 0;
            }
            graph.add_tweights(vertexID,FromSource,ToSink);
            // 3.n-weights
            double Nweights;
            // left
            if(x>0){
                Nweights = leftW.at<double>(y,x);
                graph.add_edge(vertexID,vertexID-1,Nweights,Nweights);
            }
            // upleft
            if(x>0&&y>0){
                Nweights=upleftW.at<double>(y,x);
                graph.add_edge(vertexID,vertexID-img.cols-1,Nweights,Nweights);
            }
            // up
            if(y>0){
                Nweights = upW.at<double>(y,x);
                graph.add_edge(vertexID,vertexID-img.cols,Nweights,Nweights);
            }
            // upright
            if(y>0&&x<img.cols-1){
                Nweights=uprightW.at<double>(y,x);
                graph.add_edge(vertexID,vertexID-img.cols+1,Nweights,Nweights);
            }
        }
    }
}
/**
* 7：Estimate Segmentation（调用maxflow库进行分割）
*/
void GrabCut2D::Estimate_Segmentation(Graph<double, double, double> &graph, cv::Mat &mask) {
    graph.maxflow();
    for(int y=0;y<mask.rows;y++){
        for(int x=0;x<mask.cols;x++){
            if(mask.at<uchar>(y,x)==cv::GC_PR_BGD||mask.at<uchar>(y,x)==cv::GC_PR_FGD){
                if(graph.what_segment(y*mask.cols+x)==Graph<double,double,double>::SOURCE){
                    mask.at<uchar>(y,x) = cv::GC_PR_FGD;
                }
                else{
                    mask.at<uchar>(y,x)= cv::GC_PR_BGD;
                }
            }
        }
    }
}