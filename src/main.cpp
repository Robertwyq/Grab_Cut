/**
 * 计算摄影学大作业-GrabCut
 * 简介：
 * GrabCut是一个经典的图像分割算法，它结合手工交互和Graph-Cut算法来对静态图像进行高效的前景和背景的分割
 * 姓名：王宇琪
 * 学号：3160103829
 * 日期：2019.06.01
 */

#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#define CVUI_IMPLEMENTATION // This definition must be write before the include of cvui
#include "../include/GUI.h"
#include "../include/block.h"
#include "../include/graph.h"
#include "../include/GCApplication.h"
using namespace std;
using namespace cv;

// global variable
GUI gui;
GCApplication gcapp;
static void help() {
    std::cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
                 "and then grabcut will attempt to segment it out.\n"
                 "Call:\n"
                 "./grabcut <image_name>\n"
                 "\nSelect a rectangular area around the object you want to segment\n" <<
              "\nHot keys: \n"
              "\tESC - quit the program\n"
              "\tr - restore the original image\n"
              "\tn - next iteration\n"
              "\n"
              "\tleft mouse button - set rectangle\n"
              "\n"
              "\tCTRL+left mouse button - set GC_BGD pixels\n"
              "\tSHIFT+left mouse button - set CG_FGD pixels\n"
              "\n"
              "\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
              "\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}

static void on_mouse( int event, int x, int y, int flags, void* param ) {
    gcapp.mouseClick( event, x, y, flags, param );
}
int main(int argc, const char *argv[])
{
    String file_path;
    // the path of the image, you can change it
    file_path = "./testdata/5.jpg";
    bool UI = true;
    if(UI){
        help();
        const string winName = "image";
        cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );
        cvSetMouseCallback( winName.c_str(), on_mouse, 0 );
        // RGB image
        Mat image = imread(file_path,1);
        gcapp.setImageAndWinName( image, winName );
        gcapp.showImage();
        while(true){
            int c = cvWaitKey(0);
            switch( (char) c )
            {
                case '\x1b':
                    cout << "Exiting " << endl;
                    goto exit;
                case 'r':
                    cout << endl;
                    gcapp.reset();
                    gcapp.showImage();
                    break;
                case 'b':
                    cout << "Border Matting! " << endl;
                    gcapp.bm.Show_Edge();
                    gcapp.BoardMatting();
                    break;
                case 'o':
                    cout << "original image" << endl;
                    gcapp.showOriginalImage();
                case 'n':
                    int iterCount = gcapp.getIterCount();
                    cout << "<" << iterCount << ">";
                    int newIterCount = gcapp.nextIter();
                    if( newIterCount > iterCount ){
                        gcapp.showImage();
                        cout << iterCount << ">" << endl;
                    }
                    else
                        cout << "rect must be determined>" << endl;
                    break;
            }
        }
        exit:
        cvDestroyWindow( winName.c_str() );
        return 0;
    }
    else{
        gui.Init_Window();
        gui.Load_Image(file_path);
        gui.Init_State();
        while(true){
            gui.Listen_Button();
            gui.Listen_Mouse();
            gui.Show();
            if(waitKey(20)==30 || gui.If_Exit()){
                break;
            }
        }
        waitKey(0);
        return 0;
    }


    /**
     * Compare
     * opencv 自带的grabcut对比分析
    */
//    cv::Mat image = cv::imread(file_path,1) ;
//    cout<<image.size()<<endl;
//    cv::Rect rectangle(30 , 120 , 880 , 560) ;//大致圈定图像上的前景对象
//
//    cv::Mat result ;
//    cv::Mat bgModel , fgModel ;
//
//    //grabCut()最后一个参数为cv::GC_INIT_WITH_RECT时
//    clock_t start, end;
//    start = clock();
//    cv::grabCut(image , result , rectangle , bgModel , fgModel ,4 , cv::GC_INIT_WITH_RECT) ;
//    end = clock();
//    cout<<"Time："<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;
//    cv::compare(result , cv::GC_PR_FGD , result , cv::CMP_EQ) ;
//    //result = result & 1 ;
//    cv::Mat foreground(image.size() , CV_8UC3 , cv::Scalar(0 , 0, 0)) ;
//    image.copyTo(foreground , result) ;
//
//    cv::imshow("Foreground" , foreground);
//    cv::waitKey(0) ;
//    return 0 ;



}