//
// Created by Robert on 2019/6/5.
//
#include "../include/GUI.h"
using namespace cv;

const int Offset_X =200;
void GUI::Init_Window() {
    cvui::init(WINDOW_NAME);
    Window = Mat(1000, 1000, CV_8UC3, Scalar(55,55,55));
}
void GUI::Init_State() {
    Pixels_foreground.clear();
    Pixels_background.clear();

    Before_GraphCut = true;
    draw_state = UNDRAWING;
    Draw_Flag = 0;//DRAW_RECT
    Mask = false;
    mask.setTo(Scalar::all(0));
}
void GUI::Load_Image(string path) {
    // load color image, so the flag is 1
    Image = imread(path,1);
    mask.create(Image.size(), CV_8UC1);
}
void GUI::Listen_Mouse() {
    Point cursor = cvui::mouse();
    // read the location of mouse
    int x=cursor.x-Offset_X;
    int y=cursor.y;
    static int X_begin=0, Y_begin=0;
    if(x>0){
        if(cvui::mouse(cvui::LEFT_BUTTON, cvui::DOWN)){
            if(draw_state == UNDRAWING){
                draw_state = DRAWING;
                X_begin=x, Y_begin=y;
            }
        }
        else if(cvui::mouse(cvui::LEFT_BUTTON, cvui::UP)){
            if(draw_state == DRAWING){
                if(Draw_Flag == DRAW_RECT){
                    mask.setTo(GC_BGD);// set to background
                    (mask(ChoosingBox)).setTo(Scalar(GC_PR_FGD));//set to probably foreground
                    draw_state = UNDRAWING;
                    ChoosingBox = Rect(Point(X_begin, Y_begin), Point(x,y));
                }
                else{
                    Mask = true;
                }
            }
        }
        else if(cvui::mouse(cvui::IS_DOWN)){
            if(draw_state == DRAWING){
                if(Draw_Flag == DRAW_RECT){
                    ChoosingBox = Rect(Point(X_begin, Y_begin),Point(x,y));
                }
                else if(Draw_Flag == DRAW_BACK){
                    Pixels_background.emplace_back(Point(x,y));
                    circle(mask, Point(x,y),2,GC_BGD);
                }
                else if(Draw_Flag == DRAW_FORE){
                    Pixels_foreground.emplace_back(Point(x,y));
                    circle(mask, Point(x,y),2 ,GC_FGD);
                }
            }
        }
    }
}
void GUI::Show() {
    Mat show;
    if(Image.empty()){
        cout<<"ERROR: No image load"<<endl;
        return;
    }
    if(Before_GraphCut){
        Image.copyTo(show);
    }
    else{
        Image.copyTo(show,mask & 1);
    }
    for(int i=0; i<Pixels_background.size();++i){
        circle(show,Pixels_background[i],2,Scalar(0,0,255));// BGD
    }
    for(int i=0;i<Pixels_foreground.size();++i){
        circle(show,Pixels_foreground[i],2,Scalar(255,0,0));// FGD
    }
    if(!ChoosingBox.empty()){
        rectangle(show,Point(ChoosingBox.x,ChoosingBox.y), Point(ChoosingBox.x+ChoosingBox.width, ChoosingBox.y+ ChoosingBox.height),
                Scalar(255,255,255),1);
    }
    cvui::image(Window,Offset_X,0,show);
    cvui::update();
    cvui::imshow(WINDOW_NAME, Window);
}
void GUI::Listen_Button() {
    if(cvui::button(Window,0,0,"BackGround")){
        printf("press background");
        Draw_Flag = DRAW_BACK;
    }
    if(cvui::button(Window,0,30,"ForeGround")){
        printf("press foreground");
        Draw_Flag = DRAW_FORE;
    }
    if(cvui::button(Window,0,60,"Reset")){
        printf("press reset");
        Init_State();
    }
    if(cvui::button(Window,0,90,"Do GrabCut")){
        printf("press do grabcut\n");
        if(Init_grabCut2D())
            cout<<"success"<<endl;
    }
    if(cvui::button(Window,0,120,"Border Matting")){
        printf("press border matting");
        BorderMatting bm;
        Mat rst = Mat(Image.size(), Image.type());
        Image.copyTo(rst);
        for (int i = 0; i<rst.rows; i++)
            for (int j = 0; j < rst.cols; j++)
            {
                if (mask.at<uchar>(i, j) == 0)
                    rst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            }
        bm.Initialize(Image, mask);
        bm.Run();
    }
    if(cvui::button(Window,0,150,"Exit")){
        Exit = true;
    }
}
bool GUI::If_Exit() {
    return Exit;
}
bool GUI::Init_grabCut2D() {
    if(ChoosingBox.empty())
        return false;
    printf("Init grabcut\n");
    if(Mask){
        Mat bgdModel,fdgModel;
        grabCut2D.GrabCut(Image,mask,ChoosingBox,bgdModel,fdgModel,2,2);//做两次迭代
    }
    return true;
}