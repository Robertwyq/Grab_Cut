//
// Created by Robert on 2019/6/5.
//

#ifndef GRAB_CUT_GUI_H
#define GRAB_CUT_GUI_H

#include <iostream>
#include "cvui.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "GrabCut.h"
#include "BorderMatting.h"

#define WINDOW_NAME "GrabCut"
using namespace std;
enum Draw_State{
    UNDRAWING,
    DRAWING,
};

class GUI{
public:
    /**
     * Init the GUI window
     */
    void Init_Window();
    void Init_State();
    void Load_Image(string path);
    void Listen_Mouse();
    void Show();
    void Listen_Button();
    bool If_Exit();
    GrabCut2D grabCut2D;
    bool Init_grabCut2D();
private:
    cv::Rect ChoosingBox;

    vector<cv::Point> Pixels_foreground;
    vector<cv::Point> Pixels_background;

    static const int DRAW_BACK = 1;
    static const int DRAW_FORE = 2;
    static const int DRAW_RECT = 0;
    //
    cv::Mat Window;
    cv::Mat Image;
    cv::Mat mask;
    // flags
    bool Exit_Drawing;
    bool Before_GraphCut;
    bool Mask;
    bool Exit;

    Draw_State draw_state;
    int Draw_Flag;
};
#endif //GRAB_CUT_GUI_H
