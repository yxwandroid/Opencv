//
//  MyfindContours.cpp
//  ocr
//
//  Created by 杨学武 on 2017/5/4.
//  Copyright © 2017年 yxw. All rights reserved.
//

#include "MyfindContours.hpp"

#include <cstdio>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
/*
 第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
 cv2.RETR_EXTERNAL表示只检测外轮廓
 cv2.RETR_LIST检测的轮廓不建立等级关系
 cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
 cv2.RETR_TREE建立一个等级树结构的轮廓。
 */
int MyfindContenr(){
    string str="/Users/yangxuewu/Downloads/findContours.png";
    string str1="/Users/yangxuewu/Downloads/myimageGood1.png";
    
    Mat src = imread(str1, CV_LOAD_IMAGE_COLOR);
    Mat src_gray = imread(str1, CV_LOAD_IMAGE_GRAYSCALE);
    Mat contoursImg = src.clone();
    
    Mat edge;
    blur(src_gray, src_gray, Size(3,3));  //均值滤波
    Canny(src_gray, edge, 50, 150, 3);    // 边缘检测
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    RNG rng(12345);
    findContours(edge, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    for(int i = 0; i<contours.size(); i++){
        Scalar color = Scalar( 100, 100);
        drawContours(contoursImg, contours, i, color, 2, 8, hierarchy);
    }
    
    imshow("origin", src);
    imshow("result", contoursImg);
    waitKey(0);
    
    return 0;
}
