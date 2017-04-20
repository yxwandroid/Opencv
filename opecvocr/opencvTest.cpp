//
//  opencvTest.cpp
//  opecvocr
//
//  Created by 杨学武 on 2017/4/18.
//  Copyright © 2017年 yxw. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/ml.hpp>
#include "opencvTest.hpp"

using namespace std;
using namespace cv;
void imageCvtColor(){
    // insert code here...
    //    std::cout << "Hello, World!\n";
    String str="/Users/yangxuewu/Downloads/enenen.png";
    
    String imagePath="/Users/yangxuewu/Desktop/癌蚌photo/demo.png";
//    String imagePath="/Users/yangxuewu/Downloads/221.png";

    String outImagePath="/Users/yangxuewu/Desktop/癌蚌photo/result.jpg";
    
    Mat img = imread(imagePath);
    Mat gray;
    
    cvtColor(img, gray, CV_BGR2GRAY);
    threshold(gray,gray,140,255,THRESH_BINARY);
    // cvNamedWindow("游戏原画");
    //  imshow("imageCvtColor",gray);
    
    
  //  namedWindow("gray", CV_WINDOW_NORMAL);
  //  imshow("img", img);
    imwrite(outImagePath,gray);
  //  imshow("gray", gray);

    waitKey(0);
    
}

