//
//  main.cpp
//  ocr
//
//  Created by 杨学武 on 2017/5/3.
//  Copyright © 2017年 yxw. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencvTest.hpp"
#include "ocrText.hpp"
void resize();
using namespace cv;
int main(int argc, const char * argv[]) {
    findContours();
//    getTextOcr();
//    resize();
    return 0;
}



void resize(){
    String str="/Users/yangxuewu/Downloads/result2.png";
    Mat gray=imread(str);
    double scale=0.4;
    Size ResImgSiz = Size(gray.cols*scale, gray.rows*scale);
    Mat ResImg = Mat(ResImgSiz, gray.type());
    resize(gray, ResImg, ResImgSiz, CV_INTER_CUBIC);
    imshow("w", ResImg);
    waitKey(0);
    


}
