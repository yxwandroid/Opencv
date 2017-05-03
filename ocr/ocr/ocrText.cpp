//
//  ocrText.cpp
//  ocr
//
//  Created by 杨学武 on 2017/5/3.
//  Copyright © 2017年 yxw. All rights reserved.
//
#include <tesseract/baseapi.h>
#include "ocrText.hpp"
#include <assert.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;
int getTextOcr(){
    tesseract::TessBaseAPI *tess = NULL;
    tess = new tesseract::TessBaseAPI();
    string mylang= "/Users/yangxuewu/Desktop/tesseract/tessdata";
    tess->Init(mylang.c_str(), "eng");
    string outImagePath="/Users/yangxuewu/Downloads/eng.png";
    string outImagePath1="/Users/yangxuewu/Downloads/ocrtext.png";
  
    cv::Mat img1=cv::imread(outImagePath1);
    imshow("gray", img1);
    waitKey(0);
    assert(tess && "you must init before calling this method.");
    tess->Clear();
    tess->SetImage(img1.data,img1.cols,img1.rows,img1.channels(),img1.step1());
    char* outText = tess->GetUTF8Text();
    cout <<"-----\n"<<outText<<"----"<<endl;
    
    return 0;
   
}
