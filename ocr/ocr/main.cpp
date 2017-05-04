//
//  main.cpp
//  ocr
//
//  Created by 杨学武 on 2017/5/3.
//  Copyright © 2017年 yxw. All rights reserved.
//
#include <tesseract/baseapi.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencvTest.hpp"
#include "ocrText.hpp"
void resize();
int findlunkuo();
int getTextOcr(cv::Mat);
int findContours();
using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
   
    findContours();
    return 0;
}

struct Vec2D{
    int x = 0,y = 0;
};

struct Box{
    Vec2D min;
    Vec2D max;
    
    void add(Vec2D vec){
        min.x = std::min(vec.x,min.x);
        min.y = std::min(vec.y,min.y);
        
        max.x = std::max(vec.x,max.x);
        max.y = std::max(vec.y,max.y);
    }
};

int getTextOcr(Mat img1){
    tesseract::TessBaseAPI *tess = NULL;
    tess = new tesseract::TessBaseAPI();
    String mylang= "/Users/yangxuewu/Desktop/tesseract/tessdata";
    tess->Init(mylang.c_str(), "eng");
    tess->SetVariable("tessedit_char_whitelist","0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    //tess->SetVariable("tessedit_char_whitelist","0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
//    String outImagePath="/Users/yangxuewu/Downloads/eng.png";
//    String outImagePath1="/Users/yangxuewu/Downloads/ocrtext.png";
    
    // cv::Mat img1=mat;
    //imshow("gray", img1);
    //waitKey(0);
    //assert(tess && "you must init before calling this method.");
    tess->Clear();
    tess->SetImage(img1.data,img1.cols,img1.rows,img1.channels(),img1.step1());
    char* outText = tess->GetUTF8Text();
    cout <<"-----\n"<<outText<<"----"<<endl;
    return 0;
    
}

//************************************************************* 框出文字区域     优化过的
int findContours()
{
    //    识别成功率比较好的
    String filename="/Users/yangxuewu/Downloads/myimageGood1.png";
    String filename2="/Users/yangxuewu/Downloads/mytex.png";
    String filename4="/Users/yangxuewu/Downloads/WechatIMG11.jpeg";
    String filename5="/Users/yangxuewu/Downloads/Aa.png";
    
    //识别困难
    String filename3="/Users/yangxuewu/Downloads/mytext.png";
    String outImagePath="/Users/yangxuewu/Downloads/ocrtext.png";
    
    String result="/Users/yangxuewu/Downloads/resuld.JPG";
    //String filename= "/Users/yangxuewu/Downloads/骨钉/jkfs-22-232-g001-l.jpg";
    // String filename= "/Users/yangxuewu/Downloads/骨钉/11111.png";
    //  String filename= "/Users/yangxuewu/Downloads/骨钉/A1.png";
    //String filename= "/Users/yangxuewu/Downloads/骨钉/goodImge.JPG";
    
    Mat large = imread(filename);
    Mat rgb;
    // downsample and use it for processing
    pyrDown(large, rgb);  //缩小过程
    //pyrUp(large,rgb,Size(3,3));//  放大过程
    //pyrUp( large, rgb, Size(large.cols*1.2, large.rows*1.2));
    Mat small;
    cvtColor(rgb, small, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel); //形态学梯度
    // binarize
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU); //二值化
    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(14, 3));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    // filter contours
    
    Box box;
    box.min.x = 9999999;
    box.min.y = 9999999;
    
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
    {
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
        // ratio of non-zero pixels in the filled region
        double r = (double)countNonZero(maskROI)/(rect.width*rect.height);
        
        if (r >.35 /* assume at least 45% of the area is filled if it contains text */
            &&(rect.height > 8&& rect.width >18) /* constraints on region size */
            /* these two conditions alone are not very robust. better to use something
             like the number of significant peaks in a horizontal projection as a third condition */){
                 //圈出区域
                 // rectangle(rgb, rect, Scalar(0, 0, 0), 2);
                 
                 int x=rect.x;
                 int y= rect.y;
                 Vec2D vec;
                 
                 vec.x = x;
                 vec.y = y;
                 box.add(vec);
                 
                 
                 Vec2D vecMax = vec;
                 vecMax.x += rect.width;
                 vecMax.y += rect.height;
                 box.add(vecMax);
                 
                 Rect myRect(rect.x, rect.y, rect.width, rect.height);
                 Mat gray = rgb(myRect);
                 string s1=to_string(idx);
                 imshow(s1,gray);
                 cout << "Value of str is : " <<idx <<"  "<< x << " " << y << endl;
             }
    }
    
    //获得截取的最小值
    Rect newRect(box.min.x, box.min.y, box.max.x- box.min.x,  box.max.y- box.min.y);
    Mat gray = rgb(newRect);
    
    cvtColor(gray, gray, CV_BGR2GRAY);
    threshold(gray,gray,130,255,THRESH_BINARY);
    
    //进行放大处理
    double scale=1;
    Size ResImgSiz = Size(gray.cols*scale, gray.rows*scale);
    Mat ResImg = Mat(ResImgSiz, gray.type());
    resize(gray, ResImg, ResImgSiz, CV_INTER_CUBIC);
    
    //矩形: MORPH_RECT 交叉形: MORPH_CROSS  椭圆形: MORPH_ELLIPSE
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat out;
    //进行膨胀操作
    dilate(ResImg,out, element);
    
    
    //    Mat out;
    //    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    //    //进行形态学操作
    //    morphologyEx(ResImg,out, MORPH_OPEN, element);
    //
    
    imshow("newRect",out);
    imwrite(outImagePath,out);
    
    getTextOcr(out);  //进行文字识别
    
    imshow("rgb",rgb);
    imwrite(result, rgb);
    waitKey(1000000);
    
    return 0;
}












//int findlunkuo(){
//    String str="/Users/yangxuewu/Downloads/ocrtext.png";
//
//    cv::Mat image = cv::imread(str , 0) ;
//    std::vector<std::vector<cv::Point>> contours ;
//    //获取轮廓不包括轮廓内的轮廓
//    cv::findContours(image , contours ,
//                     CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE) ;
//    cv::Mat result(image.size() , CV_8U , cv::Scalar(255)) ;
//    cv::drawContours(result , contours ,
//                     -1 , cv::Scalar(0) , 2) ;
//    cv::imshow("resultImage" , result) ;
//    
//    //获取所有轮廓包括轮廓内的轮廓
//    std::vector<std::vector<cv::Point>> allContours ;
//    cv::Mat allContoursResult(image.size() , CV_8U , cv::Scalar(255)) ;
//    cv::findContours(image , allContours ,
//                     CV_RETR_LIST , CV_CHAIN_APPROX_NONE) ;
//    cv::drawContours(allContoursResult , allContours ,-1 ,
//                     cv::Scalar(0) , 2) ;
//    cv::imshow("allContours" , allContoursResult) ;
//    
//    //获取轮廓的等级
//    std::vector<cv::Vec4i> hierarchy ;
//    cv::findContours(image , contours , hierarchy , CV_RETR_TREE ,
//                     CV_CHAIN_APPROX_NONE) ;
//    
//    cv::waitKey(0) ;  
//    return 0 ;  
//}
//
//// 图片缩放
//void resize(){
//    String str="/Users/yangxuewu/Downloads/result2.png";
//    Mat gray=imread(str);
//    double scale=0.4;
//    Size ResImgSiz = Size(gray.cols*scale, gray.rows*scale);
//    Mat ResImg = Mat(ResImgSiz, gray.type());
//    resize(gray, ResImg, ResImgSiz, CV_INTER_CUBIC);
//    imshow("w", ResImg);
//    waitKey(0);
//    
//
//
//}
