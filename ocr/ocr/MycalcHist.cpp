//
//  MycalcHist.cpp
//  opecvocr
//
//  Created by 杨学武 on 2017/4/26.
//  Copyright © 2017年 yxw. All rights reserved.
//
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/ml.hpp>
#include "MycalcHist.hpp"
using namespace std;
using namespace cv;
/*!
 @typedef EvpHandler
 @abstract 极值数据结构
 @field type 1: 极大值，-1：极小值，0:其它
 @field gray 对应的灰度值
 @field count 对应灰度值的像素个数
 @field nvalue 归一化后的值[0,1]
 @field graynvalue nvalue灰度化后的值[0,255]
 @field gleft 相对左边的梯度方向
 @field gright 相对右边的梯度方向
 */
typedef struct {
    int type = 0; // 1: 极大值，-1：极小值，0:其它
    char gray; // 对应的灰度值
    int count = 0; //对应灰度值的像素个数
    double nvalue; // 归一化后的值[0,1]
    double graynvalue; // nvalue灰度化后的值[0,255]
    int gleft; // 相对左边的梯度方向
    int gright; // 相对右边的梯度方向
} EvpHandler;

EvpHandler grayCount[256]; // 极值句柄
int evpRadius = 3;  // 梯度半径
/*!
 @function gradient
 @abstract 梯度计算函数，如果同一方向梯度为0，则递归,直到计算出梯度值，或者循环结束
 @param index 碑计算的索引
 @param direction 计算方向，0:left，1:right
 */
int gradient(int index,int direction){
    int indexTmp = index;
radius:
    int nextIndex = direction == 0 ? indexTmp-- : indexTmp++;
    
    // 超出边界 返回0
    if(nextIndex < 0 || nextIndex > 255){
        return 0;
    }
    
    // 计算相对梯度值
    double gradvalue = grayCount[index].graynvalue - grayCount[nextIndex].graynvalue;
    
    // 如果，超出半径，获取下一个点
    if(gradvalue > evpRadius){
        goto radius;
    }
    
    // 如果 gradvalue == 0 递归
    if(gradvalue == 0){
        return gradient(nextIndex, direction);
    }
    
    switch (direction) {
        case 0:
            grayCount[index].gleft = grayCount > 0 ? 1 : -1;
            break;
            
        case 1:
            grayCount[index].gright = grayCount > 0 ? 1 : -1;
            break;
    }
    
    return 0;
}

int main2222(){
     String image1="/Users/yangxuewu/Downloads/goodImge.JPG";//153
    Mat src = imread(image1);
    int offset = 40;
    Mat gray, histogram(150 + offset,256 * 2, CV_8UC3);
    
    cvtColor(src, gray, CV_BGR2GRAY);
    // 统计会度值
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            grayCount[gray.at<uchar>(i, j)].count ++;
        }
    }
    // 计算最大值
    int maxGray = 0;
    for (int i = 0; i < 256; i++) {
        maxGray = max(maxGray, grayCount[i].count);
    }
    // 归一化绘制直方图
    maxGray = maxGray == 0 ? 1 : maxGray;
    
    for (int i = 0; i < 256; i++) {
        grayCount[i].nvalue =  (double) grayCount[i].count / maxGray;
        grayCount[i].graynvalue = grayCount[i].nvalue * 255;
        
        int rowIndex = ceil((1 -  grayCount[i].nvalue) * (histogram.rows - offset) + offset);
        printf("scale %f, row index %d\n",  grayCount[i].nvalue, rowIndex);
        Point root_points[1][4];
        root_points[0][0] = Point(i*2, rowIndex);
        root_points[0][1] = Point(i*2 + 2, rowIndex);
        root_points[0][2] = Point(i*2 + 2, histogram.rows - 1 );
        root_points[0][3] = Point(i*2, histogram.rows - 1 );
        
        const Point* ppt[1] = {root_points[0]};
        int npt[] = {4};
        fillPoly(histogram, ppt, npt, 1, Scalar(255,255,0));
        //        rectangle(histogram, Point(i*2, rowIndex), Point(i*2+2, histogram.rows - 1 ), Scalar(255, 255, 0));
    }
    imshow("histogram", histogram);
    
    
    waitKey(0);
    return 0;
}
