//
//  main.cpp
//  EdgeDetection
//
//  Created by hui li on 2017/3/13.
//  Copyright © 2017年 Leedian. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <math.h>
using namespace std;
using namespace cv;
Point points[30000];
//This colors the segmentations
static void floodFillPostprocess( Mat& img, const Scalar& colorDiff=Scalar::all(1) )
{
    CV_Assert( !img.empty() );
    RNG rng = theRNG();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            if( mask.at<uchar>(y+1, x+1) == 0 )
            {
                Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( img, mask, Point(x,y), newVal, 0, colorDiff, colorDiff ,8);
            }
        }
    }
}



static void floodFillPostprocess2( Mat& img, int length, const Scalar& colorDiff=Scalar::all(1) )
{
    
    CV_Assert( !img.empty() );
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) );
    
    for (int i = 0; i < length; i++) {
        double x = points[i].x;
        double y = points[i].y;
        if( mask.at<uchar>(y+1, x+1) == 0 )
        {
            //            Scalar newVal( rng(256), rng(256), rng(256) );
            Scalar newVal( 0, 0, 255 );
            floodFill( img, mask, Point(x,y), newVal, 0, colorDiff, colorDiff ,4);
        }
        
    }
    
    //    CV_Assert( !img.empty() );
    //    int area;
    //    int lo = 2;
    //    int up = 2;
    //    int flags = 4 + (255 << 8) + CV_FLOODFILL_FIXED_RANGE ;
    //
    //    Rect ccomp;
    //    for (int i = 0; i < length; i++) {
    //
    ////            Scalar newVal( rng(256), rng(256), rng(256) );
    //            Scalar newVal( 0, 0, 255 );
    //            area = floodFill(img, points[i], newVal, &ccomp, Scalar(lo, lo, lo),
    //                         Scalar(up, up, up), flags);
    //
    //
    //    }
    
}

Mat genSobel(Mat src){
    Mat grad,src_gray,src_guess;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    GaussianBlur( src, src_guess, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    /// 转换为灰度图
    cvtColor( src_guess, src_gray, CV_RGB2GRAY );
    
    /// 创建 grad_x 和 grad_y 矩阵
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    /// 求 X方向梯度
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    
    /// 求Y方向梯度
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    /// 合并梯度(近似)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    //imshow("grad", grad);
    
    return grad;
}

uchar getMaxRegionLabel(Mat& mat){
    uchar maxlabel = 0; // 最大标签
    int maxCounter = 0;
    int counter[256];
    for (int i = 0; i < 256; i++) {
        counter[i] = 0;
    }
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            uchar label = mat.at<uchar>(i,j);
            counter[label]++;
        }
    }
    
    for (int i = 1; i < 256; i++) {
        if(counter[i] > maxCounter){
            maxCounter = counter[i];
            maxlabel = i;
        }
    }
    
    return maxlabel;
}

double getRegionLabelCount(Mat& mat,  int * counter){
    double avg;
    double sum = 0;
    int num = 0;
    for (int i = 0; i < 256; i++) {
        counter[i] = 0;
    }
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            uchar label = mat.at<uchar>(i,j);
            if(label!=0){
                counter[label]++;
            }
        }
    }
    for (int i = 0; i < 256; i++) {
        if(counter[i] != 0){
            sum += counter[i];
            num++;
        }
    }
    
    avg = sum / num;
    
    return avg;
}

Mat connectRegion2(const Mat& mat, uchar seedColor){
    // 1. 当该像素的左邻像素和上邻像素为无效值时，给该像素置一个新的label值，label ++;
    // 2. 当该像素的左邻像素或者上邻像素有一个为有效值时，将有效值像素的label赋给该像素的label值；
    // 3. 当该像素的左邻像素和上邻像素都为有效值时，选取其中较小的label值赋给该像素的label值。
    Mat dst(mat.rows, mat.cols, CV_8UC1, Scalar::all(0));
    
    // 如果遇到不是红色的，label++
    // 计算最大的行索引及列索引
    int label = 0;
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            uchar color = mat.at<uchar>(i, j);
            if(color == seedColor){
                
                // 左边和上边是否有 有效区域
                uchar left = 0, leftTop = 0;
                uchar top = 0, rightTop = 0;
                if(j > 0){
                    left = dst.at<uchar>(i,j - 1);
                }
                if(i > 0){
                    top = dst.at<uchar>(i - 1,j);
                }
                
                if(i > 0 && j > 0){
                    leftTop = dst.at<uchar>(i - 1,j - 1);
                }
                
                if(i > 0 && j < dst.cols - 1){
                    rightTop = dst.at<uchar>(i - 1,j + 1);
                }
                // 如果都不是有效区域则用++label
                if(left == 0 && top == 0 && leftTop == 0 && rightTop == 0){
                    dst.at<uchar>(i,j) = ++label;
                    continue;
                }
                
                // 四个值中大于0的最小值
                uchar minLabel = 255;
                minLabel = left > 0 ? min(left, minLabel) : minLabel;
                minLabel = top > 0 ? min(top, minLabel) : minLabel;
                minLabel = leftTop > 0 ? min(leftTop, minLabel) : minLabel;
                minLabel = rightTop > 0 ? min(rightTop, minLabel) : minLabel;
                dst.at<uchar>(i,j) = minLabel;
                if(left > 0){
                    dst.at<uchar>(i,j - 1) = minLabel;
                }
                
                if(top > 0){
                    dst.at<uchar>(i - 1,j) = minLabel;
                }
                
                if(leftTop > 0){
                    dst.at<uchar>(i - 1,j - 1) = minLabel;
                }
                
                if(rightTop > 0){
                    dst.at<uchar>(i - 1,j + 1) = minLabel;
                }
                
            }
        }
    }
    
    // 从左向右
    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            if(dst.at<uchar>(i,j) != 0 ){
                
                // 左边和上边是否有 有效区域
                uchar left = 0, leftTop = 0;
                uchar top = 0, rightTop = 0;
                if(j > 0){
                    left = dst.at<uchar>(i,j - 1);
                }
                if(i > 0){
                    top = dst.at<uchar>(i - 1,j);
                }
                
                if(i > 0 && j > 0){
                    leftTop = dst.at<uchar>(i - 1,j - 1);
                }
                
                if(i > 0 && j < dst.cols - 2){
                    rightTop = dst.at<uchar>(i - 1,j + 1);
                }
                // 如果都不是有效区域则用++label
                if(left == 0 && top == 0 && leftTop == 0 && rightTop == 0){
                    
                    continue;
                }
                
                // 四个值中大于0的最小值
                uchar minLabel = 255;
                minLabel = left > 0 ? min(left, minLabel) : minLabel;
                minLabel = top > 0 ? min(top, minLabel) : minLabel;
                minLabel = leftTop > 0 ? min(leftTop, minLabel) : minLabel;
                minLabel = rightTop > 0 ? min(rightTop, minLabel) : minLabel;
                minLabel = min(dst.at<uchar>(i,j), minLabel) ;
                
                dst.at<uchar>(i,j) = minLabel;
                if(left > 0){
                    dst.at<uchar>(i,j - 1) = minLabel;
                }
                
                if(top > 0){
                    dst.at<uchar>(i - 1,j) = minLabel;
                }
                
                if(leftTop > 0){
                    dst.at<uchar>(i - 1,j - 1) = minLabel;
                }
                
                if(rightTop > 0){
                    dst.at<uchar>(i - 1,j + 1) = minLabel;
                }
                
            }
        }
    }
    
    // 从右向左
    for(int i = 0; i < dst.rows; i++){
        for(int j = dst.cols - 1; j > -1; j--){
            if(dst.at<uchar>(i,j) != 0 ){
                
                // 左边和上边是否有 有效区域
                uchar right = 0, leftTop = 0;
                uchar top = 0, rightTop = 0;
                if(j < dst.cols - 2){
                    right = dst.at<uchar>(i,j + 1);
                }
                if(i > 0){
                    top = dst.at<uchar>(i - 1,j);
                }
                
                if(i > 0 && j > 0){
                    leftTop = dst.at<uchar>(i - 1,j - 1);
                }
                
                if(i > 0 && j < dst.cols - 2){
                    rightTop = dst.at<uchar>(i - 1,j + 1);
                }
                // 如果都不是有效区域则用++label
                if(right == 0 && top == 0 && leftTop == 0 && rightTop == 0){
                    
                    continue;
                }
                
                // 四个值中大于0的最小值
                uchar minLabel = 255;
                minLabel = right > 0 ? min(right, minLabel) : minLabel;
                minLabel = top > 0 ? min(top, minLabel) : minLabel;
                minLabel = leftTop > 0 ? min(leftTop, minLabel) : minLabel;
                minLabel = rightTop > 0 ? min(rightTop, minLabel) : minLabel;
                minLabel = min(dst.at<uchar>(i,j), minLabel) ;
                dst.at<uchar>(i,j) = minLabel;
                
                if(right > 0){
                    dst.at<uchar>(i,j + 1) = minLabel;
                }
                
                if(top > 0){
                    dst.at<uchar>(i - 1,j) = minLabel;
                }
                
                if(leftTop > 0){
                    dst.at<uchar>(i - 1,j - 1) = minLabel;
                }
                
                if(rightTop > 0){
                    dst.at<uchar>(i - 1,j + 1) = minLabel;
                }
                
            }
        }
        
    }
    
    
    // 从右向左，从下到上
    for(int i = dst.rows - 1; i > -1; i--){
        for(int j = dst.cols - 1; j > -1; j--){
            if(dst.at<uchar>(i,j) != 0 ){
                
                // 左边和上边是否有 有效区域
                uchar right = 0, leftBottom = 0;
                uchar bottom = 0, rightBottom = 0;
                if(j < dst.cols - 2){
                    right = dst.at<uchar>(i,j + 1);
                }
                if(i < dst.rows - 2){
                    bottom = dst.at<uchar>(i + 1,j);
                }
                
                if(i < dst.rows - 2 && j > 0){
                    leftBottom = dst.at<uchar>(i + 1,j - 1);
                }
                
                if(i < dst.rows - 2 && j < dst.cols - 2){
                    rightBottom = dst.at<uchar>(i + 1,j + 1);
                }
                // 如果都不是有效区域则用++label
                if(right == 0 && bottom == 0 && rightBottom == 0 && leftBottom == 0){
                    
                    continue;
                }
                
                // 四个值中大于0的最小值
                uchar minLabel = 255;
                minLabel = right > 0 ? min(right, minLabel) : minLabel;
                minLabel = bottom > 0 ? min(bottom, minLabel) : minLabel;
                minLabel = leftBottom > 0 ? min(leftBottom, minLabel) : minLabel;
                minLabel = rightBottom > 0 ? min(rightBottom, minLabel) : minLabel;
                minLabel = min(dst.at<uchar>(i,j), minLabel) ;
                dst.at<uchar>(i,j) = minLabel;
                
                if(right > 0){
                    dst.at<uchar>(i,j + 1) = minLabel;
                }
                
                if(bottom > 0){
                    dst.at<uchar>(i + 1,j) = minLabel;
                }
                
                if(leftBottom > 0){
                    dst.at<uchar>(i + 1,j - 1) = minLabel;
                }
                
                if(rightBottom > 0){
                    dst.at<uchar>(i + 1,j + 1) = minLabel;
                }
                
            }
        }
    }
    return dst;
    //    for(int i = 0; i < mat.rows; i++){
    //        for(int j = 0; j < mat.cols; j++){
    //            printf(dst.at<uchar>(i,j)>9?"%d ":"0%d ",dst.at<uchar>(i,j));
    //        }
    //        printf("\n");
    //    }
    
    //    return dst;
}

Mat connectRegion(const Mat& mat, Scalar c){
    // 1. 当该像素的左邻像素和上邻像素为无效值时，给该像素置一个新的label值，label ++;
    // 2. 当该像素的左邻像素或者上邻像素有一个为有效值时，将有效值像素的label赋给该像素的label值；
    // 3. 当该像素的左邻像素和上邻像素都为有效值时，选取其中较小的label值赋给该像素的label值。
    Mat dst(mat.rows, mat.cols, CV_8UC1, Scalar::all(0));
    
    // 如果遇到不是红色的，label++
    // 计算最大的行索引及列索引
    int label = 0;
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            Vec3b color = mat.at<Vec3b>(i, j);
            if(color[0] != c[0] || color[1] != c[1]  || color[2] != c[2] ){
                
                // 左边和上边是否有 有效区域
                uchar left = 0, leftTop = 0;
                uchar top = 0, rightTop = 0;
                if(j > 0){
                    left = dst.at<uchar>(i,j - 1);
                }
                if(i > 0){
                    top = dst.at<uchar>(i - 1,j);
                }
                
                if(i > 0 && j > 0){
                    leftTop = dst.at<uchar>(i - 1,j - 1);
                }
                
                if(i > 0 && j < dst.cols - 1){
                    rightTop = dst.at<uchar>(i - 1,j + 1);
                }
                // 如果都不是有效区域则用++label
                if(left == 0 && top == 0 && leftTop == 0 && rightTop == 0){
                    dst.at<uchar>(i,j) = ++label;
                    continue;
                }
                
                // 四个值中大于0的最小值
                uchar minLabel = 255;
                minLabel = left > 0 ? min(left, minLabel) : minLabel;
                minLabel = top > 0 ? min(top, minLabel) : minLabel;
                minLabel = leftTop > 0 ? min(leftTop, minLabel) : minLabel;
                minLabel = rightTop > 0 ? min(rightTop, minLabel) : minLabel;
                dst.at<uchar>(i,j) = minLabel;
                if(left > 0){
                    dst.at<uchar>(i,j - 1) = minLabel;
                }
                
                if(top > 0){
                    dst.at<uchar>(i - 1,j) = minLabel;
                }
                
                if(leftTop > 0){
                    dst.at<uchar>(i - 1,j - 1) = minLabel;
                }
                
                if(rightTop > 0){
                    dst.at<uchar>(i - 1,j + 1) = minLabel;
                }
                
            }
        }
    }
    
    // 从左向右
    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            if(dst.at<uchar>(i,j) != 0 ){
                
                // 左边和上边是否有 有效区域
                uchar left = 0, leftTop = 0;
                uchar top = 0, rightTop = 0;
                if(j > 0){
                    left = dst.at<uchar>(i,j - 1);
                }
                if(i > 0){
                    top = dst.at<uchar>(i - 1,j);
                }
                
                if(i > 0 && j > 0){
                    leftTop = dst.at<uchar>(i - 1,j - 1);
                }
                
                if(i > 0 && j < dst.cols - 2){
                    rightTop = dst.at<uchar>(i - 1,j + 1);
                }
                // 如果都不是有效区域则用++label
                if(left == 0 && top == 0 && leftTop == 0 && rightTop == 0){
                    
                    continue;
                }
                
                // 四个值中大于0的最小值
                uchar minLabel = 255;
                minLabel = left > 0 ? min(left, minLabel) : minLabel;
                minLabel = top > 0 ? min(top, minLabel) : minLabel;
                minLabel = leftTop > 0 ? min(leftTop, minLabel) : minLabel;
                minLabel = rightTop > 0 ? min(rightTop, minLabel) : minLabel;
                minLabel = min(dst.at<uchar>(i,j), minLabel) ;
                
                dst.at<uchar>(i,j) = minLabel;
                if(left > 0){
                    dst.at<uchar>(i,j - 1) = minLabel;
                }
                
                if(top > 0){
                    dst.at<uchar>(i - 1,j) = minLabel;
                }
                
                if(leftTop > 0){
                    dst.at<uchar>(i - 1,j - 1) = minLabel;
                }
                
                if(rightTop > 0){
                    dst.at<uchar>(i - 1,j + 1) = minLabel;
                }
                
            }
        }
    }
    
    // 从右向左
    for(int i = 0; i < dst.rows; i++){
        for(int j = dst.cols - 1; j > -1; j--){
            if(dst.at<uchar>(i,j) != 0 ){
                
                // 左边和上边是否有 有效区域
                uchar right = 0, leftTop = 0;
                uchar top = 0, rightTop = 0;
                if(j < dst.cols - 2){
                    right = dst.at<uchar>(i,j + 1);
                }
                if(i > 0){
                    top = dst.at<uchar>(i - 1,j);
                }
                
                if(i > 0 && j > 0){
                    leftTop = dst.at<uchar>(i - 1,j - 1);
                }
                
                if(i > 0 && j < dst.cols - 2){
                    rightTop = dst.at<uchar>(i - 1,j + 1);
                }
                // 如果都不是有效区域则用++label
                if(right == 0 && top == 0 && leftTop == 0 && rightTop == 0){
                    
                    continue;
                }
                
                // 四个值中大于0的最小值
                uchar minLabel = 255;
                minLabel = right > 0 ? min(right, minLabel) : minLabel;
                minLabel = top > 0 ? min(top, minLabel) : minLabel;
                minLabel = leftTop > 0 ? min(leftTop, minLabel) : minLabel;
                minLabel = rightTop > 0 ? min(rightTop, minLabel) : minLabel;
                minLabel = min(dst.at<uchar>(i,j), minLabel) ;
                dst.at<uchar>(i,j) = minLabel;
                
                if(right > 0){
                    dst.at<uchar>(i,j + 1) = minLabel;
                }
                
                if(top > 0){
                    dst.at<uchar>(i - 1,j) = minLabel;
                }
                
                if(leftTop > 0){
                    dst.at<uchar>(i - 1,j - 1) = minLabel;
                }
                
                if(rightTop > 0){
                    dst.at<uchar>(i - 1,j + 1) = minLabel;
                }
                
            }
        }
        
    }
    
    
    // 从右向左，从下到上
    for(int i = dst.rows - 1; i > -1; i--){
        for(int j = dst.cols - 1; j > -1; j--){
            if(dst.at<uchar>(i,j) != 0 ){
                
                // 左边和上边是否有 有效区域
                uchar right = 0, leftBottom = 0;
                uchar bottom = 0, rightBottom = 0;
                if(j < dst.cols - 2){
                    right = dst.at<uchar>(i,j + 1);
                }
                if(i < dst.rows - 2){
                    bottom = dst.at<uchar>(i + 1,j);
                }
                
                if(i < dst.rows - 2 && j > 0){
                    leftBottom = dst.at<uchar>(i + 1,j - 1);
                }
                
                if(i < dst.rows - 2 && j < dst.cols - 2){
                    rightBottom = dst.at<uchar>(i + 1,j + 1);
                }
                // 如果都不是有效区域则用++label
                if(right == 0 && bottom == 0 && rightBottom == 0 && leftBottom == 0){
                    
                    continue;
                }
                
                // 四个值中大于0的最小值
                uchar minLabel = 255;
                minLabel = right > 0 ? min(right, minLabel) : minLabel;
                minLabel = bottom > 0 ? min(bottom, minLabel) : minLabel;
                minLabel = leftBottom > 0 ? min(leftBottom, minLabel) : minLabel;
                minLabel = rightBottom > 0 ? min(rightBottom, minLabel) : minLabel;
                minLabel = min(dst.at<uchar>(i,j), minLabel) ;
                dst.at<uchar>(i,j) = minLabel;
                
                if(right > 0){
                    dst.at<uchar>(i,j + 1) = minLabel;
                }
                
                if(bottom > 0){
                    dst.at<uchar>(i + 1,j) = minLabel;
                }
                
                if(leftBottom > 0){
                    dst.at<uchar>(i + 1,j - 1) = minLabel;
                }
                
                if(rightBottom > 0){
                    dst.at<uchar>(i + 1,j + 1) = minLabel;
                }
                
            }
        }
    }
    //        for(int i = 0; i < mat.rows; i++){
    //            for(int j = 0; j < mat.cols; j++){
    //                printf(dst.at<uchar>(i,j)>9?"%d ":"0%d ",dst.at<uchar>(i,j));
    //            }
    //            printf("\n");
    //        }
    return dst;
    
    
    //    return dst;
}

void takeFrontground(const string& srcfilename, const string& dstfilename, double zoom = 0.5){
    Scalar seedColor(46,139,3);
    double seedRadius = 2;
    Mat src,gray,fg,fgMask,edge;
    src = imread(srcfilename);
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            double dis = pow(src.at<Vec3b>(i,j)[0]-seedColor[0],2)+pow(src.at<Vec3b>(i,j)[1]-seedColor[1],2)
            +pow(src.at<Vec3b>(i,j)[2]-seedColor[2],2);
            dis = sqrt(dis);
            if( dis < seedRadius){
                src.at<Vec3b>(i,j)[0] = 0;
                src.at<Vec3b>(i,j)[1] = 0;
                src.at<Vec3b>(i,j)[2] = 255;
            }
        }
    }
    // 计算最大值和最小值
    cvtColor( src, gray, CV_BGR2GRAY);
    
    double t = cv::threshold(gray, gray, 0, 128, CV_THRESH_OTSU);
    
    threshold(gray, gray, t, 128, THRESH_BINARY);
    
    int rMin = src.rows, rMax = 0, cMin = src.cols, cMax = 0;
    
    // 计算最大的行索引及列索引
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            uchar color = gray.at<uchar>(i, j);
            // color 等于 0 时，表示前景
            if(color == 0){
                rMin = min(rMin, i);
                rMax = max(rMax, i);
                
                cMin = min(cMin, j);
                cMax = max(cMax, j);
            }
        }
    }
    
    // 上下，左右各个 冗余部分
    double scale = 0.15;
    cMin = ceil(cMin - (cMax - cMin ) * scale);
    cMin = max(0, cMin);
    cMax = ceil(cMax + (cMax - cMin ) * scale);
    cMax = min(cMax, src.cols);
    
    rMin = ceil(rMin - (rMax - rMin ) * scale);
    rMin = max(0, rMin);
    rMax = ceil(rMax + (rMax - rMin ) * scale);
    rMax = min(rMax, src.rows);
    
    // 图像裁剪
    fg = src(Range(rMin,rMax),Range(cMin,cMax));
    fgMask = gray(Range(rMin,rMax),Range(cMin,cMax));
    cv::resize(fg, fg, Size(fg.cols*zoom,fg.rows*zoom));
    
    
    
    //    imwrite("/Users/huili/Downloads/capture1/1.1.2.png",fg);
    cvtColor( fg, gray, CV_BGR2GRAY);
    t = cv::threshold(gray, gray, 0, 128, CV_THRESH_OTSU);
    printf("t %f\n",t);
    imshow("otsu img", gray);
    threshold(gray, gray, t, 255, THRESH_BINARY);
    imshow("binary img", gray);
    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    //    Canny(fg, edge, 0, 128,5);
    
    edge = genSobel(fg);
    imshow("sobel",edge);
    Mat kernel(3,3,CV_32F,Scalar(-1));
    // 分配像素置
    //  kernel.at<float>(1,1) = 8;
    kernel.at<float>(1,1) = 8.9;
    filter2D(edge,edge,edge.depth(),kernel);
    imshow("sharped",edge);
    
    //    cv::erode(edge, edge, element);
    //     cv::erode(edge, edge, element);
    //     cv::erode(edge, edge, element);
    //
    //    cv::erode(edge, edge, element);
    //
    //    cv::erode(edge, edge, element);
    //
    
    //
    //    cv::dilate(edge, edge, element);
    //    cv::dilate(edge, edge, element);
    //    cv::dilate(edge, edge, element);
    //    cv::dilate(edge, edge, element);
    //    cv::dilate(edge, edge, element);
    //    cv::dilate(edge, edge, element);
    //    cv::erode(edge, edge, element);
    //    cv::erode(edge, edge, element);
    //    cv::erode(edge, edge, element);
    //    cv::erode(edge, edge, element);
    //    cv::erode(edge, edge, element);
    //    cv::erode(edge, edge, element);
    imshow("step 2", edge);
    //    t = cv::threshold(edge, edge, 0, 128, CV_THRESH_OTSU);
    printf("t %f\n",t);
    // 强化边界
    threshold(edge, edge, 20, 255, THRESH_BINARY);
    imshow("step 3", edge);
    // 将边界前景叠加到前景
    for(int i = 0; i < gray.rows; i++){
        for(int j = 0; j < gray.cols; j++){
            uchar color = edge.at<uchar>(i, j);
            // color 等于 0 时，表示前景
            if(color != 0){
                gray.at<uchar>(i, j) = 0;
            }
        }
    }
    imshow("gray append edge", gray);
    cv::erode(gray, gray, element);
    cv::erode(gray, gray, element);
    cv::erode(gray, gray, element);
    
    cv::dilate(gray, gray, element);
    cv::dilate(gray, gray, element);
    cv::dilate(gray, gray, element);
    // 去除白色噪音块
    Mat whiteNoise = connectRegion2(gray,255);
    
    int noiseCount[256];
    // 最大噪音色块
    double maxNoiseChunk = getRegionLabelCount(whiteNoise, noiseCount);
    // 不处理靠近边缘的白色噪音
    for(int i = 0; i < gray.rows; i++){
        for(int j = 0; j < gray.cols; j++){
            uchar noiseLabel = whiteNoise.at<uchar>(i,j);
            if(i == 0 || j == 0 || i == gray.rows - 1 || j == gray.cols - 1){
                noiseCount[noiseLabel] = maxNoiseChunk * 3;
            }
        }
    }
    // 将噪音色填充成灰色
    for(int i = 0; i < gray.rows; i++){
        for(int j = 0; j < gray.cols; j++){
            uchar noiseLabel = whiteNoise.at<uchar>(i,j);
            if(noiseCount[noiseLabel] > 0 && noiseCount[noiseLabel] < maxNoiseChunk){
                gray.at<uchar>(i, j) = 0;
            }
        }
    }
    imshow("gray  erode and dilate ", gray);
    
    // 网格等分
    int n = 120;
    double rNum = (double)fg.rows / n;
    double cNum = (double)fg.cols / n;
    Mat grayDilate;
    // 搜索范围
    
    
    cv::erode(gray, grayDilate, element);
    //    cv::dilate(grayDilate, grayDilate, element);
    //    cv::erode(grayDilate, grayDilate, element);
    //    cv::dilate(grayDilate, grayDilate, element);
    
    //     imshow("step 4", grayDilate);
    
    int length = 0;
    
    // floodfill 种子标定
    //    length = 4;
    //    points[0] = cvPoint(0.1*fg.cols, 0.1*fg.rows);
    //    points[1] = cvPoint(0.9*fg.cols, 0.1*fg.rows);
    //    points[2] = cvPoint(0.9*fg.cols, 0.9*fg.rows);
    //    points[3] = cvPoint(0.1*fg.cols, 0.9*fg.rows);
    for(int i = 0; i < n ; i++){
        for(int j = 0; j < n; j++){
            Point2f point(ceil(i * cNum), ceil(j * rNum));
            
            // 判断该点是否在前景区域内，如果是为红色，否则绿色
            if(point.y > gray.rows - 1 || point.x > gray.cols - 1 || point.y < 0 || point.x < 0){
                continue;
            }
            uchar color = gray.at<uchar>(point.y,point.x);
            uchar colorDilate = grayDilate.at<uchar>(point.y,point.x);
            
            if(color != 0 && colorDilate != 0){
                //                circle( fg, point, 1, color != 0? Scalar(0,0,255): Scalar(0,255,0), -1, 8, 0 );
                points[length].x = i * cNum;
                points[length].y = j * rNum;
                length++;
            }
        }
    }
    imshow("floodfill seeds", fg);
    //    pyrMeanShiftFiltering( fg, fg, 3, 10, 2 );
    
    //    imshow("step 5 mean shift", fg);
    //Mat imgGray;
    //cvtColor(res,imgGray,CV_RGB2GRAY);
    //imshow("res",res);
    Mat fgCopyer;
    fg.copyTo(fgCopyer);
    floodFillPostprocess2( fgCopyer, length,Scalar::all(2) );
    //    for (int i = 0; i < length; i++) {
    //
    //        circle( fg, points[i], 3, Scalar(255,0,255), -1, 8, 0 );
    //    }
    
    // 用蓝色标记前景、背景竞争点，后续实现
    // 水平去除噪音
    imshow("flood filled",fgCopyer);
    for(int i = 0; i < fgCopyer.rows ; i++){
        for(int j = 5; j < fgCopyer.cols - 6; j++){
            
            int bgCount = 0;
            // 搜索左右 8个像素的rgb值，如果有50%以上的颜色为红色则为背景，并将其替换成红色
            for(int offset= -5; offset < 6; offset++){
                // 如果，是本像素则不做处理
                if(offset == 0){continue;}
                Vec3b color = fgCopyer.at<Vec3b>(i, j+offset);
                if(color[0] == 0 && color[1] == 0 && color[2] == 255){
                    bgCount++;
                }
            }
            
            // 临近横向背景色大于50%
            if(bgCount>5){
                fgCopyer.at<Vec3b>(i, j)[0] = 0;
                fgCopyer.at<Vec3b>(i, j)[1] = 0;
                fgCopyer.at<Vec3b>(i, j)[2] = 255;
            }
        }
    }
    
    // 垂直去燥
    for(int i = 5; i < fgCopyer.rows - 6; i++){
        for(int j = 0; j < fgCopyer.cols; j++){
            
            int bgCount = 0;
            // 搜索左右 8个像素的rgb值，如果有50%以上的颜色为红色则为背景，并将其替换成红色
            for(int offset= -5; offset < 6; offset++){
                // 如果，是本像素则不做处理
                if(offset == 0){continue;}
                Vec3b color = fgCopyer.at<Vec3b>(i+offset, j);
                if(color[0] == 0 && color[1] == 0 && color[2] == 255){
                    bgCount++;
                }
            }
            
            // 临近横向背景色大于50%
            if(bgCount>5){
                fgCopyer.at<Vec3b>(i, j)[0] = 0;
                fgCopyer.at<Vec3b>(i, j)[1] = 0;
                fgCopyer.at<Vec3b>(i, j)[2] = 255;
            }
        }
    }
    
    
    // 区域连通去燥
    Mat regions = connectRegion(fgCopyer, Scalar(0,0,255));
    // 获取最大region 的标签
    uchar maxRegionLabel = getMaxRegionLabel(regions);
    // 将非最大region 标记成红色
    for(int i = 0; i < regions.rows; i++){
        for(int j = 0; j < regions.cols; j++){
            uchar label = regions.at<uchar>(i,j);
            if(label != maxRegionLabel ){
                fgCopyer.at<Vec3b>(i, j)[0] = 0;
                fgCopyer.at<Vec3b>(i, j)[1] = 0;
                fgCopyer.at<Vec3b>(i, j)[2] = 128;
            }
        }
    }
    
    // 红色替换成白色
    //    for(int i = 0; i < regions.rows; i++){
    //        for(int j = 0; j < regions.cols; j++){
    //
    //               if( fgCopyer.at<Vec3b>(i, j)[0] == 0
    //                && fgCopyer.at<Vec3b>(i, j)[1] == 0
    //                  && fgCopyer.at<Vec3b>(i, j)[2] == 255){
    //                   fgCopyer.at<Vec3b>(i, j)[0] = 0;
    //                   fgCopyer.at<Vec3b>(i, j)[1] = 255;
    //                   fgCopyer.at<Vec3b>(i, j)[2] = 255;
    //               }
    //        }
    //    }
    
    // 在此水平去噪
    for(int i = 0; i < fgCopyer.rows ; i++){
        for(int j = 10; j < fgCopyer.cols - 11; j++){
            
            int bgCount = 0;
            // 搜索左右 8个像素的rgb值，如果有50%以上的颜色为红色则为背景，并将其替换成红色
            for(int offset= -10; offset < 11; offset++){
                // 如果，是本像素则不做处理
                if(offset == 0){continue;}
                Vec3b color = fgCopyer.at<Vec3b>(i, j+offset);
                if(color[0] == 0 && color[1] == 0 && color[2] == 255){
                    bgCount++;
                }
            }
            
            // 临近横向背景色大于50%
            if(bgCount>10){
                fgCopyer.at<Vec3b>(i, j)[0] = 0;
                fgCopyer.at<Vec3b>(i, j)[1] = 0;
                fgCopyer.at<Vec3b>(i, j)[2] = 255;
            }
        }
    }
    
    imwrite(dstfilename, fgCopyer);
    //    printf("the max region label is %d .\n", maxRegionLabel);
    
    imshow("denoised",fgCopyer);
    imshow("origin", fg);
}


int mainKMeanCluster()
{
    
    IplImage* img = cvLoadImage( "/Users/huili/Downloads/capture4/1.ppm", 1);//三通道图像
    
    int total= img->height*img->width;
    int cluster_num = 2;
    CvMat *row = cvCreateMat( img->height,img->width,CV_32FC3 );
    cvConvert(img,row);//转一下类型！
    CvMat *clusters = cvCreateMat( total, 1, CV_32SC1 );
    cvReshape(row,row,0,total);//修改矩阵的形状,每个数据一行，使row指向修改后的数据，不修改通道数
    cvKMeans2( row, cluster_num, clusters,cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ));
    cvReshape(clusters,clusters,0,img->width);//聚类完的结果再reshape回来方便看
    
    int i=0,j=0;
    CvScalar s;
    IplImage* resImg = cvCreateImage( cvSize(img->width,img->height), 8, 1 );//生成用来显示结果的图像
    s=cvGet2D(img,i,j);
    for(i=0;i<img->height;i++)
    {
        for (j=0;j<img->width;j++)
        {
            if (clusters->data.i[i*img->width+j]==1)
            {
                s.val[0]=255;
                s.val[1]=255;
                s.val[2]=255;
                cvSet2D(resImg,i,j,s);//注意循环顺序
            }
            else
            {
                s.val[0]=0;
                s.val[1]=0;
                s.val[2]=0;
                cvSet2D(resImg,i,j,s);
            }
        }
    }
    cvShowImage( "original", img );
    cvShowImage( "clusters", resImg );
    
    //int key = cvWaitKey(0);
    
    cvReleaseImage(&img);//记得释放内存
    cvReleaseImage (&resImg);
    cvReleaseMat(&row);
    cvReleaseMat(&clusters);
    
    return 0;
}

uchar otsu(Mat src, int rgbIndex){
    double grmax = 0; // 最大方差
    double rt = 0; // r阀值方差
    
    for(int rx = 0; rx < 256; rx++){
        double ur0 = 0; // 前景均值
        int rc0 = 0; // 前景统计
        double wr0 = 0; // 前景比例
        
        double ur1 = 0; // 背景均值
        int rc1 = 0; // 背景统计
        double wr1 = 0; // 背景比例
        
        double ur = 0; // 全局
        double gr = 0; // 方差
        
        for(int i = 0; i < src.rows; i++){
            for(int j = 0; j < src.cols; j++){
                Vec3b color = src.at<Vec3b>(i,j);
                if(color[rgbIndex] > rx){
                    ur0 += color[rgbIndex];
                    rc0 ++;
                }else{
                    ur1 += color[rgbIndex];
                    rc1 ++;
                }
            }
        }
        
        if (rc0 != 0) {
            ur0 = ur0 / rc0;
        }
        
        if (rc1 != 0) {
            ur1 = ur1 / rc1;
        }
        wr0 = (double)rc0/(src.rows*src.cols);
        wr1 = 1 - wr0;
        ur = wr0 * ur0 + wr1 * ur1;
        gr  = wr0 * wr1 * (ur0 - ur1) * (ur0 - ur1);
        
        if(gr > grmax){
            rt = rx;
            grmax = gr;
        }
    }
    
    return rt;
}

int mainForRGBOTSU(){
    Mat src = imread("/Users/huili/Downloads/nail.jpeg");
    cv::resize(src, src, cvSize(src.cols/2, src.rows/2));
    Mat gray(src.rows,src.cols, CV_8UC1);
    Mat seedMat(300,300, CV_8UC3);
    
    
    imshow("rgb ostu", src);
    int index = 0;
    char srcfilename[80];
    Scalar seedColor(46,139,3);
    int r = 0;
    int g = 0;
    int b = 0;
    int model = 0;
    uchar c;
    for(;;){
        
        imshow("rgb ostu", gray);
        // 压缩
        c = waitKey(10);
        if (c == 27) break;
    }
    
    //    46,139,3
    for(r = 3; r < 4; r+=1)
        for(g = 0; g < 139; g+=1)
            for(b = 46; b < 47; b+=1)
                
            {
                for(int i = 0; i < 300; i++){
                    for(int j = 0; j < 300; j++){
                        seedMat.at<Vec3b>(i,j)[0] = b;
                        seedMat.at<Vec3b>(i,j)[1] = g;
                        seedMat.at<Vec3b>(i,j)[2] = r;
                    }
                }
                imshow("seed", seedMat);
                // 归一化处理
                for(int i = 0; i < src.rows; i++){
                    for(int j = 0; j < src.cols; j++){
                        Vec3b color = src.at<Vec3b>(i,j);
                        double seedDis = pow( b,2) + pow(g,2) + pow(r,2);
                        
                        double dis = pow(color[0] - b,2) + pow(color[1] - g,2) + pow(color[2] - r,2);
                        seedDis = sqrt(seedDis);
                        
                        dis = sqrt(dis);
                        //
                        double normalColor = 255 * (dis / seedDis);
                        
                        gray.at<uchar>(i,j) = normalColor;
                    }
                }
                
                double t = cv::threshold(gray, gray, 0, 255, CV_THRESH_OTSU);
                
                threshold(gray, gray, t, 255, THRESH_BINARY);
                
                
                imshow("rgb ostu", gray);
                
                // 压缩
                c = waitKey(10);
                if (c == 27) break;
            }
    
    for(;;){
        // 压缩
        c = waitKey(10);
        if (c == 27) break;
    }
    
    return 0;
}

Mat contrastStretch(Mat srcImage)
{
    Mat resultImage = srcImage.clone();//"=";"clone()";"copyTo"三种拷贝方式，前者是浅拷贝，后两者是深拷贝。
    int nRows = resultImage.rows;
    int nCols = resultImage.cols;
    //判断图像的连续性
    if (resultImage.isContinuous())
    {
        nCols = nCols*nRows;
        nRows = 1;
    }
    //图像指针操作
    uchar *pDataMat;
    int pixMax = 0, pixMin = 255;
    //计算图像的最大最小值
    for (int j = 0; j < nRows; j++)
    {
        pDataMat = resultImage.ptr<uchar>(j);//ptr<>()得到的是一行指针
        for (int i = 0; i < nCols; i++)
        {
            if (pDataMat[i] > pixMax)
                pixMax = pDataMat[i];
            if (pDataMat[i] < pixMin)
                pixMin = pDataMat[i];
        }
    }
    //对比度拉伸映射
    for (int j = 0; j < nRows; j++)
    {
        pDataMat = resultImage.ptr<uchar>(j);
        for (int i = 0; i < nCols; i++)
        {
            pDataMat[i] = (pDataMat[i] - pixMin) * 255 / (pixMax - pixMin);
        }
    }
    return resultImage;
}


// Mser车牌目标检测
std::vector<cv::Rect> mserGetPlate(cv::Mat srcImage)
{
    
    // HSV空间转换
    cv::Mat gray, gray_neg;
    cv::Mat hsi;
    cv::cvtColor(srcImage, hsi, CV_BGR2HSV);
    // 通道分离
    std::vector<cv::Mat> channels;
    cv::split(hsi, channels);
    // 提取h通道
    gray = channels[1];
    // 灰度转换
    cv::cvtColor(srcImage, gray, CV_BGR2GRAY);
    // 取反值灰度
    gray_neg = 255 - gray;
    std::vector<std::vector<cv::Point> > regContours;
    std::vector<std::vector<cv::Point> > charContours;
    
    // 创建MSER对象
 //   MSER mser1(  5, 10, cvRound(0.1*(srcImage.cols)*0.15*(srcImage.rows)),0.25,0.2);
 //   MSER mser2(  5, 10, cvRound(0.1*(srcImage.cols)*0.15*(srcImage.rows)),0.25,0.2);
    
    
    std::vector<cv::Rect> bboxes1;
    std::vector<cv::Rect> bboxes2;
    // MSER+ 检测
   // mser1(gray, regContours, Mat());
    // MSER-操作
   // mser2(gray_neg, charContours, Mat());
    
    
    cv::Mat mserMapMat =cv::Mat::zeros(srcImage.size(), CV_8UC1);
    cv::Mat mserNegMapMat =cv::Mat::zeros(srcImage.size(), CV_8UC1);
    
    for (int i = (int)regContours.size() - 1; i >= 0; i--)
    {
        // 根据检测区域点生成mser+结果
        const std::vector<cv::Point>& r = regContours[i];
        for (int j = 0; j < (int)r.size(); j++)
        {
            cv::Point pt = r[j];
            mserMapMat.at<unsigned char>(pt) = 255;
        }
    }
    // MSER- 检测
    for (int i = (int)charContours.size() - 1; i >= 0; i--)
    {
        // 根据检测区域点生成mser-结果
        const std::vector<cv::Point>& r = charContours[i];
        for (int j = 0; j < (int)r.size(); j++)
        {
            cv::Point pt = r[j];
            mserNegMapMat.at<unsigned char>(pt) = 255;
        }
    }
    // mser结果输出
    cv::Mat mserResMat;
    // mser+与mser-位与操作
    mserResMat = mserMapMat & mserNegMapMat;
    //    cv::imshow("mserMapMat", mserMapMat);
    //    cv::imshow("mserNegMapMat", mserNegMapMat);
    //    cv::imshow("mserResMat", mserResMat);
    // 闭操作连接缝隙
    cv::Mat mserClosedMat;
    cv::morphologyEx(mserResMat, mserClosedMat,
                     cv::MORPH_CLOSE, cv::Mat::ones(1, 20, CV_8UC1));
    //    cv::imshow("mserClosedMat", mserClosedMat);
    // 寻找外部轮廓
    std::vector<std::vector<cv::Point> > plate_contours;
    cv::findContours(mserClosedMat, plate_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    // 候选车牌区域判断输出
    std::vector<cv::Rect> candidates;
    for (size_t i = 0; i != plate_contours.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
        // 宽高比例
        //            double wh_ratio = rect.width / double(rect.height);
        // 不符合尺寸条件判断
        //            if (rect.height > 20 && wh_ratio > 4 && wh_ratio < 7)
        candidates.push_back(rect);
    }
    return  candidates;
}

int main1(){
    
    string str="/Users/yangxuewu/Downloads/11.png.......";

    // 扫描左右2*w个像素，其均值为阀值
    Mat src = imread(str);
    cv::resize(src, src, cvSize(src.cols/4, src.rows/4));
    Mat gray(src.rows, src.cols, CV_8UC1);
    cv::medianBlur(gray, gray, 5);
    Mat dst(src.rows, src.cols, CV_8UC1);
    cvtColor( src, gray, CV_BGR2GRAY);
    
    
    int w = 10, s = 60, t1 = 128;
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            // 计算最大值M，最小值N, 均值T
            double t;
            
            // 行索引，列索引
            int rStart = i - 0;
            int rEnd = i + 0;
            
            int cStart = j - w;
            int cEnd = j + w;
            
            rStart = max(0,rStart);
            rEnd = min(gray.rows - 1, rEnd);
            cStart = max(0,cStart);
            cEnd = min(gray.cols - 1, cEnd);
            
            double total = 0;
            int count = 0;
            
            for (int ri = rStart; ri <=  rEnd; ri++) {
                for (int ci = cStart; ci <= cEnd; ci++) {
                    uchar value = gray.at<uchar>(ri,ci);
                    count++;
                    total += value;
                }
            }
            
            t = total / count;
            
            dst.at<uchar>(i,j) = gray.at<uchar>(i,j) > t ? 255 : 0;
            
        }
    }
    
    imshow("wallen", dst);
    
    waitKey(0);
    return 0;
}

int mainForBernsen(){
    //    这个算法的中心思想是：设当前像素为P，计算以P为中心的大小为(2w+1)*(2w+1)窗口内的所有像素的最大值M与最小值N，两者的均值T,
    //       if(M-N）> S
    //       则当前点P的阈值为T。
    //       else
    //       当前窗口所在区域的灰度级差别较小，那么窗口在目标区或在背景区，若T>T'则当前点灰度值为255，否则，当前点灰度值为0.
    //       S作者最初设为15, T'设为（255+0）/2=128。
    Mat src = imread("/Users/huili/Downloads/capture4/778.ppm");
    cv::resize(src, src, cvSize(src.cols/4, src.rows/4));
    Mat gray(src.rows, src.cols, CV_8UC1);
    Mat dst(src.rows, src.cols, CV_8UC1);
    cvtColor( src, gray, CV_BGR2GRAY);
    int w = 10, s = 60, t1 = 128;
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            // 计算最大值M，最小值N, 均值T
            uchar m = 0, n = 255, t;
            
            // 行索引，列索引
            int rStart = i - w;
            int rEnd = i + w;
            
            int cStart = j - w;
            int cEnd = j + w;
            
            rStart = max(0,rStart);
            rEnd = min(gray.rows - 1, rEnd);
            cStart = max(0,cStart);
            cEnd = min(gray.cols - 1, cEnd);
            
            for (int ri = rStart; ri <=  rEnd; ri++) {
                for (int ci = cStart; ci <= cEnd; ci++) {
                    uchar value = gray.at<uchar>(ri,ci);
                    n = min(n, value);
                    m = max(m, value);
                }
            }
            
            t = (m - n) / 2;
            if(m - n > s){
                dst.at<uchar>(i,j) = dst.at<uchar>(i,j) > t ? 0 : 255;
            }else{
                dst.at<uchar>(i,j) = t > t1? 255 : 0;
            }
            
        }
    }
    
    imshow("bernsen", dst);
    
    waitKey(0);
    return 0;
}

int mainForMser()
{
    char c;
    //    VideoCapture inputVideo(0);    //0为外部摄像头的ID，1为笔记本内置摄像头的ID
    Mat srcImage;
    vector<Point2f> centers;
    double scale = 0.5; // 压缩比例
    
    //    for(;;)
    //    {
    //        inputVideo >> srcImage;
    srcImage = imread("/Users/huili/Downloads/nail-photo/2.jpg");
    cv::resize(srcImage, srcImage, cvSize(srcImage.cols/4,srcImage.rows/4));
    cv::medianBlur(srcImage, srcImage, 3);
    // 候选车牌区域检测
    std::vector<cv::Rect> candidates;
    candidates = mserGetPlate(srcImage);
    // 车牌区域显示
    for (int i = 0; i < candidates.size(); ++i)
    {
        rectangle(srcImage, candidates[i], Scalar(0,0,255));
        
    }
    cv::imshow("MSER", srcImage);
    cv::waitKey(0);
    //    }
    return 0;
}

int mainTexDetect(){
    
    Mat src = imread("/Users/huili/Downloads/nail-photo/15.JPG");
    cv::resize(src, src, cvSize(src.cols/4,src.rows/4));
    Mat gray(src.rows, src.cols, CV_8UC1);
    cvtColor( src, gray, CV_BGR2GRAY);
    // 统计颜色概率
    int countChunk = 64;
    int rgb[countChunk][countChunk][countChunk];
    int maxIndexs[3];
    int minIndexs[3];
    int maxCount = 0;
    int minCount = src.rows * src.cols;
    for(int i = 0; i < countChunk; i++){
        for(int j = 0; j < countChunk; j++){
            for(int k = 0; k < countChunk; k++){
                rgb[i][j][k]=0;
            }
        }
    }
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            Vec3b color = src.at<Vec3b>(i,j);
            int rIndex = ceil((double)color[2] / countChunk);
            int gIndex = ceil((double)color[1] / countChunk);
            int bIndex = ceil((double)color[0] / countChunk);
            rgb[rIndex][gIndex][bIndex]++;
        }
    }
    
    for(int i = 0; i < countChunk; i++){
        for(int j = 0; j < countChunk; j++){
            for(int k = 0; k < countChunk; k++){
                if(maxCount < rgb[i][j][k]){
                    maxIndexs[0] = i;
                    maxIndexs[1] = j;
                    maxIndexs[2] = k;
                }
                if(maxCount > rgb[i][j][k]){
                    minIndexs[0] = i;
                    minIndexs[1] = j;
                    minIndexs[2] = k;
                }
                maxCount = max(rgb[i][j][k],maxCount);
                minCount = min(rgb[i][j][k],minCount);
            }
        }
    }
    
    
    for(int i = 0; i < countChunk; i++){
        for(int j = 0; j < countChunk; j++){
            for(int k = 0; k < countChunk; k++){
                if(rgb[i][j][k] != 0){
                    printf("%d \n",rgb[i][j][k]);
                }
            }
        }
    }
    // 将概率最高和最低的颜色替换成黑色
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            Vec3b color = src.at<Vec3b>(i,j);
            int rIndex = ceil((double)color[2] / countChunk);
            int gIndex = ceil((double)color[1] / countChunk);
            int bIndex = ceil((double)color[0] / countChunk);
            if(rIndex == maxIndexs[0] && gIndex == maxIndexs[1] && bIndex == maxIndexs[2]){
                src.at<Vec3b>(i,j)[0] = 0;
                src.at<Vec3b>(i,j)[1] = 0;
                src.at<Vec3b>(i,j)[2] = 255;
            }
            
        }
    }
    
    // 计算反向亮度
    
    double light = 0;
    int lightCount = 0;
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            if(src.at<Vec3b>(i,j)[0] != 0
               &&src.at<Vec3b>(i,j)[1] != 0
               &&src.at<Vec3b>(i,j)[2] != 255){
                
                lightCount ++ ;
                Vec3b color = src.at<Vec3b>(i,j);
                double dis = pow(color[0]-5,2)+ pow(color[1]-5,2)+ pow(color[2]-5,2);
                dis = sqrt(dis);
                light += dis;
            }
        }
    }
    
    light = light/lightCount;
    
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            if(src.at<Vec3b>(i,j)[0] != 0
               &&src.at<Vec3b>(i,j)[1] != 0
               &&src.at<Vec3b>(i,j)[2] != 255){
                Vec3b color = src.at<Vec3b>(i,j);
                double dis = pow(color[0]-5,2)+ pow(color[1]-5,2)+ pow(color[2]-5,2);
                dis = sqrt(dis);
                gray.at<uchar>(i,j) = dis > 5? 255: 0;
            }
        }
    }
    printf("light %f\n",light);
    imshow("normalized img", src);
    imshow("light img", gray);
    uchar c;
    for(;;){
        // 压缩
        c = waitKey(10);
        if (c == 27) break;
    }
    //    cv::resize(src, src, cvSize(src.cols, src.rows));
    //    Mat gray, temp;
    //    cvtColor( src, gray, CV_BGR2GRAY);
    //    double t = cv::threshold(gray, gray, 0, 128, CV_THRESH_OTSU);
    //
    //    threshold(gray, gray, t, 255, THRESH_BINARY);
    //
    ////    for(int t = 0; t < 256; t++){
    ////        threshold(gray, temp, t, 255, THRESH_BINARY);
    ////        char srcfilename[80];
    ////        sprintf(srcfilename, "/Users/huili/Downloads/capture4/otsu-%d.jpg", t);
    ////        imwrite(srcfilename, temp);
    ////
    ////    }
    //    imshow("t", gray);
    //    waitKey(0);
    return 0;
}

int mainForTake(){
    
    //    int n = 20;
    //    Mat src = imread("/Users/huili/Downloads/capture2/766.ppm");
    //    char dstfilename[80];
    //    double cNum = src.cols / n;
    //    for (int i = 0; i < n; i++) {
    //        Mat part = src(Range(0, src.rows - 1), Range((i) * cNum, (i+1) * cNum));
    //        sprintf(dstfilename, "/Users/huili/Downloads/capture2/patterns/%d.png", i);
    //        imwrite(dstfilename, part);
    //    }
    //
    //    takeFrontground( "/Users/huili/Downloads/capture2/patterns/2.png",  "/Users/huili/Downloads/capture2/patterns/a-2.png");
    for(int i=0; i< 1; i++){
        char srcfilename[80];
        char dstfilename[80];
        sprintf(srcfilename, "/Users/huili/Downloads/capture4/%d.ppm", i);
        sprintf(dstfilename, "/Users/huili/Downloads/capture4/take4/b-%d.jpg", i);
        takeFrontground(srcfilename, dstfilename, 0.25);
        printf("%d take off background completed.\n" , i);
    }
    
    waitKey(0);
    return 0;
}

int mainForCanny(){
    string str=" /Users/yangxuewu/Downloads/1.jpg";
    Mat src,dst;
    src = imread(str);
    Canny(src,dst,1,128,3);

    imwrite("/Users/huili/Downloads/capture1/1.ppm", dst);
    waitKey(0);
    return 0;
}

int mainForThreshold2()
{
    char filename[80];
    char dstfn[80];
    int w = 0,h = 0;
    Mat src;
    Mat gray;//临时变量和目标图的定义
    
    for(int i=0; i< 948; i++){
        sprintf( filename, "/Users/huili/Downloads/capture1/%d.ppm", i);
        sprintf(dstfn, "/Users/huili/Downloads/capture/binary/%d.png", i);
        //VideoCapture inputVideo(0);    //0为外部摄像头的ID，1为笔记本内置摄像头的ID
        src = imread(filename);
        // 压缩
        // cv::resize(src, src, cvSize(src.cols*0.5, src.rows*0.5));
        
        cvtColor( src, gray, CV_BGR2GRAY);
        double t = cv::threshold(gray, gray, 0, 128, CV_THRESH_OTSU);
        printf("t %f\n",t);
        threshold(gray, gray, t, 255, THRESH_BINARY);
        
        imwrite(dstfn, gray);
        printf("%d completed.\n", i);
        printf(" max width %d  max height%d.\n",  w, h);
        
    }
    
    printf(" max width %d  max height%d.\n",  w, h);
    
    
    return 0;
}

int mainForFindCirclies()
{
    char c;
    VideoCapture inputVideo(0);    //0为外部摄像头的ID，1为笔记本内置摄像头的ID
    Mat src;
    vector<Point2f> centers;
    double scale = 0.5; // 压缩比例
    
    for(;;)
    {
        inputVideo >> src;
        // 压缩
        // cv::resize(src, src, cvSize(src.cols*0.5, src.rows*0.5));
        c = waitKey(10);
        if (c == 27) break;
        // 霍夫曼直线提取
        Mat gray;//临时变量和目标图的定义
        cvtColor( src, gray, CV_BGR2GRAY );
        // 计算二值化最佳阀值
        double t = cv::threshold(gray, gray, 0, 128, CV_THRESH_OTSU);
        // 二值化处理
        threshold(gray, gray, t, 255, THRESH_BINARY);
        
        // cv::findCirclesGrid(gray, cvSize(23,16), centers);
        //在原图中画出圆心和圆
        // for( size_t i = 0; i < centers.size(); i++ )
        // {
        //提取出圆心坐标
        //提取出圆半径
        //     int radius = 3;
        //圆心
        //    circle( src, centers[i], 3, Scalar(0,255,0), -1, 8, 0 );
        //圆
        //     circle( src, centers[i], radius, Scalar(0,0,255), 1, 8, 0 );
        //}
        //高斯模糊平滑
        GaussianBlur( gray, gray, Size(3, 3), 2, 2 );
        //medianBlur(gray, gray, 3);
        vector<Vec3f> circles;
        //霍夫变换
        HoughCircles( gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows/20, 125, 18, 10, 20 );
        
        //在原图中画出圆心和圆
        for( size_t i = 0; i < circles.size(); i++ )
        {
            //提取出圆心坐标
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            //提取出圆半径
            int radius = cvRound(circles[i][2]);
            //圆心
            circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
            //圆
            circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
        }
        
        
        
        imshow("input",src);
        
    }
    return 0;
}

static void help(char** argv)
{
    cout << "\nDemonstrate mean-shift based color segmentation in spatial pyramid.\n"
    << "Call:\n   " << argv[0] << " image\n"
    << "This program allows you to set the spatial and color radius\n"
    << "of the mean shift window as well as the number of pyramid reduction levels explored\n"
    << endl;
}



string winName = "meanshift";
int spatialRad, colorRad, maxPyrLevel;
Mat img, res;

static void meanShiftSegmentation( int, void* )
{
    cout << "spatialRad=" << spatialRad << "; "
    << "colorRad=" << colorRad << "; "
    << "maxPyrLevel=" << maxPyrLevel << endl;
    pyrMeanShiftFiltering( img, res, spatialRad, colorRad, maxPyrLevel );
    //Mat imgGray;
    //cvtColor(res,imgGray,CV_RGB2GRAY);
    //imshow("res",res);
    floodFillPostprocess( res, Scalar::all(2) );
    imshow( winName, res );
}



int mainMeanshift(int argc, char** argv)
{
    img = imread("/Users/huili/Downloads/capture/298.ppm");
    //img = imread("pic2.png");
    
    
    if( img.empty() )
        return -1;
    
    spatialRad = 10;
    colorRad = 10;
    maxPyrLevel = 1;
    
    namedWindow( winName, WINDOW_AUTOSIZE );
    //imshow("img",img);
    
    
    createTrackbar( "spatialRad", winName, &spatialRad, 80, meanShiftSegmentation );
    createTrackbar( "colorRad", winName, &colorRad, 60, meanShiftSegmentation );
    createTrackbar( "maxPyrLevel", winName, &maxPyrLevel, 5, meanShiftSegmentation );
    
    meanShiftSegmentation(0, 0);
    //floodFillPostprocess( img, Scalar::all(2) );
    //imshow("img2",img);
    waitKey();
    return 0;
}

string filename;
Mat image;
enum{NOT_SET = 0, IN_PROCESS = 1, SET = 2};
uchar rectState;
Rect rect;
Mat mask;
const Scalar GREEN = Scalar(0,255,0);
Mat bgdModel, fgdModel;
Mat blended;
void setRectInMask(){
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image.cols-rect.x);
    rect.height = min(rect.height, image.rows-rect.y);
    
}

static void getBinMask( const Mat& comMask, Mat& binMask ){
    binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

static void clip(){
    int erosion_type = 0;
    int erosion_elem = 1;
    int erosion_size = 0;
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    (mask(rect)).setTo( Scalar(GC_PR_FGD));
    
    for (int i=546; i<547; i++) {
        
        Mat eroded;
        Mat res;
        Mat binMask;
        Mat dilate,dilate2;
        
        char filename[80];
        char wfilename[80];
        sprintf(filename,"/Users/huili/Downloads/capture/%d.ppm", i);
        sprintf(wfilename,"/Users/huili/Downloads/capture/cutted2/%d.png", i);
        image = imread( filename, 1 );
        
        grabCut(image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
        
        getBinMask( mask, binMask );
        // image.copyTo(res, binMask );
        // 腐蚀
        //Mat element = getStructuringElement( erosion_type,
        //                                    Size( 2*erosion_size + 1, 2*erosion_size+1 ),
        //                                   Point( erosion_size, erosion_size ) );
        //cv::erode(res, eroded, element);
        // 膨胀
        // cv::dilate(eroded, dilate, element);
        //cv::dilate(dilate, dilate2, element);
        int rMin = INT_MAX;
        int rMax = -1;
        int cMin = INT_MAX;
        int cMax = -1;
        for(int r = 0; r < binMask.rows; r++){
            for(int c = 0; c < binMask.cols; c++){
                binMask.at<uchar>(r,c) = binMask.at<uchar>(r,c)==GC_BGD||binMask.at<uchar>(r,c)==GC_PR_BGD? 0:255;
                // 如果，是前景时，提取最小列索引和最大咧索引
                if( binMask.at<uchar>(r,c)!=GC_BGD&&binMask.at<uchar>(r,c)!=GC_PR_BGD){
                    rMin = min(rMin, r);
                    rMax = max(rMax, r);
                    
                    cMin = min(cMin, c);
                    cMax = max(cMax, c);
                }
            }
        }
        
        // 创建一张w=cMax-cMin，h=rMax-rMin的图片
        int rows = rMax-rMin;
        int cols = cMax-cMin;
        printf("index %d\n", i);
        Mat foregound(rows, cols, CV_8UC4);
        for(int r = rMin; r < rMax; r++){
            for(int c = cMin; c < cMax; c++){
                
                // 如果，是前景时，提取最小列索引和最大咧索引
                if( binMask.at<uchar>(r,c)!=GC_BGD&&binMask.at<uchar>(r,c)!=GC_PR_BGD){
                    //printf("r-rMin %d, c-cMin %d\n",r-rMin,c-cMin);
                    if(r-rMin-1<0||c-cMin-1<0){continue;}
                    foregound.at<Vec4b>(r-rMin-1,c-cMin-1)[0] = image.at<Vec3b>(r,c)[0];
                    foregound.at<Vec4b>(r-rMin-1,c-cMin-1)[1] = image.at<Vec3b>(r,c)[1];
                    foregound.at<Vec4b>(r-rMin-1,c-cMin-1)[2] = image.at<Vec3b>(r,c)[2];
                    foregound.at<Vec4b>(r-rMin-1,c-cMin-1)[3] = 255;
                }
            }
        }
        imwrite(wfilename, foregound);
        
        //imshow("binMask", binMask);
        imshow("foregound", foregound);
        //imshow("res", res);
    }
}

void on_mouse( int event, int x, int y, int flags, void* )
{
    switch( event ){
        case CV_EVENT_LBUTTONDOWN:
            if( rectState == NOT_SET){
                rectState = IN_PROCESS;
                rect = Rect( x, y, 1, 1 );
            }
            break;
        case CV_EVENT_LBUTTONUP:
            if( rectState == IN_PROCESS ){
                rect = Rect( Point(rect.x, rect.y), Point(x,y) );
                rectState = SET;
                clip();
                
            }
            break;
        case CV_EVENT_MOUSEMOVE:
            if( rectState == IN_PROCESS ){
                rect = Rect( Point(rect.x, rect.y), Point(x,y) );
                image = imread( filename, 1 );
                rectangle(image, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);
                imshow(winName, image);
            }
            break;
    }
}

void on_mouse_rect( int event, int x, int y, int flags, void* )
{
    switch( event ){
        case CV_EVENT_LBUTTONDOWN:
            if( rectState == NOT_SET){
                rectState = IN_PROCESS;
                rect = Rect( x, y, 1, 1 );
            }
            break;
        case CV_EVENT_LBUTTONUP:
            if( rectState == IN_PROCESS ){
                rect = Rect( Point(rect.x, rect.y), Point(x,y) );
                rectState = SET;
            }
            break;
        case CV_EVENT_MOUSEMOVE:
            if( rectState == IN_PROCESS ){
                rect = Rect( Point(rect.x, rect.y), Point(x,y) );
                Mat tmp = blended.clone();
                rectangle(tmp, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);
                imshow(winName, tmp);
            }
            break;
    }
}


void addWeighted2(Mat src1,  Mat src2,
                  double gamma, Mat dst){
    // 按照比重计算rgba
    for (int i = 0; i < dst.rows; i++) {
        for(int j = 0; j < dst.cols; j++){
            dst.at<Vec4b>(i,j)[0] = (i >= src1.rows|| j >= src1.cols) ? floor(dst.at<Vec4b>(i,j)[0]*gamma): floor(dst.at<Vec4b>(i,j)[0]*gamma + src1.at<Vec3b>(i,j)[0]*(1-gamma));
            dst.at<Vec4b>(i,j)[1] = (i >= src1.rows|| j >= src1.cols) ? floor(dst.at<Vec4b>(i,j)[1]*gamma):floor(dst.at<Vec4b>(i,j)[1]*gamma + src1.at<Vec3b>(i,j)[1]*(1-gamma));
            dst.at<Vec4b>(i,j)[2] = (i >= src1.rows|| j >= src1.cols) ? floor(dst.at<Vec4b>(i,j)[2]*gamma):floor(dst.at<Vec4b>(i,j)[2]*gamma + src1.at<Vec3b>(i,j)[2]*(1-gamma));
            dst.at<Vec4b>(i,j)[3] = 255;
        }
    }
    
}


void addWeighted3(Mat src1,
                  double gamma, Mat dst){
    // 按照比重计算rgba
    for (int i = 0; i < dst.rows; i++) {
        for(int j = 0; j < dst.cols; j++){
            
            int offsetR = (i >= src1.rows|| j >= src1.cols)?0:abs(dst.at<Vec4b>(i,j)[0] - src1.at<Vec3b>(i,j)[0]);
            int offsetG = (i >= src1.rows|| j >= src1.cols)?0:abs(dst.at<Vec4b>(i,j)[1] - src1.at<Vec3b>(i,j)[1]);
            int offsetB = (i >= src1.rows|| j >= src1.cols)?0:abs(dst.at<Vec4b>(i,j)[2] - src1.at<Vec3b>(i,j)[2]*(1-gamma));
            
            int dstR =  (i >= src1.rows|| j >= src1.cols) ? floor(dst.at<Vec4b>(i,j)[0]): floor(min(dst.at<Vec4b>(i,j)[0],src1.at<Vec3b>(i,j)[0])+offsetR*gamma);
            int dstG =  (i >= src1.rows|| j >= src1.cols) ? floor(dst.at<Vec4b>(i,j)[1]): floor(min(dst.at<Vec4b>(i,j)[1],src1.at<Vec3b>(i,j)[1])+offsetG*gamma);
            int dstB =  (i >= src1.rows|| j >= src1.cols) ? floor(dst.at<Vec4b>(i,j)[2]): floor(min(dst.at<Vec4b>(i,j)[2],src1.at<Vec3b>(i,j)[2])+offsetB*gamma);
            
            dst.at<Vec4b>(i,j)[0] = dstR;
            dst.at<Vec4b>(i,j)[1] = dstG;
            dst.at<Vec4b>(i,j)[2] = dstB;
            dst.at<Vec4b>(i,j)[3] = 255;
        }
    }
    
}

Mat blendImg(Mat image1, Mat image2){
    double alpha = 0.8;
    Mat image;
    //image1=imread("/Users/huili/Downloads/capture/0.ppm");
    //image2=imread("/Users/huili/Downloads/capture/500.ppm");
    addWeighted(image1,alpha,image2,1-alpha,0,image);
    return image;
}

int mainToolClip(int argc, char* argv[]){
    char filename[80];
    char dstfn[80];
    int w = 0,h = 0;
    Mat img;
    for(int i = 0; i < 1145; i++){
        sprintf( filename, "/Users/huili/Downloads/capture/cutted3/%d.png", i);
        img = imread(filename);
        w = max(img.cols, w);
        h = max(img.rows, h);
    }
    
    printf("max width: %d, min height:%d\n", w, h);
    
    for(int i = 0; i < 1145; i++){
        sprintf( filename, "/Users/huili/Downloads/capture/cutted3/%d.png", i);
        sprintf(dstfn, "/Users/huili/Downloads/capture/cutted4/%d.png", i);
        img = imread(filename);
        Mat dst(h,img.cols,CV_8UC3);
        // 垂直填充
        if(img.rows == h){
            imwrite(dstfn, img);
            continue;
        }
        
        int riOffset = (h - img.rows) / 2;
        printf("%d riOffset.\n", riOffset);
        for (int ri = 0; ri < img.rows; ri++) {
            for (int ci = 0; ci < img.cols; ci++) {
                // Vec3b srcRaw =   img.at<Vec3b>(ri,ci);
                // Vec4b dstRaw =   dst.at<Vec4b>(ri+riOffset,ci);
                dst.at<Vec3b>(ri+riOffset,ci)[0] = img.at<Vec3b>(ri,ci)[0];
                dst.at<Vec3b>(ri+riOffset,ci)[1] = img.at<Vec3b>(ri,ci)[1];
                dst.at<Vec3b>(ri+riOffset,ci)[2] = img.at<Vec3b>(ri,ci)[2];
            }
        }
        imwrite(dstfn, dst);
        printf("%d completed.\n", i);
    }
    
    waitKey(0);
    
    return 0;
    
}

int startIndex = 0;
int endEndIndex = 48;
int mainForcut(int argc, char* argv[]){
    char filename[80];
reload:
    sprintf( filename, "/Users/huili/Downloads/Green/%d.png", startIndex);
    
    blended =imread(filename);
    for (int i = startIndex + 1; i < endEndIndex && i < 48; i++) {
        if((i%1==0)){
            
            sprintf( filename, "/Users/huili/Downloads/Green/%d.png", i);
            blended = blendImg(blended, imread(filename));
        }
    }
    rectState = NOT_SET;
    //imwrite("/Users/huili/Downloads/capture/cutted/merged.png", dst);
    if(startIndex < 48){
        imshow(winName, blended);
    }
    
    
    setMouseCallback(winName, on_mouse_rect, 0);
    
    for(;;)
    {
        
        int c = waitKey(0);
        if( (c & 255) == 27 )
        {
            cout << "Exiting ...\n";
            break;
        }
        
        switch((char)c)
        {
                
            case '=':
                startIndex += 50;
                endEndIndex = startIndex+50;
                startIndex = min(startIndex,1145);
                endEndIndex = min(endEndIndex,1145);
                goto reload;
                break;
            case '-':
                startIndex -= 50;
                endEndIndex = startIndex+50;
                startIndex = max(startIndex,0);
                endEndIndex = max(endEndIndex,0);
                goto reload;
                break;
            case 'c':
                // cut images
                for (int i = startIndex; i < endEndIndex && i < 1145; i++) {
                    
                    char srcfn[80];
                    char dstfn[80];
                    sprintf( srcfn, "/Users/huili/Downloads/Green/%d.png", i);
                    sprintf( dstfn, "/Users/huili/Downloads/Green/cutted/%d.png", i);
                    Mat src = imread(srcfn);
                    Mat dst;
                    src(rect).copyTo(dst);
                    imwrite(dstfn, dst);
                    
                }
                goto reload;
                break;
                
            case 'r':
                startIndex = 0;
                endEndIndex = 50;
                goto reload;
                break;
            case 'x':
                
                goto reload;
                break;
                
        }
    }
    return 0;
    
}

int mainGrab(int argc, char* argv[]){
    filename = "/Users/huili/Downloads/capture/546.ppm";
    image = imread( filename, 1 );
    
    
    imshow(winName, image);
    mask.create(image.size(), CV_8UC1);
    rectState = NOT_SET;
    mask.setTo(GC_BGD);
    
    setMouseCallback(winName, on_mouse, 0);
    waitKey(0);
    
    return 0;
}



static void help()
{
    cout << "\nThis program demonstrated the floodFill() function\n"
    "Call:\n"
    "./ffilldemo [image_name -- Default: fruits.jpg]\n" << endl;
    
    cout << "Hot keys: \n"
    "\tESC - quit the program\n"
    "\tc - switch color/grayscale mode\n"
    "\tm - switch mask mode\n"
    "\tr - restore the original image\n"
    "\ts - use null-range floodfill\n"
    "\tf - use gradient floodfill with fixed(absolute) range\n"
    "\tg - use gradient floodfill with floating(relative) range\n"
    "\t4 - use 4-connectivity mode\n"
    "\t8 - use 8-connectivity mode\n" << endl;
}

Mat image0, gray;
int ffillMode = 1;
int loDiff = 20, upDiff = 20;
int connectivity = 4;
int isColor = true;
bool useMask = false;
int newMaskVal = 255;

static void onMouse( int event, int x, int y, int, void* )
{
    if( event != CV_EVENT_LBUTTONDOWN )
        return;
    
    Point seed = Point(x,y);
    int lo = ffillMode == 0 ? 0 : loDiff;
    int up = ffillMode == 0 ? 0 : upDiff;
    int flags = connectivity + (newMaskVal << 8) +
    (ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
    int b = (unsigned)theRNG() & 255;
    int g = (unsigned)theRNG() & 255;
    int r = (unsigned)theRNG() & 255;
    Rect ccomp;
    
    Scalar newVal = isColor ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
    Mat dst = isColor ? image : gray;
    int area;
    
    if( useMask )
    {
        threshold(mask, mask, 1, 128, CV_THRESH_BINARY);
        area = floodFill(dst, mask, seed, newVal, &ccomp, Scalar(lo, lo, lo),
                         Scalar(up, up, up), flags);
        imshow( "mask", mask );
    }
    else
    {
        area = floodFill(dst, seed, newVal, &ccomp, Scalar(lo, lo, lo),
                         Scalar(up, up, up), flags);
    }
    
    imshow("image", dst);
    cout << area << " pixels were repainted\n";
}


int mainForFloodfill( )
{
    char* filename="/Users/huili/Downloads/capture/1133.ppm";
    image0 = imread(filename, 1);
    
    if( image0.empty() )
    {
        cout << "Image empty. Usage: ffilldemo <image_name>\n";
        return 0;
    }
    help();
    image0.copyTo(image);
    cvtColor(image0, gray, CV_BGR2GRAY);
    mask.create(image0.rows+2, image0.cols+2, CV_8UC1);
    
    namedWindow( "image", 0 );
    createTrackbar( "lo_diff", "image", &loDiff, 255, 0 );
    createTrackbar( "up_diff", "image", &upDiff, 255, 0 );
    
    setMouseCallback( "image", onMouse, 0 );
    
    for(;;)
    {
        imshow("image", isColor ? image : gray);
        
        int c = waitKey(0);
        if( (c & 255) == 27 )
        {
            cout << "Exiting ...\n";
            break;
        }
        switch( (char)c )
        {
            case 'c':
                if( isColor )
                {
                    cout << "Grayscale mode is set\n";
                    cvtColor(image0, gray, CV_BGR2GRAY);
                    mask = Scalar::all(0);
                    isColor = false;
                }
                else
                {
                    cout << "Color mode is set\n";
                    image0.copyTo(image);
                    mask = Scalar::all(0);
                    isColor = true;
                }
                break;
            case 'm':
                if( useMask )
                {
                    destroyWindow( "mask" );
                    useMask = false;
                }
                else
                {
                    namedWindow( "mask", 0 );
                    mask = Scalar::all(0);
                    imshow("mask", mask);
                    useMask = true;
                }
                break;
            case 'r':
                cout << "Original image is restored\n";
                image0.copyTo(image);
                cvtColor(image, gray, CV_BGR2GRAY);
                mask = Scalar::all(0);
                break;
            case 's':
                cout << "Simple floodfill mode is set\n";
                ffillMode = 0;
                break;
            case 'f':
                cout << "Fixed Range floodfill mode is set\n";
                ffillMode = 1;
                break;
            case 'g':
                cout << "Gradient (floating range) floodfill mode is set\n";
                ffillMode = 2;
                break;
            case '4':
                cout << "4-connectivity mode is set\n";
                connectivity = 4;
                break;
            case '8':
                cout << "8-connectivity mode is set\n";
                connectivity = 8;
                break;
        }
    }
    
    return 0;
}


int mainForSegment(int, char** argv)
{
    // Load the image
    Mat src0 = imread("/Users/huili/Downloads/capture4/773.ppm");
    Mat src;
cv:resize(src0, src, cvSize(src0.cols/2, src0.rows/2));
    // Check if everything was fine
    if (!src.data)
        return -1;
    // Show source image
    imshow("Source Image", src);
    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    for( int x = 0; x < src.rows; x++ ) {
        for( int y = 0; y < src.cols; y++ ) {
            if ( src.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
                src.at<Vec3b>(x, y)[0] = 0;
                src.at<Vec3b>(x, y)[1] = 0;
                src.at<Vec3b>(x, y)[2] = 0;
            }
        }
    }
    // Show output image
    imshow("Black Background Image", src);
    // Create a kernel that we will use for accuting/sharpening our image
    Mat kernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    Mat sharp = src; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    //    imwrite("/Users/huili/Downloads/capture4/1-sharped.ppm", imgResult);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow( "New Sharped Image", imgResult );
    src = imgResult; // copy back
    // Create binary image from source image
    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, 0, 128, CV_THRESH_BINARY | CV_THRESH_OTSU);
    imshow("Binary Image", bw);
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    
    imshow("Distance Transform Image", dist);
    imwrite("/Users/huili/Downloads/capture4/773-dt.png", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
    // Draw the background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    imshow("Markers", markers*10000);
    // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i,j) = colors[index-1];
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    // Visualize the final image
    imshow("Final Result", dst);
    waitKey(0);
    return 0;
}

