
/*
 ayoungprogrammer.blogspot.com
 
 Part 1: Extracting contours from text
 
 */

#include <iostream>

//#include <Windows.h>

#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/ml.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "opencvTest.hpp"
#define OUTPUT_FOLDER_PATH      string("")
#endif

using namespace std;
using namespace cv;


int main(){
    MultiChannelBlending();
    waitKey(0);
    return 0;
}

////******************************  做灰度处理   二值化
//int main(){
//    
//    String str="/Users/yangxuewu/Downloads/enenen.png";
//   // String imagePath="/Users/yangxuewu/Downloads/221.png";
//    String imagePath="/Users/yangxuewu/Downloads/image.JPG";
//
//    String outImagePath="/Users/yangxuewu/Downloads/result.png";
//    
//    Mat img = imread(imagePath);
//    Mat gray;
//    
//    cvtColor(img, gray, CV_BGR2GRAY);
//    // colorReduce(gray,gray,100);//  颜色空间缩减
//    threshold(gray,gray,140,255,THRESH_BINARY);
//    // cvNamedWindow("游戏原画");
//    //  imshow("imageCvtColor",gray);
//    
//    
//    //  namedWindow("gray", CV_WINDOW_NORMAL);
//    //  imshow("img", img);
//    imwrite(outImagePath,gray);
//    imshow("gray", gray);
//    
//    waitKey(0);
//    
//     return 0 ;
//    
//
//    
//}

//---------------------------------【colorReduce( )函数】---------------------------------
//          描述：使用【指针访问：C操作符[ ]】方法版的颜色空间缩减函数
//----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div)
{
    //参数准备
    outputImage = inputImage.clone();  //拷贝实参到临时变量
    int rowNumber = outputImage.rows;  //行数
    int colNumber = outputImage.cols*outputImage.channels();  //列数 x 通道数=每一行元素的个数
    
    //双重循环，遍历所有的像素值
    for(int i = 0;i < rowNumber;i++)  //行循环
    {
        uchar* data = outputImage.ptr<uchar>(i);  //获取第i行的首地址
        for(int j = 0;j < colNumber;j++)   //列循环
        {
            // ---------【开始处理每个像素】-------------
            data[j] = data[j]/div*div + div/2;
            // ----------【处理结束】---------------------
        }  //行处理结束
    }
}





int main21()
{
    
    
    //构建100×100的10通道8位矩阵
    
    //  Mat M;
    // M.create(1,1,CV_8UC(10));
    
    
    //构建一个100×100×100的8位三维矩阵
    //    int sz[] = {1,1,1};
    //    Mat M(3, sz, CV_32F, Scalar::all(0));
    //
    //第4行加上第6行的3倍赋值给第4行
    //  Mat M(2,2,CV_8UC4,Scalar(1,2,3,4));
    //M.row(0) = M.row(0) + M.row(1)*3;
    // M.col(0)=M.col(0)+M.col(1)*3;
    //  cout << "M (OpenCV默认风格) = " << M << ";" << endl << endl;
    // M.at<double>(i,j);
    
    
    
    String str="/Users/yangxuewu/Downloads/1.jpg";
    
    //  Mat img(Size(320,240),CV_8UC3);
    Mat imge;
    Mat M=imread(str);
    cvtColor(M, imge, CV_BGR2Lab);
    imshow("1", imge);
    
    imshow("122", M);
    
    
    
    //    Mat roi(img, Rect(10,10,100,100));
    //    roi = Scalar(0,255,0);
    //  cout << "M (OpenCV默认风格) = " << M << ";" << endl << endl;
    waitKey();
    return 0 ;
}

//********************* 读取图片模式
//int main()
//{
//    
//    String str="/Users/yangxuewu/Downloads/1.jpg";
//    
//    Mat image1=imread(str,2);
//    Mat image2=imread(str,0);
//    Mat image3=imread(str,199);
//    imshow("1", image1);
//    imshow("2", image2);
//    imshow("3", image3);
//    waitKey(0);
//    return 0;
//}
//






//**************************   读取视频
int mainVideoCapture(){
    
    String str="/Users/yangxuewu/Downloads/video.mp4";
    
   	//【1】读入视频
    VideoCapture capture(0);
    
    Mat edage;
    //【2】循环显示每一帧
    while(1)
    {
        Mat frame;//定义一个Mat变量，用于存储每一帧的图像
        capture>>frame;  //读取当前帧
        
        cvtColor(frame, edage, COLOR_RGB2GRAY);
        blur(edage, edage,Size(11,11));
        Canny(edage, edage, 0, 30,3);
        //若视频播放完成，退出循环
        if (frame.empty())
        {
            break;
        }
        
        imshow("读取视频",edage);  //显示当前帧
        waitKey(30);  //延时30ms
    }
    return 0;
}



















//***********************************************************************直方图
int maincalcHist(  )
{
    String imagePath="/Users/yangxuewu/Desktop/癌蚌photo/IMG_8983.JPG";
    String str="/Users/yangxuewu/Downloads/enenen.png";
    
    
    Mat src,gray;
    src=imread(str);
    cvtColor(src,gray,CV_RGB2GRAY);
    int bins = 256;
    int hist_size[] = {bins};
    float range[] = { 0, 256 };
    const float* ranges[] = { range};
    MatND hist;
    int channels[] = {0};
    
    calcHist( &gray, 1, channels, Mat(), // do not use mask
             hist, 1, hist_size, ranges,
             true, // the histogram is uniform
             false );
    
    double max_val;
    minMaxLoc(hist, 0, &max_val, 0, 0);
    int scale = 2;
    int hist_height=256;
    Mat hist_img = Mat::zeros(hist_height,bins*scale, CV_8UC3);
    for(int i=0;i<bins;i++)
    {
        float bin_val = hist.at<float>(i);
        int intensity = cvRound(bin_val*hist_height/max_val);  //要绘制的高度
        rectangle(hist_img,Point(i*scale,hist_height-1),
                  Point((i+1)*scale - 1, hist_height - intensity),
                  CV_RGB(255,255,255));
    }
    imshow( "Source", src );
    imshow( "Gray Histogram", hist_img );
    waitKey(0);
    return 0;
}















class comparator{
public:
    bool operator()(vector<Point> c1,vector<Point>c2){
        
        return boundingRect( Mat(c1)).x<boundingRect( Mat(c2)).x;
        
    }
    
};



void extractContours(Mat& image,vector< vector<Point> > contours_poly){
    
    
    
    //Sort contorus by x value going from left to right
    sort(contours_poly.begin(),contours_poly.end(),comparator());
    
    
    //Loop through all contours to extract
    for( int i = 0; i< contours_poly.size(); i++ ){
        
        Rect r = boundingRect( Mat(contours_poly[i]) );
        
        
        Mat mask = Mat::zeros(image.size(), CV_8UC1);
        //Draw mask onto image
        drawContours(mask, contours_poly, i, Scalar(255), CV_FILLED);
        
        //Check for equal sign (2 dashes on top of each other) and merge
        if(i+1<contours_poly.size()){
            Rect r2 = boundingRect( Mat(contours_poly[i+1]) );
            if(abs(r2.x-r.x)<20){
                //Draw mask onto image
                drawContours(mask, contours_poly, i+1, Scalar(255), CV_FILLED);
                i++;
                int minX = min(r.x,r2.x);
                int minY = min(r.y,r2.y);
                int maxX =  max(r.x+r.width,r2.x+r2.width);
                int maxY = max(r.y+r.height,r2.y+r2.height);
                r = Rect(minX,minY,maxX - minX,maxY-minY);
                
            }
        }
        //Copy
        Mat extractPic;
        //Extract the character using the mask
        image.copyTo(extractPic,mask);
        Mat resizedPic = extractPic(r);
        
        cv::Mat image=resizedPic.clone();
        
        //Show image
        imshow("image",image);
        //char ch  =
        waitKey(0);
        stringstream searchMask;
        searchMask<<i<<".jpg";
        imwrite(searchMask.str(),resizedPic);
        
    }
    
    
    
    
    
}

void getContours(const char* filename)
{
    cv::Mat img = cv::imread(filename, 0);
    
    
    //Apply blur to smooth edges and use adapative thresholding
    cv::Size size(3,3);
    cv::GaussianBlur(img,img,size,0);
    //    cv::threshold( img,  img,  25, 255, THRESH_BINARY_INV);
    adaptiveThreshold(img, img,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,31,10);
    cv::bitwise_not(img, img);
    
    
    
    
    cv::Mat img2 = img.clone();
    
    
    std::vector<cv::Point> points;
    cv::Mat_<uchar>::iterator it = img.begin<uchar>();
    cv::Mat_<uchar>::iterator end = img.end<uchar>();
    for (; it != end; ++it)
        if (*it)
            points.push_back(it.pos());
    
    cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));
    
    double angle = box.angle;
    if (angle < -45.)
        angle += 90.;
    
    cv::Point2f vertices[4];
    box.points(vertices);
    for(int i = 0; i < 4; ++i)
        cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);
    
    
    
    cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, angle, 1);
    
    cv::Mat rotated;
    cv::warpAffine(img2, rotated, rot_mat, img.size(), cv::INTER_CUBIC);
    
    
    
    cv::Size box_size = box.size;
    if (box.angle < -45.)
        std::swap(box_size.width, box_size.height);
    cv::Mat cropped;
    
    cv::getRectSubPix(rotated, box_size, box.center, cropped);
    cv::imshow("Cropped", cropped);
    imwrite("example5.jpg",cropped);
    
    Mat cropped2=cropped.clone();
    cvtColor(cropped2,cropped2,CV_GRAY2RGB);
    
    Mat cropped3 = cropped.clone();
    cvtColor(cropped3,cropped3,CV_GRAY2RGB);
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    /// Find contours
    cv:: findContours( cropped, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0) );
    
    
    
    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    
    
    //Get poly contours
    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
    }
    
    
    //Get only important contours, merge contours that are within another
    vector<vector<Point> > validContours;
    for (int i=0;i<contours_poly.size();i++){
        
        Rect r = boundingRect(Mat(contours_poly[i]));
        if(r.area()<100)continue;
        bool inside = false;
        for(int j=0;j<contours_poly.size();j++){
            if(j==i)continue;
            
            Rect r2 = boundingRect(Mat(contours_poly[j]));
            if(r2.area()<100||r2.area()<r.area())continue;
            if(r.x>r2.x&&r.x+r.width<r2.x+r2.width&&
               r.y>r2.y&&r.y+r.height<r2.y+r2.height){
                
                inside = true;
            }
        }
        if(inside)continue;
        validContours.push_back(contours_poly[i]);
    }
    
    
    //Get bounding rects
    for(int i=0;i<validContours.size();i++){
        boundRect[i] = boundingRect( Mat(validContours[i]) );
    }
    
    
    //Display
    Scalar color = Scalar(0,255,0);
    for( int i = 0; i< validContours.size(); i++ )    {
        if(boundRect[i].area()<100)continue;
        drawContours( cropped2, validContours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        rectangle( cropped2, boundRect[i].tl(), boundRect[i].br(),color, 2, 8, 0 );
    }
    
    //imwrite("example6.jpg",cropped2);
    imshow("Contours",cropped2);
    
    extractContours(cropped3,validContours);
    
    cv::waitKey(0);
    
}


std::vector<cv::Rect> detectLetters(cv::Mat img)
{
    std::vector<cv::Rect> boundRect;
    cv::Mat img_gray, img_sobel, img_threshold, element;
    cvtColor(img, img_gray, CV_BGR2GRAY);
    cv::Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
    element = getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3) );
    cv::morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element); //Does the trick
    std::vector< std::vector< cv::Point> > contours;
    cv::findContours(img_threshold, contours, 0, 1);
    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
        if (contours[i].size()>100)
        {
            cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
            cv::Rect appRect( boundingRect( cv::Mat(contours_poly[i]) ));
            if (appRect.width>appRect.height)
                boundRect.push_back(appRect);
        }
    return boundRect;
}





//************************************** 文字看轮廓
//int main(int argc,char** argv)
//{
//    String filename="/Users/yangxuewu/Downloads/image.JPG";
//    String filename1="/Users/yangxuewu/Downloads/11.png";
//    String outImagePath="/Users/yangxuewu/Downloads/result1.png";
//    //
//    //Read
//    cv::Mat img1=cv::imread(filename);
//   // cv::Mat img2=cv::imread(filename1);
//    
//    //Detect
//    std::vector<cv::Rect> letterBBoxes1=detectLetters(img1);
//   // std::vector<cv::Rect> letterBBoxes2=detectLetters(img2);
//    
//    //Display
//    for(int i=0; i< letterBBoxes1.size(); i++)
//        cv::rectangle(img1,letterBBoxes1[i],cv::Scalar(0,255,0),3,8,0);
//    cv::imwrite(outImagePath, img1);
//    imshow("7h4SJ.jpg",img1);
////   
////    for(int i=0; i< letterBBoxes2.size(); i++)
////        cv::rectangle(img2,letterBBoxes2[i],cv::Scalar(0,255,0),3,8,0);
////    cv::imwrite( "8ipeJ.jpg", img2);
////    imshow("8ipeJ.jpg",img2);
//    waitKey(1000000);
//    return 0;
//}







//************************************************************* 框出文字区域     优化过的
int mainfindContours(int argc,char** argv)
{
    //    String filename="/Users/yangxuewu/Downloads/8ipeJ.jpg";
    String filename="/Users/yangxuewu/Downloads/221.png";
    String result="/Users/yangxuewu/Downloads/resuld.JPG";
    //String filename= "/Users/yangxuewu/Downloads/骨钉/jkfs-22-232-g001-l.jpg";
    // String filename= "/Users/yangxuewu/Downloads/骨钉/11111.png";
    //  String filename= "/Users/yangxuewu/Downloads/骨钉/A1.png";
    //String filename= "/Users/yangxuewu/Downloads/骨钉/goodImge.JPG";
    Mat large = imread(filename);
    Mat rgb;
    // downsample and use it for processing
    pyrDown(large, rgb);
    Mat small;
    cvtColor(rgb, small, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
    // binarize
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
    {
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
        // ratio of non-zero pixels in the filled region
        double r = (double)countNonZero(maskROI)/(rect.width*rect.height);
        
        if (r > .45 /* assume at least 45% of the area is filled if it contains text */
            &&
            (rect.height > 8 && rect.width > 8) /* constraints on region size */
            /* these two conditions alone are not very robust. better to use something
             like the number of significant peaks in a horizontal projection as a third condition */
            )
        {
            rectangle(rgb, rect, Scalar(0, 255, 0), 2);
        }
    }
    imshow("游戏原画",rgb);
    imwrite(result, rgb);
    waitKey(1000000);
    
    return 0;
}




//********************************************************* 查找文字轮廓
//int main(){
//
//    // String filename="/Users/yangxuewu/Downloads/goodImge.JPG";
//
//    //String filename="/Users/yangxuewu/Downloads/11111.png";
//
//    String filename="/Users/yangxuewu/Downloads/eng.png";
//
////    char* fileName;
////    cin>>fileName;
//   // getContours(fileName);
////    getContours("/Users/yangxuewu/Downloads/example1.jpg");
//  // getContours(filename);
//  //  getContours("/Users/yangxuewu/Downloads/eng.png");
//
//  getContours("/Users/yangxuewu/Downloads/11.png");
//
//}

