
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






int main()
{

    imageCvtColor();
    return 0;
}















//int main(  )
//{
//    String imagePath="/Users/yangxuewu/Desktop/癌蚌photo/IMG_8983.JPG";
//    String str="/Users/yangxuewu/Downloads/enenen.png";
//    
// 
//    Mat src,gray;
//    src=imread(str);
//    cvtColor(src,gray,CV_RGB2GRAY);
//    int bins = 256;
//    int hist_size[] = {bins};
//    float range[] = { 0, 256 };
//    const float* ranges[] = { range};
//    MatND hist;
//    int channels[] = {0};
//    
//    calcHist( &gray, 1, channels, Mat(), // do not use mask
//             hist, 1, hist_size, ranges,
//             true, // the histogram is uniform
//             false );
//    
//    double max_val;
//    minMaxLoc(hist, 0, &max_val, 0, 0);
//    int scale = 2;
//    int hist_height=256;
//    Mat hist_img = Mat::zeros(hist_height,bins*scale, CV_8UC3);
//    for(int i=0;i<bins;i++)
//    {
//        float bin_val = hist.at<float>(i);
//        int intensity = cvRound(bin_val*hist_height/max_val);  //要绘制的高度
//        rectangle(hist_img,Point(i*scale,hist_height-1),
//                  Point((i+1)*scale - 1, hist_height - intensity),
//                  CV_RGB(255,255,255));
//    }
//    imshow( "Source", src );
//    imshow( "Gray Histogram", hist_img );
//    waitKey(0);
//    return 0;  
//}
//














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






//int main(int argc,char** argv)
//{
//     String filename="/Users/yangxuewu/Downloads/7h4SJ.jpg";
//      String filename1="/Users/yangxuewu/Downloads/8ipeJ.jpg";
//    //Read
//    cv::Mat img1=cv::imread("/Users/yangxuewu/Downloads/7h4SJ.jpg");
//    cv::Mat img2=cv::imread("/Users/yangxuewu/Downloads/8ipeJ.jpg");
//    
//    //Detect
//    std::vector<cv::Rect> letterBBoxes1=detectLetters(img1);
//    std::vector<cv::Rect> letterBBoxes2=detectLetters(img2);
// 
//    //Display
//    for(int i=0; i< letterBBoxes1.size(); i++)
//        cv::rectangle(img1,letterBBoxes1[i],cv::Scalar(0,255,0),3,8,0);
//    cv::imwrite( "7h4SJ.jpg", img1);
//    imshow("7h4SJ.jpg",img1);
//    for(int i=0; i< letterBBoxes2.size(); i++)
//        cv::rectangle(img2,letterBBoxes2[i],cv::Scalar(0,255,0),3,8,0);
//    cv::imwrite( "8ipeJ.jpg", img2);
//      imshow("8ipeJ.jpg",img2);
//    waitKey(1000000);
//    return 0;
//}
//





//
//int main(int argc,char** argv)
//{
//    String filename="/Users/yangxuewu/Downloads/8ipeJ.jpg";
//    //String filename="/Users/yangxuewu/Downloads/7h4SJ.jpg";
//    //String filename= "/Users/yangxuewu/Downloads/骨钉/jkfs-22-232-g001-l.jpg";
//   // String filename= "/Users/yangxuewu/Downloads/骨钉/11111.png";
//   //  String filename= "/Users/yangxuewu/Downloads/骨钉/A1.png";
////     String filename= "/Users/yangxuewu/Downloads/骨钉/A.JPG";
//    Mat large = imread(filename);
//    Mat rgb;
//    // downsample and use it for processing
//    pyrDown(large, rgb);
//    Mat small;
//    cvtColor(rgb, small, CV_BGR2GRAY);
//    // morphological gradient
//    Mat grad;
//    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
//    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
//    // binarize
//    Mat bw;
//    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
//    // connect horizontally oriented regions
//    Mat connected;
//    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
//    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
//    // find contours
//    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//    // filter contours
//    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
//    {
//        Rect rect = boundingRect(contours[idx]);
//        Mat maskROI(mask, rect);
//        maskROI = Scalar(0, 0, 0);
//        // fill the contour
//        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
//        // ratio of non-zero pixels in the filled region
//        double r = (double)countNonZero(maskROI)/(rect.width*rect.height);
//        
//        if (r > .45 /* assume at least 45% of the area is filled if it contains text */
//            &&
//            (rect.height > 8 && rect.width > 8) /* constraints on region size */
//            /* these two conditions alone are not very robust. better to use something
//             like the number of significant peaks in a horizontal projection as a third condition */
//            )
//        {
//            rectangle(rgb, rect, Scalar(0, 255, 0), 2);
//        }
//    }
//    imshow("游戏原画",rgb);
//  //  imwrite(OUTPUT_FOLDER_PATH + string("rgb.jpg"), rgb);
//    waitKey(1000000);
//    
//    return 0;
//}


//
//int main()
//{
//     String filename="/Users/yangxuewu/Downloads/8ipeJ.jpg";
//    cv::Mat image = cv::imread(filename,0) ;
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
//
//
//int main(void){
//    
////    char* fileName;
////    cin>>fileName;
//   // getContours(fileName);
////    getContours("/Users/yangxuewu/Downloads/example1.jpg");
//   getContours("/Users/yangxuewu/Downloads/骨钉/jkfs-22-232-g001-l.jpg");
////    getContours("/Users/yangxuewu/Downloads/骨钉/11111.png");
//
//
//    
//}





//double alpha; /**< 控制对比度 */
//int beta;  /**< 控制亮度 */
//
//int main( int argc, char** argv )
//{
//    String filename="/Users/yangxuewu/Downloads/enenen.png";
//    
//    /// 读入用户提供的图像
//    Mat image = imread(filename,0);
//    Mat new_image = Mat::zeros( image.size(), image.type() );
//    
//    /// 初始化
//    cout << " Basic Linear Transforms " << endl;
//    cout << "-------------------------" << endl;
//    cout << "* Enter the alpha value [1.0-3.0]: ";
//    cin >> alpha;
//    cout << "* Enter the beta value [0-100]: ";
//    cin >> beta;
//    
//    /// 执行运算 new_image(i,j) = alpha*image(i,j) + beta
//    for( int y = 0; y < image.rows; y++ )
//    {
//        for( int x = 0; x < image.cols; x++ )
//        {
//            for( int c = 0; c < 3; c++ )
//            {
//                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
//            }
//        }
//    }
//    
//    /// 创建窗口
//    namedWindow("Original Image", 1);
//    namedWindow("New Image", 1);
//    
//    /// 显示图像
//    imshow("Original Image", image);
//    imshow("New Image", new_image);
//    
//    /// 等待用户按键
//    waitKey();
//    return 0;
//}
