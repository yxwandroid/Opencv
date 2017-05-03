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
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

//************************************** 文字看轮廓
std::vector<cv::Rect> detectLetters(cv::Mat img)
{
    String outImagePath="/Users/yangxuewu/Downloads/result2.png";
    
    std::vector<cv::Rect> boundRect;
    cv::Mat img_gray, img_sobel, img_threshold, element,image2;
    cvtColor(img, img_gray, CV_BGR2GRAY);//灰度处理
 
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

int StructuringElement()
{
    String filename="/Users/yangxuewu/Downloads/image.JPG";
      String filename1="/Users/yangxuewu/Downloads/11.png";
    String outImagePath="/Users/yangxuewu/Downloads/result1.png";
    
    String image3="/Users/yangxuewu/Downloads/IMG_8998.JPG";//147
    
    String image4="/Users/yangxuewu/Downloads/image_new_test.png";//147
    String image5="/Users/yangxuewu/Downloads/image1_1.png";//147
    
    String image6="/Users/yangxuewu/Downloads/result.png";//147
    String image7="/Users/yangxuewu/Downloads/result_new.png";//147
    //
    String image8="/Users/yangxuewu/Downloads/Bonebutyl/WechatIMG6.jpeg";
    String image9="/Users/yangxuewu/Downloads/Bonebutyl/WechatIMG7.jpeg";

    
    //Read
    cv::Mat img1=cv::imread(image9);
    cv::Mat img2;
       //Detect
    std::vector<cv::Rect> letterBBoxes1=detectLetters(img1);
 
    //Display
    for(int i=0; i< letterBBoxes1.size(); i++)
        cv::rectangle(img1,letterBBoxes1[i],cv::Scalar(0,255,0),3,8,0);
    cv::imwrite(outImagePath, img1);
    imshow("7h4SJ.jpg",img1);
    
    waitKey(1000000);
    return 0;
}



////******************************  截取指定区域
int imageCutOut(){
    
    String str="/Users/yangxuewu/Downloads/enenen.png";
    // String imagePath="/Users/yangxuewu/Downloads/221.png";
    String imagePath="/Users/yangxuewu/Downloads/image.JPG";
    String outImagePath="/Users/yangxuewu/Downloads/result.png";
    
    
    String image1="/Users/yangxuewu/Downloads/image1.png";//153
    String image2="/Users/yangxuewu/Downloads/image2.png";//147
    String image3="/Users/yangxuewu/Downloads/IMG_8998.JPG";//147
    String image4="/Users/yangxuewu/Downloads/IMG_8996.JPG";//180
    String image5="/Users/yangxuewu/Downloads/Bonebutyl/WechatIMG6.jpeg";//224
    String image6="/Users/yangxuewu/Downloads/Bonebutyl/image6.png";//224
    String str2="/Users/yangxuewu/Downloads/image_calcHist.png";//149
    String str3="/Users/yangxuewu/Downloads/goodImge.JPG";//170
    
    
    Mat img = imread(str3);
   // Rect(int x, int y, int width, int height)
    Rect rect(10, 20, 100, 50);
    Mat gray = img(rect);
   // Mat gray=img(Range(0,10),Range(20,30));

    imwrite(outImagePath,gray);
//    imshow("img", img);
    imshow("gray", gray);
    
    waitKey(0);
    return 0 ;
}


// 获取感兴趣区域
int getROI(Mat image, Rect rect)
{
    Mat img=image.clone();
    Mat roi;
    int cols=img.cols, rows=img.rows;
    //ROI越界，返回
    if(cols-1-rect.x<rect.width||rows-1-rect.y<rect.height)
        return -1;
    roi=img(Rect(rect.x, rect.y, rect.width, rect.height));
    rectangle(img, rect, Scalar(0, 0, 255),2);
    imshow("SignROI",img);
    image.copyTo(img);    //ROI和它的父图像指向同一块缓冲区，经次操作消除 标记ROI的矩形框
    imshow("ROI",roi);
    return 0;
}


////******************************  做灰度处理   二值化
int threshold(){
    
    String str="/Users/yangxuewu/Downloads/enenen.png";
    // String imagePath="/Users/yangxuewu/Downloads/221.png";
    String imagePath="/Users/yangxuewu/Downloads/image.JPG";
    String outImagePath="/Users/yangxuewu/Downloads/result.png";
    
    
    String image1="/Users/yangxuewu/Downloads/image1.png";//153
    String image2="/Users/yangxuewu/Downloads/image2.png";//147
    String image3="/Users/yangxuewu/Downloads/IMG_8998.JPG";//147
    String image4="/Users/yangxuewu/Downloads/IMG_8996.JPG";//180
    String image5="/Users/yangxuewu/Downloads/Bonebutyl/WechatIMG6.jpeg";//224
    String image6="/Users/yangxuewu/Downloads/Bonebutyl/image6.png";//224
    String str2="/Users/yangxuewu/Downloads/image_calcHist.png";//149
    String str3="/Users/yangxuewu/Downloads/goodImge.JPG";//170

   
    Mat img = imread(str3);
    Mat gray;
    
    cvtColor(img, gray, CV_BGR2GRAY);
    // colorReduce(gray,gray,100);//  颜色空间缩减
    threshold(gray,gray,170,255,THRESH_BINARY);
    // cvNamedWindow("游戏原画");
    //  imshow("imageCvtColor",gray);
    
    
    //  namedWindow("gray", CV_WINDOW_NORMAL);
    //  imshow("img", img);
    imwrite(outImagePath,gray);
    imshow("gray", gray);
    
    waitKey(0);
    return 0 ;
}



void drawHistImg(const Mat &src, Mat &dst){
    int histSize = 256;
    float histMaxValue = 0;
    for(int i=0; i<histSize; i++){
        float tempValue = src.at<float>(i);
        if(histMaxValue < tempValue){
            histMaxValue = tempValue;
        }
    }
    
    float scale = (0.9*256)/histMaxValue;
    for(int i=0; i<histSize; i++){
        int intensity = static_cast<int>(src.at<float>(i)*scale);
        line(dst,Point(i,255),Point(i,255-intensity),Scalar(0));
    }
}


//***********************************************************************直方图
int calcHist()
{
    String imagePath="/Users/yangxuewu/Desktop/癌蚌photo/IMG_8983.JPG";
    String str="/Users/yangxuewu/Downloads/equalizeHist.png";
    String str2="/Users/yangxuewu/Downloads/image_calcHist.png";

    Mat src,gray;
    src=imread(str2);
    cvtColor(src,gray,CV_RGB2GRAY);   //转成灰度
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

//************************************************************* 框出文字区域     优化过的
int findContours()
{
//        String filename="/Users/yangxuewu/Downloads/Bonebutyl/WechatIMG11.jpeg";
    String filename="/Users/yangxuewu/Downloads/221.png";
    String outImagePath="/Users/yangxuewu/Downloads/ocrtext.png";
//    String filename2="/Users/yangxuewu/Downloads/WechatIMG6.jpeg";
    
    String result="/Users/yangxuewu/Downloads/resuld.JPG";
    //String filename= "/Users/yangxuewu/Downloads/骨钉/jkfs-22-232-g001-l.jpg";
    // String filename= "/Users/yangxuewu/Downloads/骨钉/11111.png";
    //  String filename= "/Users/yangxuewu/Downloads/骨钉/A1.png";
    //String filename= "/Users/yangxuewu/Downloads/骨钉/goodImge.JPG";
    
    Mat large = imread(filename);
    Mat rgb;
    // downsample and use it for processing
    pyrDown(large, rgb);  //缩小过程
    //pyrUp(large,rgb);//  放大过程
    Mat small;
    cvtColor(rgb, small, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel); // 开闭运算
    // binarize
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU); //二值化
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
        
        if (r >.25 /* assume at least 45% of the area is filled if it contains text */
            &&(rect.height > 8&& rect.width >8) /* constraints on region size */
            /* these two conditions alone are not very robust. better to use something
             like the number of significant peaks in a horizontal projection as a third condition */){
             //圈出区域
            //rectangle(rgb, rect, Scalar(0, 0, 0), 2);
            
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
    threshold(gray,gray,150,255,THRESH_BINARY);
   
    double scale=4;

    Size ResImgSiz = Size(gray.cols*scale, gray.rows*scale);
    Mat ResImg = Mat(ResImgSiz, gray.type());
    resize(gray, ResImg, ResImgSiz, CV_INTER_CUBIC);
   
    imshow("newRect",ResImg);
    imwrite(outImagePath,ResImg);
    

    imshow("游戏原画",rgb);
    imwrite(result, rgb);
    waitKey(1000000);
    
    return 0;
}



