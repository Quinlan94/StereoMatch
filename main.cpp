#include <ctime>

#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <string>
#include <ctime>
#include <cstdlib>

#include "StereoMatching.h"

//using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;




void on_trackbars(int , void *);








int  main()
{

 

         Mat left_temp = imread("/home/quinlan/桌面/MiddEval3/trainingQ/Recycle/im0.png");
         Mat right_temp = imread("/home/quinlan/桌面/MiddEval3/trainingQ/Recycle/im1.png");


         Mat left,right;
         /*GaussianBlur(left_temp,left,Size(3,3),0);
         GaussianBlur(right_temp,right,Size(3,3),0);
         bilateralFilter(left_temp,left,20,10,10);
    bilateralFilter(right_temp,right,20,10,10);*/







    Mat test = imread("/home/quinlan/Learn/StereoMatch/dataset/initial_ArtL.png",0);
    Mat test_err = imread("/home/quinlan/Learn/StereoMatch/dataset/error_depth_ArtL.png",0);
    int height = test.size().height;
    int width = test.size().width;
//    Mat test_1(test.size().height, test.size().width, CV_8UC1);
//    Mat test_err_1(test.size().height, test.size().width, CV_8UC1);
//    for (int y = 0; y < height; ++y)
//        for (int x = 0; x < width; ++x)
//        {
//            test_1.at<uchar>(y,x) = test.at<uchar>(y,x);
//            test_err_1.at<uchar>(y,x) = test_err.at<uchar>(y,x);
//        }





        string disp1_name = "/home/quinlan/Learn/StereoMatch/dataset/optimal_ArtL.png";

    /*fliter_err(test_1,left,3);
    consistent_check(width, height, test_1, test_err_1);
    //fliter_err(test_1,left,3);
    fliter_err(test_1,test_err_1,left,6);

    imwrite("/home/quinlan/Learn/StereoMatch/dataset/shit_abs_disp.png", test_1);
    normalize(test_1, test_1, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("tu", test_1);
    imwrite("/home/quinlan/Learn/StereoMatch/dataset/shit.png", test_1);
*/
        Mat disp = ASW(left_temp, right_temp, "check");
        //Mat disp = asw(left_temp, right_temp, "left");


        imwrite("/home/quinlan/Learn/StereoMatch/dataset/abs_disp_ArtL.png", disp);

        normalize(disp, disp, 0, 255, NORM_MINMAX, CV_8UC1);
        imshow("tu", disp);
        imwrite(disp1_name, disp);
        waitKey(0);


    return 0;
}


/*
void on_trackbars(int  , void *)
{
    namedWindow("【效果图窗口】", 1);
//canny边缘检测
Mat DstPic, edge, grayImage;

//创建与src同类型和同大小的矩阵
DstPic.create(left_1.size(), left_1.type());

//将原始图转化为灰度图
cvtColor(left_1, grayImage, COLOR_BGR2GRAY);

//先使用3*3内核来降噪
blur(grayImage, edge, Size(3, 3));
    int thresh_2=  thresh_1*3;
//运行canny算子
Canny(edge, edge, thresh_1, thresh_2, 3);

imshow("边缘提取效果", edge);



}*/
