#include <ctime>

#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <string>
#include <ctime>
#include <cstdlib>

#include "StereoMatching.h"

//using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;




int thresh_1 = 3;

cv::Mat left_1;
void on_trackbars(int , void *);


/*
void
test_image_quality(string type)
{
    double average_rate_left = 0;
    double average_rate_right = 0;
    for (auto name : img_name)
    {
        Mat ground_truth_left = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/disp1.png");
        Mat ground_truth_right = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/disp5.png");
        Mat test_img_left = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp1_" + type + ".png");
        Mat test_img_right = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp5_" + type + ".png");

        ground_truth_left /= 3;
        ground_truth_right /= 3;
        test_img_left /= 3;
        test_img_right /= 3;

        int width = ground_truth_left.size().width;
        int height = ground_truth_left.size().height;
        int bad_pixels_left = 0;
        int bad_pixels_right = 0;
        int count = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int delta = abs(test_img_left.at<uchar>(y, x) - ground_truth_left.at<uchar>(y, x));
                if (delta > 1 && test_img_left.at<uchar>(y, x) != 0)
                {
                    bad_pixels_left++;
                }
            }
        }

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int delta = abs(test_img_right.at<uchar>(y, x) - ground_truth_right.at<uchar>(y, x));
                if (delta > 1 && test_img_right.at<uchar>(y, x) != 0)
                {
                    bad_pixels_right++;
                }
            }
        }

        double rate_left = double(bad_pixels_left) / (height * width);
        double rate_right = double(bad_pixels_right) / (height * width);
        printf("[%s %s]: left_rate:%.4lf%% right_rate:%.4lf%%\n", name.c_str(), type.c_str(), rate_left * 100, rate_right * 100);
        average_rate_left += rate_left;
        average_rate_right += rate_right;
    }
    average_rate_left /= 21;
    average_rate_right /= 21;
    printf("[%s average]: left:%.4lf%% right:%.4lf%%\n", type.c_str(), average_rate_left * 100, average_rate_right * 100);
}
*/





int  main()
{

    /*test_image_quality("SSD");
    test_image_quality("NCC");
    test_image_quality("SSD_CONSTANT");
    test_image_quality("NCC_CONSTANT");
    test_image_quality("ASW");
*/

         Mat left_temp = imread("/home/quinlan/桌面/MiddEval3/trainingQ/ArtL/im0.png");
         Mat right_temp = imread("/home/quinlan/桌面/MiddEval3/trainingQ/ArtL/im1.png");

    Mat left,right;
         GaussianBlur(left_temp,left,Size(3,3),0);
        GaussianBlur(right_temp,right,Size(3,3),0);



            //创建窗口
/*
         namedWindow("【效果图窗口】", 1);


         //创建轨迹条
        createTrackbar("thresh_1：", "【效果图窗口】", &thresh_1, 200,on_trackbars);




        on_trackbars(thresh_1,0);
*/



        string disp1_name = "/home/quinlan/Learn/StereoMatch/dataset/asw-ArtL-1原图.png";
        //string disp5_name = "C:/Users/Administrator/Desktop/DA/dataset/disp_row3col3.png";

        cout << "creating " << disp1_name << endl;

        Mat disp = ASW(left, right, "left");

        imwrite("/home/quinlan/Learn/StereoMatch/dataset/abs_disp.png", disp);

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
