#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#define pi 3.1415926

using namespace std;
using namespace cv;
int max_offset = 64;
int kernel_size = 5; // window size

Mat
bgr_to_grey(const Mat& bgr)
{
    int width = bgr.size().width;
    int height = bgr.size().height;
    Mat grey(height, width, 0);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uchar r = 0.333 * bgr.at<Vec3b>(y, x)[2];
            uchar g = 0.333 * bgr.at<Vec3b>(y, x)[1];
            uchar b = 0.333 * bgr.at<Vec3b>(y, x)[0];
            grey.at<uchar>(y, x) = uchar(r + g + b);
        }
    }

    return grey;
}

vector<Mat> getAmpAndAngle(Mat in1,int type)
{
    int width = in1.size().width;
    int height = in1.size().height;
    int two_side = 0;
    vector<Mat> AmpAndAngle;
    Mat Amp_mat;
    Mat Angle_mat;
    std::cout<<"类型："<<type<<endl;
   if(in1.channels()==3)
   {
       Mat Amp_mat_tmp(height, width, CV_32FC3);
       Mat Angle_mat_tmp(height, width, CV_32FC3);
       Amp_mat = Amp_mat_tmp;
       Angle_mat = Angle_mat_tmp;

       for (int k = 1; k < height; ++k)
       {

           for (int l = 0; l < width-1; ++l)
           {

               for (int c = 0; c < in1.channels(); ++c)
               {
                   if(type==two_side)
                   {
                       float g_x = in1.at<Vec3b>(k+1, l)[c] - in1.at<Vec3b>(k-1, l)[c];
                       float g_y = in1.at<Vec3b>(k, l+1)[c] - in1.at<Vec3b>(k, l-1)[c];

                       Amp_mat.at<Vec3f>(k,l)[c] = sqrt(g_x * g_x + g_y * g_y);

                       float angle = (atan2(g_y, g_x) * 180) / pi;
                       Angle_mat.at<Vec3f>(k,l)[c] = angle;
                   }
                   else
                   {
                       float g_x = in1.at<Vec3b>(k+1, l)[c] - in1.at<Vec3b>(k, l)[c];
                       float g_y = in1.at<Vec3b>(k, l+1)[c] - in1.at<Vec3b>(k, l)[c];

                       Amp_mat.at<Vec3f>(k,l)[c] = sqrt(g_x * g_x + g_y * g_y);

                       float angle = (atan2(g_y, g_x) * 180) / pi;
                       Angle_mat.at<Vec3f>(k,l)[c] = angle;
                   }

               }

           }

       }
       for (int l = 0; l < width; ++l)
       {
           for (int c = 0; c < in1.channels(); ++c)
           {
               Amp_mat.at<Vec3f>(0,l)[c] = Amp_mat.at<Vec3f>(1,l)[c];
               Angle_mat.at<Vec3f>(0,l)[c] = Angle_mat.at<Vec3f>(1,l)[c];
           }
       }
       for (int k = 0; k < height; ++k)
       {
           for (int c = 0; c < in1.channels(); ++c)
           {
               Amp_mat.at<Vec3f>(k,width-1)[c] = Amp_mat.at<Vec3f>(k,width-2)[c];
               Angle_mat.at<Vec3f>(k,width-1)[c] = Angle_mat.at<Vec3f>(k,width-2)[c];
           }
       }



   }
   else
   {
       Mat Amp_mat_tmp(height, width, CV_32FC1);
       Mat Angle_mat_tmp(height, width, CV_32FC1);
       Amp_mat = Amp_mat_tmp;
       Angle_mat = Angle_mat_tmp;
       for (int k = 0; k < height; ++k)
       {
           if (k < 1 || k > height - 1)
           {
               continue;
           }
           for (int l = 0; l < width; ++l)
           {
               if (l < 1 || l > width - 1)
                   continue;
               for (int c = 0; c < in1.channels(); ++c)
               {
                   if(type==two_side)
                   {
                       float g_x = in1.at<uchar>(k+1, l) - in1.at<uchar>(k-1, l);
                       float g_y = in1.at<uchar>(k, l+1) - in1.at<uchar>(k, l-1);

                       Amp_mat.at<float>(k,l) = sqrt(g_x * g_x + g_y * g_y);
                       float h = Amp_mat.at<float>(k,l);

                       float angle = (atan2(g_y, g_x) * 180) / pi;
                       Angle_mat.at<float>(k,l) = angle;
                   }
                   else
                   {
                       float g_x = in1.at<uchar>(k+1, l) - in1.at<uchar>(k, l);
                       float g_y = in1.at<uchar>(k, l+1) - in1.at<uchar>(k, l);

                       Amp_mat.at<float>(k,l) = sqrt(g_x * g_x + g_y * g_y);


                       float angle = (atan2f(g_y, g_x) * 180) / pi;
                       Angle_mat.at<float>(k,l) = angle;
                   }



               }
           }

       }
       for (int l = 0; l < width; ++l)
       {
               Amp_mat.at<float>(0,l)= Amp_mat.at<float>(1,l);
               Angle_mat.at<float>(0,l) = Angle_mat.at<float>(1,l);
       }
       for (int k = 0; k < height; ++k)
       {
               Amp_mat.at<float>(k,width-1) = Amp_mat.at<float>(k,width-2);
               Angle_mat.at<float>(k,width-1) = Angle_mat.at<float>(k,width-2);

       }
   }



       AmpAndAngle.push_back(Amp_mat);
       AmpAndAngle.push_back(Angle_mat);





    return AmpAndAngle;

}

bool best_to_bad_ordering(const pair<double,int> a, const pair<double,int> b)
{
    return a.first < b.first;
}
Mat CheckDepth(const Mat &depth_left,const Mat &depth_right)
{
    int width = depth_left.size().width;
    int height = depth_left.size().height;
    Mat depth_err(height, width, CV_8UC1,cv::Scalar::all(255));
    for(int y = 0;y<height;y++)
        for(int x=0;x<width;x++)
        {
            int offset_tmp = depth_left.at<uchar>(y,x);
            if(fabs(depth_right.at<uchar>(y,x-offset_tmp)-offset_tmp) > 1)
                depth_err.at<uchar>(y,x)=0;
        }
    return  depth_err;

}
Mat ncc_improve(Mat in1, Mat in2, string type, bool add_constant = false)
{
    int width = in1.size().width;
    int height = in1.size().height;



    Mat left = bgr_to_grey(in1);
    Mat right = bgr_to_grey(in2);
    clock_t timest = clock();
    vector<Mat> left_cost = getAmpAndAngle(in1,0);
    vector<Mat> right_cost = getAmpAndAngle(in1,0);
    cout << "梯度和角度计算时间：" << (clock() - timest) / (double)CLOCKS_PER_SEC << endl;

    int p =left_cost[0].at<Vec3b>(0,1)[2];

    if (add_constant)
    {
        right += 10;
    }

    Mat depth(height, width, 0);
    vector< vector<double> > max_ncc; // store max NCC value

    for (int i = 0; i < height; ++i)
    {
        vector<double> tmp(width, -2);
        max_ncc.push_back(tmp);
    }

    /*for (int offset = 1; offset <= max_offset; offset++)
    {
        Mat tmp(height, width, 0);
        // shift image depend on type to save calculation time
        if (type == "left")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < offset; x++)
                {
                    //tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
                    tmp.at<uchar>(y, x) = right_cost[0].at<uchar>(y, x);
                }

                for (int x = offset; x < width; x++)
                {
                    //tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);
                    tmp.at<uchar>(y, x) = right_cost[0].at<uchar>(y, x - offset);
                }
            }
        }
        else if (type == "right")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width - offset; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x + offset);
                }

                for (int x = width - offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x);
                }
            }
        }
        else {
            Mat tmp(0, 0, 0);
            return tmp;
        }

        // calculate each pixel's NCC value
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int start_x = max(0, x - kernel_size);
                int start_y = max(0, y - kernel_size);
                int end_x = min(width - 1, x + kernel_size);
                int end_y = min(height - 1, y + kernel_size);
                double n = (end_y - start_y) * (end_x - start_x);
                double res_ncc = 0;

                if (type == "left")
                {
                    double left_mean = 0, right_mean = 0;
                    double left_std = 0, right_std = 0;
                    double numerator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
//                            left_mean += left.at<uchar>(i, j);
//                            right_mean += tmp.at<uchar>(i, j);

                            left_mean += left_cost[0].at<uchar>(i, j);
                            right_mean += tmp.at<uchar>(i, j);
                        }
                    }

                    left_mean /= n;
                    right_mean /= n;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
//                            left_std += pow(left.at<uchar>(i, j) - left_mean, 2);
//                            right_std += pow(tmp.at<uchar>(i, j) - right_mean, 2);
//                            numerator += (left.at<uchar>(i, j) - left_mean) * (tmp.at<uchar>(i, j) - right_mean);

                            left_std += pow(left_cost[0].at<uchar>(i, j) - left_mean, 2);
                            right_std += pow(tmp.at<uchar>(i, j) - right_mean, 2);
                            numerator +=
                                    (left_cost[0].at<uchar>(i, j) - left_mean) * (tmp.at<uchar>(i, j) - right_mean);
                        }
                    }

                    numerator /= n;
                    left_std /= n;
                    right_std /= n;
                    res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;
                }
                else
                {
                    double left_mean = 0, right_mean = 0;
                    double left_std = 0, right_std = 0;
                    double numerator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_mean += tmp.at<uchar>(i, j);
                            right_mean += right.at<uchar>(i, j);
                        }
                    }

                    left_mean /= n;
                    right_mean /= n;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_std += pow(tmp.at<uchar>(i, j) - left_mean, 2);
                            right_std += pow(right.at<uchar>(i, j) - right_mean, 2);
                            numerator += (tmp.at<uchar>(i, j) - left_mean) * (right.at<uchar>(i, j) - right_mean);
                        }
                    }

                    numerator /= n;
                    left_std /= n;
                    right_std /= n;
                    res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;

                }
                if (res_ncc > max_ncc[y][x])
                {
                    max_ncc[y][x] = res_ncc;

                    depth.at<uchar>(y, x) = (uchar)(offset);
                }
            }
        }
    }*/
    for (int y = 0; y < height; y++)
    {

        for (int x = 0; x < width; x++)
        {

            for (int offset = 1; offset <= max_offset; offset++)
            {

                if(x-offset<0)
                    continue;
//                double left_mean = 0, right_mean = 0;
//                double left_std = 0, right_std= 0;
//                double numerator = 0;
//                double res_ncc = 0;

                double left_mean[3]={0,0,0}, right_mean[3]={0,0,0};
                double left_std[3]={0,0,0}, right_std[3]={0,0,0};
                double numerator[3]={0,0,0};
                double res_ncc[3]={0,0,0};


                double res_ncc_mean = 0;
                int n = 0;


                for (int i = y-kernel_size; i <= y+ kernel_size; i++)
                {
                    if(i<0||i>height)
                        continue;
                    for (int j = x-kernel_size; j <= x+kernel_size; j++)
                    {
                        if(j<0||j>width||(j-offset<0))
                            continue;
                        for (int c = 0; c < in1.channels();++c)
                        {

                            n++;
                            left_mean[c]+= in1.at<Vec3b>(i,j)[c];
                            right_mean[c]+= in2.at<Vec3b>(i,j-offset)[c];

                           /* left_mean[c]+= left_cost[0].at<float>(i, j);

                            right_mean[c]+= right_cost[0].at<float>(i, j-offset);*/

                        }

                    }
                }

                for (int c = 0; c < in1.channels();++c)
                {
                    left_mean[c]= left_mean[c]/n;
                    right_mean[c]= right_mean[c]/n;
                }


                for (int i = y-kernel_size; i <= y+ kernel_size; i++)
                {
                    if(i<0||i>height)
                        continue;
                    for (int j = x-kernel_size; j <= x+kernel_size; j++)
                    {
                        if(j<0||j>width)
                            continue;
                        for (int c = 0; c < in1.channels();++c)
                        {
                            left_std[c]+= pow(in1.at<Vec3b>(i, j)[c]- left_mean[c], 2);
                            right_std[c] += pow(in2.at<Vec3b>(i, j-offset)[c] - right_mean[c], 2);
                            numerator[c] += (in1.at<Vec3b>(i, j)[c] - left_mean[c]) * (in2.at<Vec3b>(i, j-offset)[c] - right_mean[c]);

                        }

                    }
                }


                for (int c = 0; c < in1.channels();++c)
                {
                    numerator[c]/= n;
                    left_std[c] /= n;
                    right_std[c] /= n;
                    res_ncc[c] = numerator[c]/ (sqrt(left_std[c]) * sqrt(right_std[c])) / n;

                }



                res_ncc_mean = (res_ncc[0]+res_ncc[1]+res_ncc[2])/3;

                if (res_ncc_mean > max_ncc[y][x])
                {
                    max_ncc[y][x] = res_ncc_mean;

                    depth.at<uchar>(y, x) = (uchar)(offset);
                }
            }

        }

    }






    return depth;
}


Mat ASW(Mat in1, Mat in2, string type)
{

    int width = in1.size().width;
    int height = in1.size().height;
    double k=3,gamma_a = 0.12, gamma_c = 25, gamma_g = 36;

    Mat depth(height, width, CV_8UC1);
    Mat depth_right(height, width, CV_8UC1);
    Mat depth_err(height, width, CV_8UC1);

    vector< vector<double> > min_asw_left,min_asw_right;
    Mat left = bgr_to_grey(in1);
    Mat right = bgr_to_grey(in2);
    vector< vector< vector< pair<double,int> > > > value_asw_left(
            height,vector< vector< pair<double,int> > >(
                    width)),value_asw_right(
            height,vector< vector< pair<double,int> > >(
                    width));

    vector<Mat> left_cost = getAmpAndAngle(in1,0);
    vector<Mat> right_cost = getAmpAndAngle(in2,0);


    for (int i = 0; i < height; ++i)
    {
        vector<double> tmp(width, numeric_limits<double>::max());
        min_asw_left.push_back(tmp);
        min_asw_right.push_back(tmp);

    }

    /*for (int offset = 0; offset <= max_offset; offset++)
    {
        Mat tmp(height, width, 0);

        if (type == "left")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < offset; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
                }

                for (int x = offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);//相当于右移
                }
            }
        }
        for (int y = 0; y < height; y++)
        {
            Vec3b* data_left = in1.ptr<Vec3b>(y);
            Vec3b* data_right = in2.ptr<Vec3b>(y);

//            Vec3b* grey_data_left = left.ptr<Vec3b>(y);
//            Vec3b* grey_data_right = right.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {

                int start_x = max(0, x - kernel_size);
                int start_y = max(0, y - kernel_size);
                int end_x = min(width - 1, x + kernel_size);
                int end_y = min(height - 1, y + kernel_size);

                int sum_e = 0;
                float e = 0,E=0, center_color = 0;
                double delta_c1=0, delta_c2=0;

                if (type == "left")
                {

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {

                            if(i<0||i>height||j<0||j>width||(i == y&&j == x))
                                continue;
                            for (int c = 0; c < in1.channels();++c)
                            {
                                delta_c1+= fabs(in1.at<Vec3b>(i, j)[c] - data_left[x][c]);
                                delta_c2+= fabs(in2.at<Vec3b>(i, j)[c] - data_right[x][c]);//指针访问速度快一点
                            }
                            delta_c1 = delta_c1 / 3;
                            delta_c2 = delta_c2 / 3;
                            double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
                            double w1 = exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                            double w2 = exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
                            for (int c = 0; c < in1.channels(); ++c)
                            {
                                float angle = fabs(left_cost[1].at<Vec3b>(i, j)[c] - right_cost[1].at<Vec3b>(i, j )[c]);
                                if (!(angle >= 0 && angle <= 180))
                                    angle = 2 * 180 - angle;
                                e += gamma_a* fabs(left_cost[0].at<Vec3b>(i, j)[c] - right_cost[0].at<Vec3b>(i, j )[c])+angle ;

                            }
                            sum_e += w1 * w2 * e;
                        }
                    }
                    center_color = sqrt(pow(data_left[x][0] - data_right[x - offset][0], 2) +
                                        pow(data_left[x][1] - data_right[x - offset][1], 2) +
                                        pow(data_left[x][2] - data_right[x - offset][2], 2));
                    //center_color = sqrt(pow(grey_data_left[x][0] - grey_data_right[x - offset][0], 2));//差距太大
                    E = k*sum_e+ center_color;

                }

                // smaller ASW found
                if (E < min_ssd[y][x])
                {
                    min_ssd[y][x] = E;
                    // for better visualization
                    depth.at<uchar>(y, x) = (uchar)(offset);
                }
            }
        }
    }
*/

    clock_t timest = clock();
        for (int y = 0; y < height; y++)
        {

            for (int x = 0; x < width; x++)
            {

                for (int offset = 0; offset <= max_offset; offset++)
                {
                    if(x-offset<0)
                    {
                        continue;
                    }


                    float sum_e = 0;
                    double E=0;
                    double numerator = 0;
                    double denominator = 0;

                    for (int i = y-kernel_size; i <= y+ kernel_size; i+=2)
                    {
                        if(i<0||i>height-1)
                            continue;
                        for (int j = x-kernel_size; j <= x+kernel_size; j+=2)
                        {
                            float e = 0;
                            double delta_c1=0, delta_c2=0;
                            if(j>width-1||j-offset<0)
                                continue;
                            for (int c = 0; c < in1.channels();++c)
                            {
                                delta_c1+= fabs(in1.at<Vec3b>(i, j)[c] - in1.at<Vec3b>(y, x)[c]);
                                delta_c2+= fabs(in2.at<Vec3b>(i, j-offset)[c] - in2.at<Vec3b>(y, x-offset)[c]);//指针访问速度快一点
                            }
                            delta_c1 = delta_c1 / in1.channels();
                            delta_c2 = delta_c2 / in1.channels();
                            double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
                            double w1 = k*exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                            double w2 = k*exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
                            for (int c = 0; c < in1.channels(); ++c)
                            {
                                float angle = fabs(left_cost[1].at<Vec3f>(i, j)[c] - right_cost[1].at<Vec3f>(i, j - offset)[c]);
                                if (!(angle >= 0 && angle <= 180))
                                    angle = 2 * 180 - angle;
                                e += gamma_a* fabs(left_cost[0].at<Vec3f>(i, j)[c] - right_cost[0].at<Vec3f>(i, j - offset)[c])+angle ;

                            }
                            sum_e += w1 * w2 * e;
                            denominator += w1 * w2;

                        }
                    }
                    for (int i = y-kernel_size+1; i <= y+ kernel_size; i+=2)
                    {
                        if(i<0||i>height-1)
                            continue;
                        for (int j = x-kernel_size+1; j <= x+kernel_size; j+=2)
                        {
                            float e = 0;
                            double delta_c1=0, delta_c2=0;
                            if(j>width-1||j-offset<0)
                                continue;
                            for (int c = 0; c < in1.channels();++c)
                            {
                                delta_c1+= fabs(in1.at<Vec3b>(i, j)[c] - in1.at<Vec3b>(y, x)[c]);
                                delta_c2+= fabs(in2.at<Vec3b>(i, j-offset)[c] - in2.at<Vec3b>(y, x-offset)[c]);//指针访问速度快一点
                            }
                            delta_c1 = delta_c1 / in1.channels();
                            delta_c2 = delta_c2 / in1.channels();
                            double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
                            double w1 = k*exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                            double w2 = k*exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
                            for (int c = 0; c < in1.channels(); ++c)
                            {
                                float angle = fabs(left_cost[1].at<Vec3f>(i, j)[c] - right_cost[1].at<Vec3f>(i, j - offset)[c]);
                                if (!(angle >= 0 && angle <= 180))
                                    angle = 2 * 180 - angle;
                                e += gamma_a* fabs(left_cost[0].at<Vec3f>(i, j)[c] - right_cost[0].at<Vec3f>(i, j - offset)[c])+angle ;

                            }
                            sum_e += w1 * w2 * e;
                            denominator += w1 * w2;


                        }
                    }


                    E = sum_e/denominator;
                    pair<double,int> offset_e ;
                    offset_e = std::make_pair(E,offset);
                    value_asw_left[y][x].push_back(offset_e);
//                    if(offset==max_offset)
//                        std::sort(value_asw_left[y][x].begin(),value_asw_left[y][x].end(),best_to_bad_ordering);
                    if(offset==max_offset||x-offset==0)
                    {
                        if(y==11&&x==121)
                            int u =0;
                        std::sort(value_asw_left[y][x].begin(),value_asw_left[y][x].end(),best_to_bad_ordering);

                        depth.at<uchar>(y, x) = (uchar)(value_asw_left[y][x][0].second);
                        for(int i=0;i<5&&i<=x;i++)
                        {
                            int offset_num =  value_asw_left[y][x][i].second;
                            double sum=0;
                            for (int c = 0; c < in1.channels(); ++c)
                            {
                                sum+=fabs(in1.at<Vec3b>(y, x)[c] - in2.at<Vec3b>(y, x-offset_num)[c]);

                            }
                            if(sum/3 <= 25&&value_asw_left[y][x][i].first-value_asw_left[y][x][0].first<50)
                            {
                                depth.at<uchar>(y, x) = (uchar)(offset_num);
                                break;
                            }

                        }

                    }



//                    if (E < min_asw_left[y][x])
//                    {
//                        min_asw_left[y][x] = E;
//
//                        depth.at<uchar>(y, x) = (uchar)(offset);
//                    }
                }

            }

        }

    cout << "初步计算时间：" << (clock() - timest) / (double)CLOCKS_PER_SEC << endl;




//一致性检测
    clock_t timest_check = clock();
    for (int y = 0; y < height; y++)
    {

        for (int x = 0; x < width; x++)
        {

            for (int offset = 0; offset <= max_offset; offset++)
            {
                if(x+offset>width-1)
                    continue;
                double sum_e = 0;

                double E=0;

                double denominator = 0;


                for (int i = y-kernel_size; i <= y+ kernel_size; i+=2)
                {
                    if(i<0||i>height-1)
                        continue;
                    for (int j = x-kernel_size; j <= x+kernel_size; j+=2)
                    {
                        float e = 0;
                        double delta_c1=0, delta_c2=0;
                        if(j<0||j+offset>width-1)
                            continue;
                        for (int c = 0; c < in1.channels();++c)
                        {
                            delta_c1+= fabs(in2.at<Vec3b>(i, j)[c] - in2.at<Vec3b>(y, x)[c]);
                            delta_c2+= fabs(in1.at<Vec3b>(i, j+offset)[c] - in1.at<Vec3b>(y, x+offset)[c]);//指针访问速度快一点
                        }
                        delta_c1 = delta_c1 / in1.channels();
                        delta_c2 = delta_c2 / in1.channels();
                        double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
                        double w1 = k*exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                        double w2 = k*exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
                        for (int c = 0; c < in1.channels(); ++c)
                        {
                            float angle = fabs(right_cost[1].at<Vec3f>(i, j)[c] - left_cost[1].at<Vec3f>(i, j +offset)[c]);
                            if (!(angle >= 0 && angle <= 180))
                                angle = 2 * 180 - angle;
                            e += gamma_a* fabs(right_cost[0].at<Vec3f>(i, j)[c] - left_cost[0].at<Vec3f>(i, j + offset)[c])+angle ;

                        }
                        sum_e += w1 * w2 * e;
                        denominator += w1 * w2;

                    }
                }
                for (int i = y-kernel_size+1; i <= y+ kernel_size; i+=2)
                {
                    if(i<0||i>height-1)
                        continue;
                    for (int j = x-kernel_size+1; j <= x+kernel_size; j+=2)
                    {
                        float e = 0;
                        double delta_c1=0, delta_c2=0;
                        if(j<0||j+offset>width-1)
                            continue;
                        for (int c = 0; c < in1.channels();++c)
                        {
                            delta_c1+= fabs(in2.at<Vec3b>(i, j)[c] - in2.at<Vec3b>(y, x)[c]);
                            delta_c2+= fabs(in1.at<Vec3b>(i, j+offset)[c] - in1.at<Vec3b>(y, x+offset)[c]);//指针访问速度快一点
                        }
                        delta_c1 = delta_c1 / in1.channels();
                        delta_c2 = delta_c2 / in1.channels();
                        double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
                        double w1 = k*exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                        double w2 = k*exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
                        for (int c = 0; c < in1.channels(); ++c)
                        {
                            float angle = fabs(right_cost[1].at<Vec3f>(i, j)[c] - left_cost[1].at<Vec3f>(i, j +offset)[c]);
                            if (!(angle >= 0 && angle <= 180))
                                angle = 2 * 180 - angle;
                            e += gamma_a* fabs(right_cost[0].at<Vec3f>(i, j)[c] - left_cost[0].at<Vec3f>(i, j + offset)[c])+angle ;

                        }
                        sum_e += w1 * w2 * e;
                        denominator += w1 * w2;

                    }
                }
                E = sum_e/denominator;
                if(x==302)
                    int u =0;

                pair<double,int> offset_e ;
                offset_e = std::make_pair(E,offset);
                value_asw_right[y][x].push_back(offset_e);
                if(offset==max_offset||x+offset==width-1)
                {

                    std::sort(value_asw_right[y][x].begin(),value_asw_right[y][x].end(),best_to_bad_ordering);

                    depth_right.at<uchar>(y, x) = (uchar)(value_asw_right[y][x][0].second);
                    for(int i=0;i<5&&i<=width-1-x;i++)
                    {
                        int offset_num =  value_asw_right[y][x][i].second;
                        double sum=0;
                        for (int c = 0; c < in1.channels(); ++c)
                        {
                            sum+=fabs(in2.at<Vec3b>(y, x)[c] - in1.at<Vec3b>(y, x+offset_num)[c]);

                        }
                        if(sum/3 <= 25&&value_asw_left[y][x][i].first-value_asw_left[y][x][0].first<50)
                        {
                            depth_right.at<uchar>(y, x) = (uchar)(offset_num);
                            break;
                        }

                    }

                }

//                if (E < min_asw_right[y][x])
//                {
//                    min_asw_right[y][x] = E;
//
//                    depth_right.at<uchar>(y, x) = (uchar)(offset);
//                }
            }

        }

    }


    depth_err = CheckDepth(depth,depth_right);
    cout << "一致性检测计算时间：" << (clock() - timest_check) / (double)CLOCKS_PER_SEC << endl;

    imwrite("/home/quinlan/Learn/StereoMatch/dataset/error_depth.png", depth_err);
    imwrite("/home/quinlan/Learn/StereoMatch/dataset/initial.png", depth);


    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if(depth_err.at<uchar>(y,x)!=0)
                continue;
            int l = -1;
            for (int i = x; i >= 0 ; --i)
            {
                if(depth_err.at<uchar>(y,i)!=0)
                {
                    l = depth.at<uchar>(y,i);
                    break;
                }

            }
            int r = -1;
            for (int i = x; i < width ; ++i)
            {
                if(depth_err.at<uchar>(y,i)!=0)
                {
                    r = depth.at<uchar>(y,i);
                    break;
                }
            }
            if(l!= (-1) && r!= (-1))
            {
                depth.at<uchar>(y,x) = (l < r ?l:r);
            }
            else if(l== -1)
            {
                depth.at<uchar>(y,x) = r;
            }
            else
                depth.at<uchar>(y,x) = l;

        }
    }

    return depth;
}

Mat ncc(Mat in1, Mat in2, string type, bool add_constant = false)
{
    int width = in1.size().width;
    int height = in1.size().height;


    Mat left = bgr_to_grey(in1);
    Mat right = bgr_to_grey(in2);

    if (add_constant)
    {
        right += 10;
    }

    Mat depth(height, width, 0);
    vector< vector<double> > max_ncc; // store max NCC value

    for (int i = 0; i < height; ++i)
    {
        vector<double> tmp(width, -2);
        max_ncc.push_back(tmp);
    }

    for (int offset = 1; offset <= max_offset; offset++)
    {
        Mat tmp(height, width, 0);
        // shift image depend on type to save calculation time
        if (type == "left")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < offset; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
                }

                for (int x = offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);
                }
            }
        }
        else if (type == "right")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width - offset; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x + offset);
                }

                for (int x = width - offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x);
                }
            }
        }
        else
        {
            Mat tmp(0, 0, 0);
            return tmp;
        }

        // calculate each pixel's NCC value
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int start_x = max(0, x - kernel_size);
                int start_y = max(0, y - kernel_size);
                int end_x = min(width - 1, x + kernel_size);
                int end_y = min(height - 1, y + kernel_size);
                double n = (end_y - start_y) * (end_x - start_x);
               // int n = 0;
                double res_ncc = 0;

                if(y==52&&x==86&&offset==20)
                    int k = 0;

                if (type == "left")
                {
                    double left_mean = 0, right_mean = 0;
                    double left_std = 0, right_std = 0;
                    double numerator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {

                            int l = left.at<uchar>(i, j);
                            left_mean += left.at<uchar>(i, j);
                            int r = tmp.at<uchar>(i, j);
                            right_mean += tmp.at<uchar>(i, j);

                        }
                    }

                    left_mean /= n;
                    right_mean /= n;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_std += pow(left.at<uchar>(i, j) - left_mean, 2);
                            right_std += pow(tmp.at<uchar>(i, j) - right_mean, 2);
                            numerator += (left.at<uchar>(i, j) - left_mean) * (tmp.at<uchar>(i, j) - right_mean);
                        }
                    }

                    numerator /= n;
                    left_std /= n;
                    right_std /= n;
                    res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;
                }
                else
                {
                    double left_mean = 0, right_mean = 0;
                    double left_std = 0, right_std = 0;
                    double numerator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_mean += tmp.at<uchar>(i, j);
                            right_mean += right.at<uchar>(i, j);
                        }
                    }

                    left_mean /= n;
                    right_mean /= n;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_std += pow(tmp.at<uchar>(i, j) - left_mean, 2);
                            right_std += pow(right.at<uchar>(i, j) - right_mean, 2);
                            numerator += (tmp.at<uchar>(i, j) - left_mean) * (right.at<uchar>(i, j) - right_mean);
                        }
                    }

                    numerator /= n;
                    left_std /= n;
                    right_std /= n;
                    res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;
                }

                // greater NCC value found
                if (res_ncc > max_ncc[y][x])
                {

                    max_ncc[y][x] = res_ncc;
                    // for better visualization
                    depth.at<uchar>(y, x) = (uchar)(offset);
                }
            }
        }
    }


    return depth;
}


Mat
asw(Mat in1, Mat in2, string type)
{
    int width = in1.size().width;
    int height = in1.size().height;



    double k = 3, gamma_c = 7, gamma_g = 36; // ASW parameters

    Mat depth(height, width, 0);
    vector< vector<double> > min_asw; // store min ASW value


    Mat left = bgr_to_grey(in1);
    Mat right = bgr_to_grey(in2);

    for (int i = 0; i < height; ++i)
    {
        vector<double> tmp(width, numeric_limits<double>::max());
        min_asw.push_back(tmp);
    }

    for (int offset = 1; offset <= max_offset; offset++)
    {
        Mat tmp(height, width, 0);
        // shift image depend on type to save calculation time
        if (type == "left")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < offset; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
                }

                for (int x = offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);
                }
            }
        }
        else if (type == "right")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width - offset; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x + offset);
                }

                for (int x = width - offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x);
                }
            }
        }
        else
        {
            Mat tmp(0, 0, 0);
            return tmp;
        }

        // calculate each pixel's ASW value
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {

                int start_x = max(0, x - kernel_size);
                int start_y = max(0, y - kernel_size);
                int end_x = min(width - 1, x + kernel_size);
                int end_y = min(height - 1, y + kernel_size);
                double E = 0;

                if (type == "left")
                {
                    double numerator = 0;
                    double denominator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            double delta_c1 = fabs(left.at<uchar>(i, j) - left.at<uchar>(y, x));
                            double delta_c2 = fabs(tmp.at<uchar>(i, j) - tmp.at<uchar>(y, x));
                            double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
                            double w1 = k * exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                            double w2 = k * exp(-(delta_c2 / gamma_c + delta_g / gamma_g));


                            numerator += w1 * w2 * fabs(left.at<uchar>(i, j) - tmp.at<uchar>(i, j));
                            denominator += w1 * w2;
                        }
                    }

                    E = numerator / denominator;
                }
                else
                {
                    double numerator = 0;
                    double denominator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            double delta_c1 = fabs(right.at<uchar>(i, j) - right.at<uchar>(y, x));
                            double delta_c2 = fabs(tmp.at<uchar>(i, j) - tmp.at<uchar>(y, x));
                            double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
                            double w1 = k * exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                            double w2 = k * exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
                            numerator += w1 * w2 * fabs(right.at<uchar>(i, j) - tmp.at<uchar>(i, j));
                            denominator += w1 * w2;
                        }
                    }

                    E = numerator / denominator;
                }

                // smaller ASW found
                if (E < min_asw[y][x])
                {
                    min_asw[y][x] = E;
                    // for better visualization
                    depth.at<uchar>(y, x) = (uchar)(offset);
                }
            }
        }
    }

    return depth;
}