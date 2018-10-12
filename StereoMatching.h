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
int max_offset = 59;
int kernel_size = 4; // window size

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

vector<Mat> getAmpAndAngle(Mat in1)
{
    int width = in1.size().width;
    int height = in1.size().height;
    Mat Amp_mat(height, width, CV_8UC3);
    Mat Angle_mat(height, width, CV_8UC3);
    vector<Mat> AmpAndAngle;

    for (int k = 0; k < height; ++k)
    {
        if (k < 15||k>height-15)
            continue;
        Vec3b* data_left = in1.ptr<Vec3b>(k);
        Vec3b* data_left_bottom = in1.ptr<Vec3b>(k + 1);
        Vec3b* data_left_top = in1.ptr<Vec3b>(k - 1);

        Vec3b* data_Amp_mat = Amp_mat.ptr<Vec3b>(k);
        Vec3b* data_Angle_mat = Angle_mat.ptr<Vec3b>(k);

        float Amp[3], Angle[3];
        for (int l = 0; l < width; ++l)
        {
            if (l < 15 || l>width-15)
                continue;
            for (int c = 0; c < in1.channels(); ++c)
            {
                float g_x = data_left[l + 1][c] - data_left[l - 1][c];
                float g_y = data_left_bottom[l][c] - data_left_top[l][c];

                data_Amp_mat[l][c] = sqrt(g_x*g_x + g_y*g_y);
                float angle = (atan2f(g_y, g_x) * 180) / pi;
                data_Angle_mat[l][c] = angle;

            }
        }
    }
    AmpAndAngle.push_back(Amp_mat);
    AmpAndAngle.push_back(Angle_mat);
    return AmpAndAngle;

}


Mat
ASW(Mat in1, Mat in2, string type, bool add_constant = false)
{
    int width = in1.size().width;
    int height = in1.size().height;
    double k=0.008,gamma_a = 0.12,sigma =30,  gamma_c = 7, gamma_g = 36;
    clock_t timest = clock();
    vector<Mat> left_cost = getAmpAndAngle(in1);
    vector<Mat> right_cost = getAmpAndAngle(in2);
    cout << "梯度计算时间：" << (clock() - timest) / (double)CLOCKS_PER_SEC << endl;

    Mat depth(height, width, CV_8UC1);
    vector< vector<int> > min_ssd; // store min SSD values
    Mat left = bgr_to_grey(in1);
    Mat right = bgr_to_grey(in2);



    if (add_constant)
    {
        right += 10;
    }

    for (int i = 0; i < height; ++i)
    {
        vector<int> tmp(width, numeric_limits<int>::max());
        min_ssd.push_back(tmp);
    }

    //for (int offset = 0; offset <= max_offset; offset++)
    {
        Mat tmp(height, width, 0);
        // shift image depend on type to save calculation time
        /*
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

        */
        for (int y = 0; y < height; y++)
        {
            Vec3b* data_left = in1.ptr<Vec3b>(y);
            Vec3b* data_right = in2.ptr<Vec3b>(y);

            Vec3b* grey_data_left = left.ptr<Vec3b>(y);
            Vec3b* grey_data_right = right.ptr<Vec3b>(y);

            for (int x = 0; x < width; x++)
            {
                if (x < 18 || y<18 || x>width - 18 || y>height - 18)
                {
                    depth.at<uchar>(y, x) = 0;
                    continue;
                }
                double numerator = 0;
                double denominator = 0;

                /*int start_x = max(0, x - kernel_size);
                int start_y = max(0, y - kernel_size);
                int end_x = min(width - 1, x + kernel_size);
                int end_y = min(height - 1, y + kernel_size);*/


                for (int offset = 0; offset <= max_offset; offset++)
                {
                    int sum_e = 0;
                    float e = 0,E=0, center_color = 0;
                    double delta_c1=0, delta_c2=0;

                    for (int i = y-kernel_size; i <= y+ kernel_size; i++)
                    {
                        for (int j = x-kernel_size; j <= x+kernel_size; j++)
                        {
                            if (i == y&&j == x)//不算中心像素
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
                                float angle = fabs(left_cost[1].at<Vec3b>(i, j)[c] - right_cost[1].at<Vec3b>(i, j - offset)[c]);
                                if (angle >= 0 && angle <= 180)
                                     angle = angle;
                                else
                                    angle = 2 * 180 - angle;
                                e += gamma_a* fabs(left_cost[0].at<Vec3b>(i, j)[c] - right_cost[0].at<Vec3b>(i, j - offset)[c])+angle ;

                            }
                            sum_e += w1 * w2 * e;
                            //denominator += w1 * w2;


                        }
                    }
                    center_color = sqrt(pow(data_left[x][0] - data_right[x - offset][0], 2) +
                                        pow(data_left[x][1] - data_right[x - offset][1], 2) +
                                        pow(data_left[x][2] - data_right[x - offset][2], 2));
                    //center_color = sqrt(pow(grey_data_left[x][0] - grey_data_right[x - offset][0], 2));//差距太大
                    E = k*sum_e+ center_color;
                    //E = E/denominator;
                   /* float E_2 = E*E;
                    E = E / (E + sigma*sigma);*/

                    if (E < min_ssd[y][x])
                    {
                        min_ssd[y][x] = E;
                        // for better visualization
                        depth.at<uchar>(y, x) = (uchar)(offset);
                    }
                }

            }

        }


/*
// calculate each pixel's SSD value
		for (int y = 0; y < height; y++)//重复累赘计算？
		{
			for (int x = 0; x < width; x++)
			{
				int start_x = max(0, x - kernel_size);
				int start_y = max(0, y - kernel_size);
				int end_x = min(width - 1, x + kernel_size);
				int end_y = min(height - 1, y + kernel_size);
				int sum_sd = 0;

				if (type == "left")
				{
					for (int i = start_y; i <= end_y; i++)
					{
						for (int j = start_x; j <= end_x; j++)
						{
							int delta = abs(left.at<uchar>(i, j) - tmp.at<uchar>(i, j));
							sum_sd += delta * delta;
						}
					}
				}
				else
				{
					for (int i = start_y; i <= end_y; i++)
					{
						for (int j = start_x; j <= end_x; j++)
						{
							int delta = abs(right.at<uchar>(i, j) - tmp.at<uchar>(i, j));
							sum_sd += delta * delta;
						}
					}
				}

				// smaller SSD value found
				if (sum_sd < min_ssd[y][x])
				{
					min_ssd[y][x] = sum_sd;
					// for better visualization
					depth.at<uchar>(y, x) = (uchar)(offset);
				}
			}
		}*/

    }


    return depth;
}

Mat
ncc(Mat in1, Mat in2, string type, bool add_constant = false)
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
                            left_mean += left.at<uchar>(i, j);
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
                    depth.at<uchar>(y, x) = (uchar)(offset * 3);
                }
            }
        }
    }

    return depth;
}

// Adaptive Support Window
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
                    depth.at<uchar>(y, x) = (uchar)(offset * 3);
                }
            }
        }
    }

    return depth;
}