
#ifndef STEREOMATCH_STEREOMATCHING_H
#define STEREOMATCH_STEREOMATCHING_H

#include<omp.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <cstdlib>
#include "progress_display.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
Mat
asw(Mat in1, Mat in2, string type);


Mat bgr_to_grey(const Mat& bgr);


vector<Mat> getAmpAndAngle(Mat in1,int type);


bool best_to_bad_ordering(const pair<double,int> a, const pair<double,int> b);

Mat CheckDepth(const Mat &depth_left,const Mat &depth_right);


void fliter_err(Mat & depth,Mat in1,int num_iter);



void fliter_err(Mat & depth,Mat &depth_err,Mat in1,vector< vector< vector< pair<double,int> > > > best_arry,int num_iter);



void consistent_check(int width, int height, Mat &depth, Mat depth_err);


Mat ASW(Mat in1, Mat in2, string type);


#endif //STEREOMATCH_PROGRESS_H
