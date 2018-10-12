#include <ctime>

#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <string>
#include <ctime>
#include <cstdlib>

#include "StereoMatching.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const string img_name[] = {
        "Aloe", "Baby1", "Baby2", "Baby3",
        "Bowling1", "Bowling2", "Cloth1", "Cloth2",
        "Cloth3", "Cloth4", "Flowerpots",
        "Lampshade1", "Lampshade2", "Midd1", "Midd2",
        "Monopoly", "Plastic", "Rocks1", "Rocks2",
        "Wood1", "Wood2"
};


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


int
main(void)
{

    /*test_image_quality("SSD");
    test_image_quality("NCC");
    test_image_quality("SSD_CONSTANT");
    test_image_quality("NCC_CONSTANT");
    test_image_quality("ASW");
*/



    {
        Mat left = imread("/home/quinlan/Learn/StereoMatch/dataset/im0.png");
        Mat right = imread("/home/quinlan/Learn/StereoMatch/dataset/im1.png");

        Mat descriptors_1, descriptors_2,descriptors_3, descriptors_4;
        std::vector< DMatch > tri_matches;
        std::vector< KeyPoint > keypts1,keypts2;

        Ptr<SiftFeatureDetector> detector = SIFT::create();
        detector->detect(left, keypts1);
        detector->detect(right, keypts2);

        Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();


        extractor->compute(left, keypts1, descriptors_1);
        extractor->compute(right, keypts2, descriptors_2);
        Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-L1" );
        matcher->match ( descriptors_1, descriptors_2, tri_matches );



        Mat img_matches;
        drawMatches(left, keypts1, right, keypts2,
                    tri_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imwrite("../fliter_matches.jpg", img_matches);


        string disp1_name = "/home/quinlan/Learn/StereoMatch/dataset/wwamp_angle_center_color44teddy.png";
        //string disp5_name = "C:/Users/Administrator/Desktop/DA/dataset/disp_row3col3.png";

        cout << "creating " << disp1_name << endl;
        clock_t timest = clock();
        //Mat disp = ssd(left, right, "left");
        Mat disp = ASW(left, right, "left");
        cout << "计算时间：" <<  (clock() - timest) / (double)CLOCKS_PER_SEC << endl;
        imwrite("/home/quinlan/Learn/StereoMatch/dataset/abs_disp.png", disp);

        normalize(disp, disp, 0, 255, NORM_MINMAX, CV_8UC1);
        imshow("tu", disp);
        imwrite(disp1_name, disp);
        waitKey(0);

        /*cout << "creating " << disp5_name << endl;
        imwrite(disp5_name, ssd(left, right, "right"));*/
    }
/*

	// add intensity value to right eye image
	for (auto name : img_name)
	{
		Mat left = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/view1.png");
		Mat right = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/view5.png");
		string disp1_name = "C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp1_SSD_CONSTANT.png";
		string disp5_name = "C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp5_SSD_CONSTANT.png";

		cout << "creating " << disp1_name << endl;
		imwrite(disp1_name, ssd(left, right, "left", true));

		cout << "creating " << disp5_name << endl;
		imwrite(disp5_name, ssd(left, right, "right", true));
	}

	// matching cost function: Normalized Cross Correlation
	for (auto name : img_name)
	{
		Mat left = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/view1.png");
		Mat right = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/view5.png");
		string disp1_name = "C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp1_NCC.png";
		string disp5_name = "C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp5_NCC.png";

		cout << "creating " << disp1_name << endl;
		imwrite(disp1_name, ncc(left, right, "left"));

		cout << "creating " << disp5_name << endl;
		imwrite(disp5_name, ncc(left, right, "right"));
	}

	// add intensity value to right eye image
	for (auto name : img_name)
	{
		Mat left = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/view1.png");
		Mat right = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/view5.png");
		string disp1_name = "C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp1_NCC_CONSTANT.png";
		string disp5_name = "C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp5_NCC_CONSTANT.png";

		cout << "creating " << disp1_name << endl;
		imwrite(disp1_name, ncc(left, right, "left", true));

		cout << "creating " << disp5_name << endl;
		imwrite(disp5_name, ncc(left, right, "right", true));
	}

	// matching cost function: Adaptive Support Window
	for (auto name : img_name)
	{
		Mat left = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/view1.png");
		Mat right = imread("C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/ALL-2views/" + name + "/view5.png");
		string disp1_name = "C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp1_ASW.png";
		string disp5_name = "C:/Users/Administrator/Desktop/DA/stereo-matching-master/stereo-matching-master/result/" + name + "_disp5_ASW.png";

		cout << "creating " << disp1_name << endl;
		imwrite(disp1_name, asw(left, right, "left"));

		cout << "creating " << disp5_name << endl;
		imwrite(disp5_name, asw(left, right, "right"));
	}
	*/
    return 0;
}