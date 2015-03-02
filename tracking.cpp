#include <stdio.h>
#include "opencv2\opencv.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <vector>
#include <math.h>

#define MAX_COUNT 100      
#define PI 3.1415     

using namespace cv;

inline void detect(VideoCapture &capture, int threshold, bool nonmaxSupression, int winSize, int maxLevel, int cornerCount, TermCriteria &criteria){
	//T, T-1 image
	Mat frame;
	Mat nextFrame;

	//Optical Image     
	Mat grayFrame;
	Mat nextGrayFrame;

	//Window     
	namedWindow("Origin", 1);

	vector<KeyPoint> keypoints;
	vector<Point2f> cornersA;
	vector<Point2f> cornersB;

	vector<uchar> features_found;
	vector<float> feature_errors;
	features_found.reserve(MAX_COUNT);
	feature_errors.reserve(MAX_COUNT);


	//capture a frame form cam        
	if (!capture.isOpened())
		return;

	//Routine Start     
	while (1) {
		capture >> frame;
		capture >> nextFrame;

		//RGB to Gray for Optical Flow     
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		cvtColor(nextFrame, nextGrayFrame, CV_BGR2GRAY);

		FAST(frame, keypoints, threshold, nonmaxSupression);

		for (vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); it++)
		{
			cornersA.push_back(it->pt);
		}
		cornerSubPix(grayFrame, cornersA, cvSize(winSize, winSize), cvSize(-1, -1), criteria);

		features_found.clear();
		feature_errors.clear();

		calcOpticalFlowPyrLK(grayFrame, nextGrayFrame, cornersA, cornersB, features_found, feature_errors,
			cvSize(winSize, winSize), maxLevel, criteria, 0);

		IplImage image = frame;
		for (int i = 0; i < cornerCount; ++i)
		{
			cvLine(&image, cvPoint(cornersA[i].x, cornersA[i].y), cvPoint(cornersB[i].x, cornersA[i].y), CV_RGB(0, 255, 0), 2);
		}
		Mat result(&image);
		imshow("Origin", result);


		//break        
		if (cvWaitKey(1) >= 0)
			break;
	}

	//release capture point        
	capture.release();
	//close the window        
	destroyWindow("Origin");
}

void main()
{
	int cornerCount = MAX_COUNT;

	//for FAST
	int threshold = 39;// 0 - 39
	bool nonmaxSupression = true;

	//for optical flow
	int maxLevel = 3;
	int winSize = 30;
	TermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 15, 0.3);


	//Video Load     
	//VideoCapture capture("D:\\Movies\\Experiment\\123.avi");
	VideoCapture capture("C:\\Users\\kkuznets\\Desktop\\CV\\Experiment\\4.avi");

	detect(capture, threshold, nonmaxSupression, winSize, maxLevel, cornerCount, criteria);
	
}

