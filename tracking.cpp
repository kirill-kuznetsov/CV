#include <stdio.h>
#include "opencv2\opencv.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <vector>
#include <math.h>

#define MAX_COUNT 100      
#define PI 3.1415     
#include <locale>

using namespace cv;

inline void detect(VideoCapture &capture, int threshold, bool nonmaxSupression, int winSize, int maxLevel, int cornerCount, int iterations, double epsilon){
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



	TermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, iterations, epsilon);

	//capture a frame form cam        
	if (!capture.isOpened())
		return;


	CvSize cv_size = cvSize(winSize, winSize);

	//Routine Start     
	while (1) {
		//break        
		if (cvWaitKey(30) >= 0)
			break;

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
		cornerSubPix(grayFrame, cornersA, cv_size, cvSize(-1, -1), criteria);

		calcOpticalFlowPyrLK(grayFrame, nextGrayFrame, cornersA, cornersB, features_found, feature_errors,
			cv_size, maxLevel, criteria, 0);

		IplImage image = grayFrame;
		for (int i = 0; i < cornerCount; ++i)
		{
			cvLine(&image, cvPoint(cornersA[i].x, cornersA[i].y), cvPoint(cornersB[i].x, cornersA[i].y), CV_RGB(0, 255, 0), 2);
		}
		Mat result(&image);

		imshow("Origin", result);


		
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
	int iterations = 15;
	double epsilon = 0.3;


	//Video Load     
	VideoCapture capture("D:\\Movies\\Experiment\\123.avi");
//	VideoCapture capture("C:\\Users\\kkuznets\\Desktop\\CV\\Experiment\\4.avi");

	for (maxLevel = 1; maxLevel <= 3; maxLevel++)
	{
		for (iterations = 1; iterations < 30; iterations++)
		{
			for (winSize = 5; winSize < 40; winSize += 5)
			{
				clock_t timer = clock();
				detect(capture, threshold, nonmaxSupression, winSize, maxLevel, cornerCount, iterations, epsilon);
				timer -= clock();
				double timerSec = timer / CLOCKS_PER_SEC;
				printf("Level: %d\n, Iterations: %d\n, WinSize: %d\n, Time: %f\n\n", maxLevel, iterations, winSize, timerSec);
			}
		}
	}
	
}

