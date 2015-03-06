#include <stdio.h>
#include "opencv2\opencv.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <vector>
#include <math.h>    
#include <locale>
#define MAX_COUNT 300      
#define PI 3.1415 

using namespace cv;

inline void detect(char* file, int threshold, bool nonmaxSupression, int winSize, int maxLevel, int cornerCount, int iterations, double epsilon){
	//Optical Image     
	Mat prevGrayFrame, grayFrame,  image;


	vector<KeyPoint> keypoints;
	vector<Point2f> cornersA;
	vector<Point2f> cornersB;

	TermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, iterations, epsilon);

	VideoCapture capture(file);

	//capture a frame form cam        
	if (!capture.isOpened())
		return;

	CvSize windowSize = cvSize(winSize, winSize);
	int subPixSide = winSize < 11 ? winSize : winSize / 3 >= 11 ? winSize / 3 : 11;
	CvSize subPixSize = cvSize(subPixSide, subPixSide);


	bool needToInit = true;
	//Routine Start     
	while (1) {
		


		Mat frame;
		capture >> frame;

		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, grayFrame, COLOR_BGR2GRAY);

		if (needToInit)
		{
			// automatic initialization
			FAST(grayFrame, keypoints, threshold, nonmaxSupression);
			for (vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); it++)
			{
				cornersA.push_back(it->pt);
			}
			cornerSubPix(grayFrame, cornersA, subPixSize, Size(-1, -1), criteria);
		}
		else if (!cornersA.empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGrayFrame.empty())
				grayFrame.copyTo(prevGrayFrame);
			calcOpticalFlowPyrLK(prevGrayFrame, grayFrame, cornersA, cornersB, status, err, windowSize,
				maxLevel, criteria, 0, 0.001);
			size_t i, k;

			for (i = k = 0; i < cornersB.size(); i++)
			{
				if (!status[i])
					continue;

				cornersB[k++] = cornersB[i];
//				circle(image, cornersB[i], 3, Scalar(0, 255, 0), -1, 8);
				cvLine(&(IplImage)image, cvPoint(cornersA[i].x, cornersA[i].y), cvPoint(cornersB[i].x, cornersB[i].y), Scalar(0, 255, 0), 2);
			}
			cornersB.resize(k);
		}

		needToInit = false;
		imshow("Origin", image);

		char c = (char)cvWaitKey(10);
		if (c == 27)
			break;

		std::swap(cornersB, cornersA);
		cv::swap(prevGrayFrame, grayFrame);
	}

	//release capture point     
	capture.release();

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
	int iterations = 30;
	double epsilon = 0.003;


	//Video Load     
	char* file = "C:\\Users\\kkuznets\\Desktop\\CV\\Experiment\\4.avi";

//	char* file = "D:\\Movies\\Experiment\\123.avi";


	//Window     
		namedWindow("Origin", 1);

//	for (maxLevel = 1; maxLevel <= 3; maxLevel++)
//	{
//		for (iterations = 2; iterations < 30; iterations++)
//		{
//			for (winSize = 5; winSize < 40; winSize += 5)
//			{
				clock_t timer = clock();
				detect(file, threshold, nonmaxSupression, winSize, maxLevel, cornerCount, iterations, epsilon);
				timer -= clock();
				double timerSec = timer / CLOCKS_PER_SEC;
				printf("Level: %d\n, Iterations: %d\n, WinSize: %d\n, Time: %f\n\n", maxLevel, iterations, winSize, timerSec);
//			}
//		}
//	}

	waitKey(0);
	//close the window        
	destroyWindow("Origin");
	
}

