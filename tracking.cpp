#include <stdio.h>
#include <vector>
#include <math.h> 
#include <iostream>
#include <algorithm>
#include <iterator>


#include "opencv2\opencv.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
   
#include <locale>
#include <iostream>
#include <fstream>

using namespace std;

using namespace cv;


#define PRINT_TRACKS true;
#define PRINT_LENGTHS false;
#define SHOW_VIDEO true;
#define MAX_COUNT 300;      
#define PI 3.1415; 
#define PRINT_VALUES false

int trackNumber = 0;

inline int printTrace(vector<vector<Point2f>> cornersMatrix, vector<bool> ended, vector<size_t> length, ofstream &myfile){
//	std::ofstream myfile = file;
//	myfile.open("C:\\Users\\Кирилл\\Desktop\\traces.txt");
	int frameNumber = 0;
	bool printTracks = PRINT_TRACKS;
	if (printTracks)
	{
		int frame = 0;
		for (vector<Point2f> corners : cornersMatrix){
			int feature_num = 0;
			for (Point2f corner : corners){
				myfile << frame << " " << feature_num << " " << 1 << " " << corner.x << "\n";
				myfile << frame << " " << feature_num << " " << 2 << " " << corner.y << "\n";
				feature_num++;
			}
			frame++;
		}
		
	}

	int avgLength = 0;
	if (PRINT_VALUES){
		int notEnded = 0;
		int size = ended.size();
		for (int i = 0; i < size; i++)
		{
			//		myfile << ended[i] << ",";	
			if (!ended[i])
			{
				notEnded++;
			}
		}
		myfile << "\n";
		myfile << "Tracks: " << ended.size() << "\n";
		myfile << "Tracks at the end: " << notEnded << "\n";
		bool printLengths = PRINT_LENGTHS;
		if (printLengths)
		{
			myfile << "Lengths: ";
		}
		for (int i = 0; i < size; i++)
		{
			avgLength += length[i];
			if (printLengths)
			{
				myfile << length[i] << " ,";
			}
		}
		myfile << "\n";
		avgLength = avgLength / size;
		int lifeTime = avgLength * 100 / cornersMatrix.size();
		myfile << "Avg length: " << avgLength << "\n";
	}
	return avgLength;
}
inline int detect(char* file, int threshold, bool nonmaxSupression, int winSize, int maxLevel, int cornerCount, int iterations, double epsilon, ofstream &myfile){
	//Optical Image     
	Mat prevGrayFrame, grayFrame,  image;


	vector<KeyPoint> keypoints;
	vector<Point2f> cornersA;
	vector<Point2f> cornersB;

	TermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, iterations, epsilon);

	VideoCapture capture(file);

	//capture a frame form cam        
	if (!capture.isOpened())
		return 0;

	CvSize windowSize = cvSize(winSize, winSize);
	int subPixSide = winSize < 11 ? winSize : winSize / 3 >= 11 ? winSize / 3 : 11;
	CvSize subPixSize = cvSize(subPixSide, subPixSide);


	bool needToInit = true;
	

	vector<vector<Point2f>> tracesMatrix;

	vector<bool> ended(0);
	vector<size_t> length(0);
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
			cornerSubPix(grayFrame, cornersB, subPixSize, Size(-1, -1), criteria);
			if (tracesMatrix.size() == 0)
			{
				//initial track points
				trackNumber = cornersB.size();
				ended.resize(trackNumber);
				length.resize(trackNumber);
				vector<Point2f> trackStarts(trackNumber);
				for (int i = 0; i < trackNumber; i++)
				{
					trackStarts[i] = cornersB[i];
					ended[i] = false;
					length[i] = 1;
				}
				tracesMatrix.push_back(trackStarts);
			}

			size_t i, k, j;

			vector<Point2f> trackStep(trackNumber);
			vector<bool> visited = ended;
			for (i = k = 0; i < cornersB.size(); i++)
			{
				//looking for next unfinished track
				for (j = i; j < trackNumber && visited[j]; j++);
				if (j >= trackNumber)
				{
//					std::cout << "New track";
				}
				if (!status[i])
				{
					if (j < trackNumber)
					{
						ended[j] = true;
						visited[j] = true;
						trackStep[j] = (Point2f)NULL;						
					}
					continue;
				}
				if (j < trackNumber)
				{
					trackStep[j] = cornersB[i];
					length[j] += 1;
					visited[j] = true;
				}
				cornersB[k++] = cornersB[i];
				circle(image, cornersB[i], 3, Scalar(0, 255, 0), -1, 8);
//				cvLine(&(IplImage)image, cvPoint(cornersA[i].x, cornersA[i].y), cvPoint(cornersB[i].x, cornersB[i].y), Scalar(0, 255, 0), 2);
			}

			tracesMatrix.push_back(trackStep);

			cornersB.resize(k);
		}

		needToInit = false;
		bool showVideo = SHOW_VIDEO;
		if (showVideo)
			imshow("Origin", image);

		char c = (char)cvWaitKey(10);
		if (c == 27)
			break;

		std::swap(cornersB, cornersA);
		cv::swap(prevGrayFrame, grayFrame);
	}

	//release capture point     
	capture.release();
	
	return printTrace(tracesMatrix, ended, length, myfile);
}

void main()
{

	int cornerCount = MAX_COUNT;

	//for FAST
	int threshold = 40;// 0 - 39
	bool nonmaxSupression = true;

	//for optical flow
	int maxLevel = 10;
	int winSize = 30;
	int iterations = 30;
	double epsilon = 0.003;


	//Video Load     
//	char* file = "C:\\Users\\kkuznets\\Desktop\\CV\\Experiment\\4.avi";

	char* file = "D:\\Movies\\Experiment\\123.avi";


	//Window     
	namedWindow("Origin", 1);
	std::ofstream myfile;
	myfile.open("C:\\Users\\Кирилл\\Desktop\\exper_result.txt");

	string bestTimeCoord, bestLengthCoord, worseTimeCoord, worseLengthCoord;
	int bestLength = 0, bestTimeLength = 0, worseLength = 1000, worseTimeLength = 1000;
	double bestTime = 1000, bestLengthTime = 1000, worseTime = 0, worseLengthTime = 0;
//	for (threshold = 6; threshold < 256; threshold += 25)
//	{
//		for (maxLevel = 1; maxLevel <= 3; maxLevel++)
//		{
//			for (iterations = 3; iterations <= 30; iterations += 3)
//			{
//				for (winSize = 10; winSize <= 30; winSize += 5)
//				{
					clock_t timer = clock();
					int length = detect(file, threshold, nonmaxSupression, winSize, maxLevel, cornerCount, iterations, epsilon, myfile);
					timer = clock() - timer;
					double timerSec = (double)timer / CLOCKS_PER_SEC;
					printf("T: %d, L: %d,I: %d, WS: %d, Time: %f\n", threshold, maxLevel, iterations, winSize, timerSec);
					if (PRINT_VALUES){
						if (timerSec < bestTime)
						{
							bestTime = timerSec;
							bestTimeLength = length;
							ostringstream o;
							o << "T:" << threshold << "L:" << maxLevel << ",I: " << iterations << ", WS: " << winSize << "\n";
							bestTimeCoord = o.str();

						}
						if (length > bestLength)
						{
							bestLength = length;
							bestLengthTime = timerSec;
							ostringstream o;
							o << "T:" << threshold << "L:" << maxLevel << ",I: " << iterations << ", WS: " << winSize << "\n";
							bestLengthCoord = o.str();
						}
						if (timerSec > worseTime)
						{
							worseTime = timerSec;
							worseTimeLength = length;
							ostringstream o;
							o << "T:" << threshold << "L:" << maxLevel << ",I: " << iterations << ", WS: " << winSize << "\n";
							worseTimeCoord = o.str();

						}
						if (length < worseLength)
						{
							worseLength = length;
							worseLengthTime = timerSec;
							ostringstream o;
							o << "T:" << threshold << "L:" << maxLevel << ",I: " << iterations << ", WS: " << winSize << "\n";
							worseLengthCoord = o.str();
						}
						myfile << "T:" << threshold << "L:" << maxLevel << ",I: " << iterations << ", WS: " << winSize << "\nTime : " << timerSec << "\n";
					}
//				}
//			}
//		}
//	}
					if (PRINT_VALUES){
						myfile << "Best time: " << bestTime << ", Avg track length: " << bestTimeLength << ". " << bestTimeCoord;
						myfile << "Best length: " << bestLength << ", Time: " << bestLengthTime << ". " << bestLengthCoord;
						myfile << "Worse time: " << worseTime << ", Avg track length: " << worseTimeLength << ". " << worseTimeCoord;
						myfile << "Worse length: " << worseLength << ", Time: " << worseLengthTime << ". " << worseLengthCoord;
					}
	myfile.close();

	waitKey(0);
	//close the window        
	destroyWindow("Origin");
	
}