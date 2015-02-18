#include <stdio.h>
#include "opencv2\opencv.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <vector>
#include <math.h>

#define MAX_COUNT 150     
#define DELAY_T 1     
#define PI 3.1415     

using namespace cv;

void main()
{     
	IplImage* image = 0;

	//T, T-1 image     
	IplImage* current_Img = 0;
	IplImage* Old_Img = 0;

	//Optical Image     
	IplImage * imgA = 0;
	IplImage * imgB = 0;

	//Video Load     
	CvCapture * capture = cvCreateFileCapture("D:\\Movies\\Experiment\\123.avi");      

	//Window     
	cvNamedWindow("Origin");
   
	//Optical Flow Variables      
	IplImage * eig_image = 0;
	IplImage * tmp_image = 0;
	int corner_count = MAX_COUNT;
	

	vector<KeyPoint> keypoints;
	CvPoint2D32f* cornersA = new CvPoint2D32f[MAX_COUNT];
	CvPoint2D32f * cornersB = new CvPoint2D32f[MAX_COUNT];

	CvSize img_sz;
	int win_size = 10;

	IplImage* pyrA = 0;
	IplImage* pyrB = 0;

	char features_found[MAX_COUNT];
	float feature_errors[MAX_COUNT];
    
	//Variables for time different video     
	int one_zero = 0;
	int t_delay = 0;

	int threshold = 20;

	//Routine Start     
	while (1) {

		//capture a frame form cam        
		if (cvGrabFrame(capture) == 0)
			break;

		//Image Create     
		if (Old_Img == 0)
		{
			image = cvRetrieveFrame(capture);
			current_Img = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);
			Old_Img = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);
		}

		
			//copy to image class     
			memcpy(Old_Img->imageData, current_Img->imageData, sizeof(char)*image->imageSize);
			image = cvRetrieveFrame(capture);
			memcpy(current_Img->imageData, image->imageData, sizeof(char)*image->imageSize);
   
			//Create image for Optical flow     
			if (imgA == 0)
			{
				imgA = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
				imgB = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
			}

			//RGB to Gray for Optical Flow     
			cvCvtColor(current_Img, imgA, CV_BGR2GRAY);
			cvCvtColor(Old_Img, imgB, CV_BGR2GRAY);

			Mat mat(imgA);
			FAST(mat, keypoints, threshold, false);

			int i = 0;
			for (vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(), i < MAX_COUNT; it++, i++)
			{
				cornersA[i] = it->pt;
			}
			cvFindCornerSubPix(imgA, cornersA, corner_count, cvSize(win_size, win_size), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));

			CvSize pyr_sz = cvSize(imgA->width + 8, imgB->height / 3);
			if (pyrA == 0)
			{
				pyrA = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);
				pyrB = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);
			}

			cvCalcOpticalFlowPyrLK(imgA, imgB, pyrA, pyrB, cornersA, cornersB, corner_count, cvSize(win_size, win_size), 3, features_found, feature_errors, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3), 0);

			for (int i = 0; i< corner_count; ++i)
			{

				if (features_found[i] == 0 || feature_errors[i] > MAX_COUNT)
					continue;
       
				//Vector Length     
				float fVecLength = sqrt((float)((cornersA[i].x - cornersB[i].x)*(cornersA[i].x - cornersB[i].x) + (cornersA[i].y - cornersB[i].y)*(cornersA[i].y - cornersB[i].y)));
				//Vector Angle     
				float fVecSetha = fabs(atan2((float)(cornersB[i].y - cornersA[i].y), (float)(cornersB[i].x - cornersA[i].x)) * 180 / PI);

				cvLine(image, cvPoint(cornersA[i].x, cornersA[i].y), cvPoint(cornersB[i].x, cornersA[i].y), CV_RGB(0, 255, 0), 2);

				//printf("[%d] - Sheta:%lf, Length:%lf\n", i, fVecSetha, fVecLength);
			} 
		cvShowImage("Origin", image);


		//break        
		if (cvWaitKey(10) >= 0)
			break;
	}

	//release capture point        
	cvReleaseCapture(&capture);
	//close the window        
	cvDestroyWindow("Origin");

	cvReleaseImage(&Old_Img);   
	cvReleaseImage(&imgA);
	cvReleaseImage(&imgB);
	delete cornersA;
	delete cornersB;
	cvReleaseImage(&pyrA);
	cvReleaseImage(&pyrB);
}