#ifndef OPTICALFLOW_CPP
#define OPTICALFLOW_CPP
// Farneback dense optical flow calculate and show in Munsell system of colors
// Author : Zouxy
// Date   : 2013-3-15
// HomePage : http://blog.csdn.net/zouxy09
// Email  : zouxy09@qq.com

// API calcOpticalFlowFarneback() comes from OpenCV, and this
// 2D dense optical flow algorithm from the following paper:
// Gunnar Farneback. "Two-Frame Motion Estimation Based on Polynomial Expansion".
// And the OpenCV source code locate in ..\opencv2.4.3\modules\video\src\optflowgf.cpp

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cv;
using namespace std;
using namespace cv::gpu;

void makecolorwheel(vector<Scalar> &colorwheel);
void motionToColor(Mat flow, Mat &color);
int optical();

#define UNKNOWN_FLOW_THRESH 1e9

// Color encoding of flow vectors from:
// http://members.shaw.ca/quadibloc/other/colint.htm
// This code is modified from:
// http://vision.middlebury.edu/flow/data/
inline void makecolorwheel(vector<Scalar> &colorwheel)
{
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;

    int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,	   255*i/RY,	 0));
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,		 0));
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,		   255,		 255*i/GC));
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,		   255-255*i/CB, 255));
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,	   0,		 255));
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,	   0,		 255-255*i/MR));
}

inline void motionToColor(Mat flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:
    float maxrad = -1;

	// Find max flow to normalize fx and fy
	for (int i= 0; i < flow.rows; ++i) 
	{
		for (int j = 0; j < flow.cols; ++j) 
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}

	for (int i= 0; i < flow.rows; ++i) 
	{
		for (int j = 0; j < flow.cols; ++j) 
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; // uncomment to see original color wheel

			for (int b = 0; b < 3; b++) 
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius
				else
					col *= .75; // out of range
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}

inline int optical()
{
    VideoCapture cap("Color%1d.png");

	//cap.set(CV_CAP_PROP_FRAME_WIDTH,320.);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT,240.);
	//cout<< cap.get(CV_CAP_PROP_FRAME_WIDTH) << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << cap.get(CV_CAP_PROP_FORMAT) << endl;

    if( !cap.isOpened() )
        return -1;

    Mat prevgray, gray, flow, cflow, frame;
	Mat flowx, flowy;
	GpuMat g_prevgray, g_gray;
	GpuMat d_flowx, d_flowy;
	FarnebackOpticalFlow d_calc;
    //namedWindow("flow", 1);

	//Mat motion2color;

    for(;;)
    {
		double t=0;

        cap >> frame;
        cvtColor(frame, gray, CV_BGR2GRAY);
		//imshow("original", frame);

        if( prevgray.data )
        {
			t = (double)cvGetTickCount();
		    g_prevgray.upload(prevgray);
			g_gray.upload(gray);
			d_calc(g_prevgray, g_gray, d_flowx, d_flowy);
			t = (double)cvGetTickCount() - t;
            //calcOpticalFlowFarneback(prevgray, gray, flow, d_calc.pyrScale, d_calc.numLevels, d_calc.winSize,
            //           d_calc.numIters, d_calc.polyN, d_calc.polySigma, d_calc.flags);
			
			//motionToColor(flow, motion2color);
            //imshow("flow", motion2color);
			d_flowx.download(flowx);
			cout << format(flowx,"python") <<endl;
			break;
        }
        if(waitKey(10)>=0)
            break;
        std::swap(prevgray, gray);

		cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
    }
    return 0;
}
#endif