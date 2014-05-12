#ifndef SEGMENTATION_KINECTDRIVER_H
#define SEGMENTATION_KINECTDRIVER_H

#include <string>
#include <opencv2\opencv.hpp>
#include <OpenNI.h>

class KinectDriver
{
public:
	KinectDriver(int width=640, int height=480, int Fps=30);
	virtual ~KinectDriver(void);
	void CheckOpenNIError(openni::Status result, std::string status )
	{ 
		if( result != openni::STATUS_OK ) 
			std::cerr << status << " Error: " << openni::OpenNI::getExtendedError() << std::endl;
	}
	const cv::Mat &nextColor();
	const cv::Mat &nextDepth();

	openni::Status result;

	cv::Mat cvDepthImg;
	cv::Mat cvRGBImg;

	int rows;
	int cols;

private:
	openni::VideoFrameRef oniDepthImg;
    openni::VideoFrameRef oniColorImg;

	openni::VideoStream oniDepthStream;
	openni::VideoStream oniColorStream;

	cv::Mat cvRawImg16U;
	
	openni::Device device;
};

#endif

