#include <stdlib.h>
#include <iostream>
#include <time.h>

#include "KinectDriver.h"

using namespace std;
using namespace cv;
using namespace openni;

const double MaxPixelValue = 10000.0;

KinectDriver::KinectDriver(int width, int height, int Fps):
	result(STATUS_OK), cvRGBImg(height, width, CV_8UC3), cvDepthImg(height, width, CV_8U), cvRawImg16U(height, width, CV_16U), rows(height), cols(width)
{
	result = OpenNI::initialize();
	CheckOpenNIError( result, "initialize context" );
	
	result = device.open(ANY_DEVICE);
	result = oniDepthStream.create(device, SENSOR_DEPTH );

	// set depth video mode
    VideoMode modeDepth;
    modeDepth.setResolution(width, height);
    modeDepth.setFps(Fps);
    modeDepth.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
    oniDepthStream.setVideoMode(modeDepth);
	// start depth stream
    result = oniDepthStream.start();

	result = oniColorStream.create(device, SENSOR_COLOR);
	// set color video mode
	VideoMode modeColor;
    modeColor.setResolution(width, height);
    modeColor.setFps(Fps);
    modeColor.setPixelFormat(PIXEL_FORMAT_RGB888);
    oniColorStream.setVideoMode(modeColor);
	// start color stream
    result = oniColorStream.start();
}

KinectDriver::~KinectDriver(void)
{
	//OpenNI2 destroy
    oniDepthStream.destroy();
    oniColorStream.destroy();
    device.close();
    OpenNI::shutdown();
}

const Mat &KinectDriver::nextColor()
{
	if(oniColorStream.readFrame( &oniColorImg ) == STATUS_OK)
	{
		cvRGBImg.data = (uchar*)oniColorImg.getData();
		//convert data into BGR
		//Mat BGRimg;
		//cvtColor(cvRGBImg, BGRimg, CV_RGB2BGR);
		//imshow( "openNI_image", BGRimg);
		//waitKey();
	}
	else
	{
		cout<<"KinectDriver::nextColor() status is wrong"<<endl;
	}
	return cvRGBImg;
}

const Mat &KinectDriver::nextDepth()
{
	if( oniDepthStream.readFrame( &oniDepthImg ) == STATUS_OK )
	{
		cvRawImg16U.data = (uchar*)oniDepthImg.getData();
		//cvRawImg16U.convertTo(cvDepthImg, CV_8U, 255.0/(oniDepthStream.getMaxPixelValue()));
		cvRawImg16U.convertTo(cvDepthImg, CV_8U, 255.0/MaxPixelValue);
	}
	else
	{
		cout<<"KinectDriver::nextDepth() status is wrong"<<endl;
	}
	return cvDepthImg;
}