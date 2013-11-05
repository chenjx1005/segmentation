#include <opencv\highgui.h>

int main()
{
	IplImage* img = cvLoadImage("lena.jpg");
	cvNamedWindow("Example 1", CV_WINDOW_AUTOSIZE);
	cvShowImage("img", img);
	cvWaitKey(0);
	cvReleaseImage(&img);
	cvDestroyWindow("Example 1");
}

