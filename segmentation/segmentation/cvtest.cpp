#include "PottsModel.hpp"
#include "FastLabel.hpp"
#include "OpticalFlow.cpp"
#include "gpu_common.h"

using namespace cv;
using namespace std;

void iterate(PottsModel *potts_model)
{
	while (potts_model->iterable()){
		potts_model->MetropolisOnce();
		potts_model->ShowStates();
		potts_model->SaveStates();
	}
	potts_model->GenBoundry();
	Mat boundry = potts_model->get_boundrymap();
	FastLabel f(boundry);
	f.FirstScan();
	f.SecondScan();
	potts_model->UpdateStates(f.get_labels());
	for (int j=0; j < 4; j++) {
		potts_model->MetropolisOnce();
	}
	potts_model->GenBoundry();
	potts_model->SaveBoundry();
	boundry = potts_model->get_boundrymap();
	/*Mat boundry = imread("startboundry.jpg", 0);
	FastLabel f2(boundry);
	f2.FirstScan();
	f2.SecondScan();
	potts_model->UpdateStates(f2.get_labels());
	potts_model->SaveStates("img1result.jpg");*/
}

void rest_iterate(PottsModel *potts_model)
{
	static int n = 2;
	potts_model->Freeze();
	for (int j=0; j < 4; j++) {
		potts_model->MetropolisOnce();
		potts_model->SaveStates();
	}
	potts_model->GenBoundry();
	potts_model->SaveBoundry();
	Mat boundry = potts_model->get_boundrymap();
	FastLabel f(boundry);
	f.FirstScan();
	f.SecondScan();
	potts_model->UpdateStates(f.get_labels());
	char c[4];
	sprintf(c, "img%dresult.jpg", n++);
	potts_model->SaveStates(c);
}

int main()
{
	//mymain();
	//!Single frame segment code
	Mat img;
	img = imread("ColorOnlyTest/Color01.png");
	Mat depth;
	cvtColor(imread("ColorOnlyTest/Depth01.png"), depth, CV_BGR2GRAY);

	CudaSetup(img.rows, img.cols);
	GpuPottsModel m(img, depth);
	m.Metropolis();
	m.GenBoundry();
	m.Label();
	m.CopyStates();
	for(int i = 0; i < 10; i++)
	{
		m.MetropolisOnce();
	}
	m.GenBoundry();
	m.Label();
	m.SaveStates();
	m.CopyStates();
	char d_title[300];
	char d_deptitle[300];
	for(int i = 2; i < 20; i++)
	{
		//time_print("",0);
		sprintf(d_title, "ColorOnlyTest/Color%02d.png", i);
		sprintf(d_deptitle, "ColorOnlyTest/Depth%02d.png", i);
		img = imread(d_title);
		cvtColor(imread(d_deptitle), depth, CV_BGR2GRAY);
		m.LoadNextFrame(img, depth);
		for(int i = 0; i < 20; i++)
		{
			m.MetropolisOnce();
		}
		m.GenBoundry();
		m.Label();
		m.SaveStates();
		m.CopyStates();
		//time_print("new frame");
	}
	CudaRelease();

	PottsModel *potts_model = new PottsModel(img, depth, PottsModel::RGB);
	iterate(potts_model);
	//!Single frame segment code end
	
	/*!optical flow test code*/
	/*Mat img2 = imread("Color01.png");
	Mat depth2;
	cvtColor(imread("Depth01.png"), depth2, CV_BGR2GRAY);
	
	PottsModel *potts2 = new PottsModel(img2, depth2, *potts_model);
	potts2->SaveStates();
	rest_iterate(potts2);
	potts_model = potts2;

	img2 = imread("Color02.png");
	cvtColor(imread("Depth02.png"), depth2, CV_BGR2GRAY);
	
	potts2 = new PottsModel(img2, depth2, *potts_model);
	potts2->SaveStates();
	rest_iterate(potts2);
	potts_model = potts2;

	img2 = imread("Color03.png");
	cvtColor(imread("Depth03.png"), depth2, CV_BGR2GRAY);
	
	potts2 = new PottsModel(img2, depth2, *potts_model);
	potts2->SaveStates();
	rest_iterate(potts2);*/

	/*!optical flow test code end*/
	
	/*consequent frame segment code*/
	/*VideoCapture color_cap("C:/Users/chenjx/Documents/Visual Studio 2010/Projects/cv_segment/cv_segment/ColorOnlyTest/Color%02d.png");
	VideoCapture depth_cap("C:/Users/chenjx/Documents/Visual Studio 2010/Projects/cv_segment/cv_segment/ColorOnlyTest/Depth%02d.png");
	if( !(color_cap.isOpened() && depth_cap.isOpened()) )
        return -1;*/
	char title[300];
	char deptitle[300];
	Mat color, dep;
	PottsModel *potts2;
	for(int i = 1; i < 20; i++)
	{
		//if( !(color_cap.read(color) && depth_cap.read(dep)) ) break;
		sprintf(title, "ColorOnlyTest/Color%02d.png", i);
		sprintf(deptitle, "ColorOnlyTest/Depth%02d.png", i);
		color = imread(title);
		cvtColor(imread(deptitle), dep, CV_BGR2GRAY);
		potts2 = new PottsModel(color, dep, *potts_model);
		potts2->SaveStates();
		rest_iterate(potts2);
		delete potts_model;
		potts_model = potts2;
	}
	delete potts_model;
    return 0;
}