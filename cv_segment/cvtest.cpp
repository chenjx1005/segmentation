#include "PottsModel.hpp"
#include "FastLabel.hpp"
#include "OpticalFlow.cpp"

using namespace cv;
using namespace std;

void iterate(PottsModel &potts_model)
{
	/*while (potts_model.iterable()){
		potts_model.MetropolisOnce();
	}
	potts_model.GenBoundry();
	Mat boundry = potts_model.get_boundrymap();
	FastLabel f(boundry);
	f.FirstScan();
	potts_model.UpdateStates(f.get_labels());
	f.SecondScan();
	potts_model.UpdateStates(f.get_labels());
	for (int j=0; j < 4; j++) {
		potts_model.MetropolisOnce();
	}
	potts_model.GenBoundry();
	potts_model.SaveBoundry();
	boundry = potts_model.get_boundrymap();*/
	Mat boundry = imread("boundry.jpg", 0);
	FastLabel f2(boundry);
	f2.FirstScan();
	f2.SecondScan();
	potts_model.UpdateStates(f2.get_labels());
	potts_model.SaveStates("img1result.jpg");
}

int main()
{
	//!Single frame segment code
	Mat img;
	img = imread("Color0.png");
	//cvtColor(imread("Color0.png"), img, CV_BGR2HSV);
	Mat depth;
	cvtColor(imread("Depth0.png"), depth, CV_BGR2GRAY);
	
	PottsModel potts_model(img, depth, PottsModel::RGB);
	iterate(potts_model);
	//!Single frame segment code end
	//!optical flow test code
	Mat img2 = imread("Color1.png");
	Mat depth2;
	cvtColor(imread("Depth1.png"), depth2, CV_BGR2GRAY);
	PottsModel potts2(img2, depth2, potts_model);
	potts2.SaveStates();
	potts2.Freeze();
	for (int j=0; j < 7; j++) {
		potts2.MetropolisOnce();
		potts2.SaveStates();
	}
	//!optical flow test code end
    return 0;
}