#include "PottsModel.hpp"
#include "FastLabel.hpp"

using namespace cv;
using namespace std;

int main(int, char**)
{
	Mat img;
	cvtColor(imread("Color0.png"), img, CV_BGR2HSV);
	Mat depth;
	cvtColor(imread("Depth0.png"), depth, CV_BGR2GRAY);
	
	PottsModel potts_model(img, depth);
	//potts_model.ShowDifference();
	while (potts_model.iterable()){
		potts_model.MetropolisOnce();
		potts_model.SaveStates();
	}
	potts_model.GenBoundry();
	potts_model.ShowBoundry(5000);
	potts_model.SaveBoundry();
	Mat boundry = potts_model.get_boundrymap();
	//Mat boundry = imread("b.jpg", 0);
	FastLabel f(boundry);
	f.FirstScan();
	potts_model.UpdateStates(f.get_labels());
	potts_model.SaveStates();
	f.SecondScan();
	potts_model.UpdateStates(f.get_labels());
	potts_model.SaveStates();
	for (int i=0; i < 3; i++)
	{
		potts_model.Freeze();
		for (int j=0; j < 4; j++) {
			potts_model.MetropolisOnce();
			potts_model.SaveStates();
		}
		potts_model.GenBoundry();
		potts_model.SaveBoundry();
		FastLabel f2(potts_model.get_boundrymap());
		f2.FirstScan();
		f2.SecondScan();
		potts_model.UpdateStates(f2.get_labels());
		potts_model.SaveStates();
	}
    return 0;
}
