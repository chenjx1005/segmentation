#include "PottsModel.hpp"
#include "FastLabel.hpp"

using namespace cv;
using namespace std;

double HSVNorm(const Vec3b &, const Vec3b &);

int main(int, char**)
{
	Mat img = imread("Color0.png");
	Mat depth;
	cvtColor(imread("Depth0.png"), depth, CV_BGR2GRAY);
	
	PottsModel potts_model(img, depth);
	/*while (potts_model.iterable()){
		potts_model.MetropolisOnce();
		potts_model.SaveStates();
	}
	potts_model.GenBoundry();
	potts_model.ShowBoundry(5000);
	potts_model.SaveBoundry();
	Mat boundry = potts_model.get_boundrymap();*/
	Mat boundry = imread("b.jpg", 0);
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

double HSVNorm(const Vec3b &a, const Vec3b &b)
{
	Vec3d ad = Vec3d(a[1] * a[2] / 65025.0 * cos(double(a[0] * 2)), 
					 a[1] * a[2] / 65025.0 * sin(double(a[0] * 2)),
					 a[2] / 255.0);
	Vec3d bd = Vec3d(b[1] * b[2] / 65025.0 * cos(double(b[0] * 2)), 
					 b[1] * b[2] / 65025.0 * sin(double(b[0] * 2)),
					 b[2] / 255.0);
	return norm(ad - bd);
}
