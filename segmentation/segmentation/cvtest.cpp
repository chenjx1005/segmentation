#include <stdio.h>
#include <cmath>
#include <vector>
#include "cv.h"
#include "highgui.h"

#include "PottsModel.hpp"

using namespace cv;
using namespace std;

double HSVNorm(const Vec3b &, const Vec3b &);

const int PottsModel::kPixel[3][3] = {{4, 2, 6}, {0, -1, 1}, {7, 3 ,5}};

int main(int, char**)
{
	Mat img = imread("Color01.png");
	Mat depth;
	cvtColor(imread("Depth01.png"), depth, CV_BGR2GRAY);
	
	PottsModel potts_model(img, depth);
	while (potts_model.Iterable()){
		potts_model.MetropolisOnce();
		potts_model.SaveStates();
	}
	potts_model.GenBoundry();
	potts_model.ShowBoundry(5000);
	potts_model.SaveBoundry();
	//find boundaries
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
