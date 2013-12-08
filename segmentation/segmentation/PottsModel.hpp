#ifndef SEGMENTATION_POTTSMODEL_H
#define SEGMENTATION_POTTSMODEL_H

#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

typedef Vec<double, 8> Vec8d;
typedef Matx<int, 3, 3> Matx33i;

class PottsModel
{
public:
    PottsModel(const Mat &color, const Mat &depth);
	virtual ~PottsModel();
	void ComputeDifference();
	double PixelEnergy(int pi, int pj) const;
	void MetropolisOnce();
	bool iterable() const { return t_ >= min_t_; }
	void GenStatesResult();
	void ShowStates(int milliseconds=0);
	void SaveStates();
	void UpdateStates(const vector<vector<int>> &states);
	void GenBoundry();
	void ShowBoundry(int milliseconds=0)
	{
		imshow("PottsModel", boundry_);
		waitKey(milliseconds);
	}
	void SaveBoundry() { imwrite("boundry.jpg", boundry_); }
	Mat get_boundrymap() const { return boundry_; }
	
private:
	//the factor for computing the averaged color vector difference of all
	//neighbors<i, j> and range in [0, 10]
	double alpha_;
	//default 256 for convenience
	int num_spin_;
	double init_t_;
	double min_t_;
	//the annealing coefficient, less than 1
	double a_c_;
	double t_;
	//the Boltzman constant
	const double kK;
	//the neighbors of the computed pixel, {{4, 2, 6}, {0, -1, 1}, {7, 3 ,5}}
	const Matx33i kPixel;
	//the J for the difference of depthes of pixels > 30cm
	const int kMaxJ;
	double mean_diff_;
	int num_result_;
	int num_result_gen_;
	Mat color_;
	Mat depth_;
	Mat diff_;
	Mat states_result_;
	Mat boundry_;
	vector<vector<int>> states_;
};
#endif

