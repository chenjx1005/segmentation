#ifndef SEGMENTATION_POTTSMODEL_H
#define SEGMENTATION_POTTSMODEL_H

#include <cstdio>
#include <cmath>
#include <vector>
#include <list>

#include "cv.h"
#include "highgui.h"


typedef cv::Vec<double, 8> Vec8d;
typedef cv::Matx<int, 3, 3> Matx33i;

class PottsModel
{
public:
    PottsModel(const cv::Mat &color, const cv::Mat &depth);
    PottsModel(const cv::Mat &color);
	virtual ~PottsModel();
	void ComputeDifference();
	double PixelEnergy(int pi, int pj) const;
	void MetropolisOnce();
	bool iterable() const { return t_ >= min_t_; }
	void GenStatesResult();
	void ShowStates(int milliseconds=0);
	void SaveStates();
	void UpdateStates(const std::vector<std::vector<int> > &states);
	void GenBoundry();
	void ShowBoundry(int milliseconds=0) const
	{
		cv::imshow("PottsModel", boundry_);
		cv::waitKey(milliseconds);
	}
	void SaveBoundry() const { cv::imwrite("boundry.jpg", boundry_); }
	cv::Mat get_boundrymap() const { return boundry_; }
	void set_temperature(double t) { t_ = t; }
	void Freeze() { t_ = min_t_; }
	void HorizontalColor() const;
	void VerticalColor() const;
	void RightDiagColor() const;
	void LeftDiagColor() const;
	
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
	cv::Mat color_;
	cv::Mat depth_;
	cv::Mat diff_;
	cv::Mat states_result_;
	cv::Mat boundry_;
	std::vector<std::vector<int> > states_;
};
#endif

