#ifndef SEGMENTATION_POTTSMODEL_H
#define SEGMENTATION_POTTSMODEL_H

#include <cstdio>
#include <cmath>
#include <vector>
#include <list>
#include <string>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

class BasicPottsModel
{
public:
	typedef cv::Vec<double, 8> Vec8d;
	typedef cv::Matx<int, 3, 3> Matx33i;
	enum ColorSpace {HSV = 1, RGB = 2,};
	BasicPottsModel():
		start_frame_(1), alpha_(1.0), num_spin_(256), init_t_(1.3806488e+7), min_t_(0.1), a_c_(0.33),
		t_(init_t_), kK(1.3806488e-4), kMaxJ(250.0), num_result_(0),
		num_result_gen_(-1),
		kPixel(4, 2, 6, 0, -1, 1, 7, 3 ,5), color_space_(RGB) {};
	BasicPottsModel(const cv::Mat &color, const cv::Mat &depth, int color_space=RGB):
		start_frame_(1), alpha_(1.0), num_spin_(256), init_t_(1.3806488e+7), min_t_(0.1), a_c_(0.33),
		t_(init_t_), kK(1.3806488e-4), kMaxJ(250.0), num_result_(0),
		num_result_gen_(-1), color_(color), depth_(depth),
		boundry_(color.rows, color.cols, CV_8U),
		kPixel(4, 2, 6, 0, -1, 1, 7, 3 ,5), color_space_(color_space),
		segment_depth_(color.rows, color.cols, CV_8U) 
		{
			if(color_.size == depth_.size) rows_ = color_.rows, cols_ = color_.cols;
			else
			{
				std::cout<<"color image and depth image size error!"<<std::endl;
				exit(0);
			}
		};
	virtual ~BasicPottsModel() {};
	virtual void ComputeDifference() = 0;
	virtual void MetropolisOnce() = 0;
	bool iterable() const { return t_ >= min_t_; }
	cv::Mat get_boundrymap() const { return boundry_; }
	void set_temperature(double t) { t_ = t; }
	void Freeze() { t_ = min_t_; }
	void ShowBoundry(int milliseconds=0) const
	{
		cv::imshow("PottsModel", boundry_);
		cv::waitKey(milliseconds);
	}
	void SaveBoundry() const { cv::imwrite("boundry.jpg", boundry_); }
	virtual void ShowDifference() const {};
	virtual void HorizontalColor() const {};
	virtual void VerticalColor() const {};
	virtual void RightDiagColor() const {};
	virtual void LeftDiagColor() const {};
	
	//the flag whether this frame is the start frame
	int start_frame_;
	//the factor for computing the averaged color vector difference of all
	//neighbors<i, j> and range in [0, 10]
	double alpha_;
	//default 256 for convenience
	const int num_spin_;
	double init_t_;
	double min_t_;
	//the annealing coefficient, less than 1
	double a_c_;
	double t_;
	//the Boltzman constant
	const double kK;
	//the J for the difference of depthes of pixels > 30cm
	const double kMaxJ;
	//the variables used to control a result image whether to save
	int num_result_;
	int num_result_gen_;
	//init matrix, color and depth
	cv::Mat color_;
	cv::Mat depth_;
	//the matrix used for showing
	cv::Mat states_result_;
	cv::Mat boundry_;
	//the average range value of each segment
	cv::Mat segment_depth_;
	//the farneback optical flow object
	static cv::gpu::FarnebackOpticalFlow FarneCalc;
	//TODO:the spin variable of each pixel is not declare
	size_t rows_;
	size_t cols_;

protected:
	virtual double PixelEnergy(int pi, int pj) const = 0;
	//the neighbors of the computed pixel, {{4, 2, 6}, {0, -1, 1}, {7, 3 ,5}}
	//4, 2, 6
	//0,-1, 1
	//7, 3, 5
	const Matx33i kPixel;
	int color_space_;
};


class PottsModel : public BasicPottsModel
{
public:
	//color is a hsv/bgr Mat(CV_8UC3), depth is a gray Mat(CV_8U)
    PottsModel(const cv::Mat &color, const cv::Mat &depth, int color_space=RGB);
    PottsModel(const cv::Mat &color, int color_space=RGB);
	PottsModel(const cv::Mat &color, const cv::Mat &depth, PottsModel &last_frame, int color_space=RGB);
	virtual ~PottsModel();
	virtual void ComputeDifference();
	virtual void MetropolisOnce();
	virtual void GenStatesResult();
	virtual void ShowStates(int milliseconds=0);
	virtual void SaveStates(const std::string &title="");
	//update states of the model after label
	virtual void UpdateStates(const std::vector<std::vector<int> > &states);
	virtual void UpdateSegmentDepth();
	virtual void GenBoundry();
	virtual void ShowDifference() const;
	virtual void HorizontalColor() const;
	virtual void VerticalColor() const;
	virtual void RightDiagColor() const;
	virtual void LeftDiagColor() const;
	//the spin variable of each pixel
	std::vector<std::vector<int> > states_;
	//init matrix, difference of neighbor pixels
	cv::Mat diff_;
protected:
	virtual double PixelEnergy(int pi, int pj) const;
	double Distance(const cv::Vec3b &a, const cv::Vec3b &b) const;
};


class GpuPottsModel : public BasicPottsModel
{
public:
	GpuPottsModel(const cv::Mat &color, const cv::Mat &depth, int color_space=RGB);
	//GpuPottsModel(const cv::Mat &color, const cv::Mat &depth, PottsModel &last_frame, int color_space=RGB);
	virtual ~GpuPottsModel();
	virtual void ComputeDifference();
	virtual void MetropolisOnce();

	uchar *states_;
	double (*diff_)[8];

private:
	virtual double PixelEnergy(int pi, int pj) const;
};

#endif

