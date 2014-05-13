#include <limits>
#include <map>
#include <algorithm>

#include <time.h>

#include "PottsModel.hpp"
#include "gpu_common.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

FarnebackOpticalFlow BasicPottsModel::FarneCalc;

namespace {
const double EPSILON = numeric_limits<double>::epsilon();
//When show color difference, if the color difference < kwhite, show white
const double kWhite = 0.0;
//the map of states_ and result showed when ShowStates to distinguish similar but different states
const int kStatesResult[256] = {103, 132, 101, 209, 230, 222, 44, 79, 247, 59, 62, 77, 148, 241, 184, 240, 221, 173, 21, 190,
								58, 140, 246, 144, 119, 115, 111, 170, 50, 81, 141, 16, 121, 164, 219, 155, 197, 2, 163, 57,
								134, 129, 56, 126, 235, 47, 78, 80, 231, 55, 210, 248, 114, 104, 54, 189, 70, 63, 5, 27,
								229, 36, 161, 4, 244, 67, 172, 97, 153, 212, 7, 93, 89, 107, 193, 12, 217, 202, 13, 68, 168,
								69, 100, 149, 42, 123, 249, 192, 1, 208, 46, 41, 120, 72, 43, 86, 35, 223, 167, 25, 53, 162,
								130, 19, 157, 234, 87, 15, 152, 215, 242, 128, 98, 169, 200, 45, 185, 245, 224, 214, 174, 195,
								182, 96, 196, 0, 88, 30, 37, 76, 253, 207, 95, 238, 83, 116, 125, 105, 187, 133, 199, 122,
								137, 85, 90, 11, 65, 51, 225, 124, 145, 243, 64, 237, 61, 228, 117, 8, 143, 194, 18, 178,
								135, 20, 142, 82, 166, 136, 92, 179, 211, 109, 91, 154, 252, 32, 138, 28, 204, 239, 49, 150,
								73, 99, 110, 38, 158, 118, 255, 236, 60, 26, 139, 218, 156, 24, 160, 31, 29, 22, 39, 181,
								232, 112, 48, 147, 201, 188, 52, 75, 177, 203, 191, 165, 176, 213, 14, 6, 127, 151, 71, 9,
								251, 250, 186, 205, 23, 216, 108, 227, 102, 146, 180, 40, 10, 226, 84, 131, 74, 94, 3, 34,
								220, 66, 198, 183, 254, 206, 106, 175, 171, 113, 233, 33, 159, 17};
//use for funciton c in GpuPottsModel Label
const int kC[5][2] = {{0, 0}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};

double hsv_distance(const Vec3b &a, const Vec3b &b)
{
	Vec3d ad(a[1] * a[2] / 65025.0 * cos(double(a[0] * 2)),
			 a[1] * a[2] / 65025.0 * sin(double(a[0] * 2)),
			 a[2] / 255.0);
	Vec3d bd(b[1] * b[2] / 65025.0 * cos(double(b[0] * 2)),
			 b[1] * b[2] / 65025.0 * sin(double(b[0] * 2)),
			 b[2] / 255.0);
	return norm(ad - bd);
}
}

PottsModel::PottsModel(const Mat &color, const Mat &depth, int color_space)
	:BasicPottsModel(color, depth, color_space),
	states_(color.rows, vector<int>(color.cols, -1))
{
	for (int i = 0; i < color_.rows; i++)
		for (int j = 0; j < color_.cols; j++)
			states_[i][j] = depth.at<uchar>(i, j);
	ComputeDifference();
	//namedWindow("PottsModel");
}

PottsModel::PottsModel(const Mat &color, int color_space)
	:BasicPottsModel(color, Mat(color.rows, color.cols, CV_8U, Scalar::all(0)), color_space),
	states_(color.rows, vector<int>(color.cols))
{
	RNG r;
	for (int i = 0; i < color_.rows; i++)
		for (int j = 0; j < color_.cols; j++)
			states_[i][j] = r.next() % num_spin_;
	ComputeDifference();
	//namedWindow("PottsModel");
}

PottsModel::PottsModel(const Mat &color, const Mat &depth, PottsModel &last_frame, int color_space)             
	:BasicPottsModel(color, depth, color_space),
	states_(color.rows, vector<int>(color.cols, -1))
{
	start_frame_ =0;
	GpuMat d_flowx, d_flowy;
	Mat flowx, flowy;
	Mat gary_last, gary;

	cvtColor(color, gary, CV_BGR2GRAY);
	cvtColor(last_frame.color_, gary_last, CV_BGR2GRAY);
	double t = (double)cvGetTickCount();
	FarneCalc(GpuMat(gary_last), GpuMat(gary), d_flowx, d_flowy);
	t = (double)cvGetTickCount() - t;
	cout << "optical flow cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
	d_flowx.download(flowx);
	d_flowy.download(flowy);

	last_frame.UpdateSegmentDepth();
	RNG r;
	int x, y;
	for (int i = 0; i < color_.rows; i++)
		for (int j = 0; j < color_.cols; j++)
		{
			x = i + static_cast<int>(flowx.at<float>(i, j));
			y = j + static_cast<int>(flowy.at<float>(i, j));
			if (x >= 0 && y >=0 && x < color_.rows && y < color_.cols && 
				abs(static_cast<int>(depth.at<uchar>(x, y)) - static_cast<int>(last_frame.depth_.at<uchar>(i, j))) <= 30 &&
				abs(static_cast<int>(depth.at<uchar>(x, y)) - static_cast<int>(last_frame.segment_depth_.at<uchar>(i, j))) <= 100)
			{
				states_[x][y] = last_frame.states_[i][j];
			}
			//else
			//{
			//	cout << "x:"<<x<<" y:"<<y<<" depth:"<<static_cast<int>(depth.at<uchar>(x, y))<<" "<<static_cast<int>(last_frame.depth_.at<uchar>(i, j))<<" "<<static_cast<int>(last_frame.segment_depth_.at<uchar>(i, j))<<endl;
			//}
		}
	for (int i = 0; i < color_.rows; i++)
		for (int j = 0; j < color_.cols; j++)
		{
			if (states_[i][j] == -1)
			{
				states_[i][j] = r.next() % num_spin_;
			}
		}
	ComputeDifference();
	//namedWindow("PottsModel");
}

PottsModel::~PottsModel()
{
	cout << "Destructor"<<endl;
	//destroyWindow("PottsModel");
}

void PottsModel::ComputeDifference()
{
	CV_Assert(color_.type() == CV_8UC3);
	//time_print("", 0);
	//initialize diff matrix
	int sz[3] = {color_.rows, color_.cols, 8};
	diff_.create(3, sz, CV_64F);

	double sum = 0, color_diff = 0, mean_diff_ = 0;
	int count = 0, depth_diff = 0;
	Vec3b gi, gj;
	uchar di,dj;
	list<Vec3s> later_update;
	//horizontal diff
	for (int i = 0; i < color_.rows; i++)
	{
		const Vec3b *p = color_.ptr<Vec3b>(i);
		const uchar *d = depth_.ptr<uchar>(i);
		gi = p[0];
		di = d[0];
		for (int j = 1; j < color_.cols; j++)
		{
			count++;
			gj = p[j];
			dj = d[j];
			color_diff = Distance(gi, gj);
			depth_diff = abs(static_cast<int>(di) - static_cast<int>(dj));
			sum += color_diff;
			if (depth_diff > 30){
				later_update.push_back(Vec3s(i, j-1, 1));
				later_update.push_back(Vec3s(i, j, 0));
			} else {
				diff_.at<double>(i, j-1, 1) = diff_.at<double>(i, j, 0) = color_diff;
			}
			gi = gj;
			di = dj;
		}
	}
	//vertical diff
	for (int j = 0; j < color_.cols; j++)
	{
		gi = color_.at<Vec3b>(0, j);
		di = depth_.at<uchar>(0, j);
		for (int i = 1; i < color_.rows; i++)
		{
			count++;
			gj = color_.at<Vec3b>(i, j);
			dj = depth_.at<uchar>(i, j);
			color_diff = Distance(gi, gj);
			depth_diff = abs(static_cast<int>(di) - static_cast<int>(dj));
			sum += color_diff;
			if (depth_diff > 30) {
				later_update.push_back(Vec3s(i-1, j, 3));
				later_update.push_back(Vec3s(i, j, 3));
			} else {
				diff_.at<double>(i-1, j, 3) = diff_.at<double>(i, j, 2) = color_diff;
			}
			gi = gj;
			di = dj;
		}
	}
	//right diagonal diff
	for(int d = 0; d < color_.cols + color_.rows - 1 ; d++)
	{
		int flag = 0;
		for (int i = 0; i <= d; i++)
		{
			int j = d - i;
			if (i < color_.rows && j < color_.cols)
			{
				gj = color_.at<Vec3b>(i, j);
				dj = depth_.at<uchar>(i, j);
				if (!flag) {
					flag = 1;
				} else {
					count++;
					color_diff = Distance(gi, gj);
					depth_diff = abs(static_cast<int>(di) - static_cast<int>(dj));
					sum += color_diff;
					if (depth_diff > 30) {
						later_update.push_back(Vec3s(i - 1, d - i + 1, 7));
						later_update.push_back(Vec3s(i, j, 6));
					} else {
						diff_.at<double>(i - 1, d - i + 1, 7) = diff_.at<double>(i, j, 6) = color_diff;
					}
				}
				gi = gj;
				di = dj;
			}
		}
	}
	//left diagonal diff
	for (int n = -1 * color_.rows + 2; n < color_.cols - 1; n++)
	{
		Mat color_diag = color_.diag(n);
		Mat depth_diag = depth_.diag(n);
		int pi,pj;
		if (n<0) pi = -n, pj=0;
		else pi = 0, pj = n;
		const Vec3b *p = color_diag.ptr<Vec3b>(0);
		const uchar *d = depth_diag.ptr<uchar>(0);
		gi = p[0];
		di = d[0];
		for (int j = 1; j < color_diag.rows; j++)
		{
			count++;
			gj = *color_diag.ptr<Vec3b>(j);
			dj = *depth_diag.ptr<uchar>(j);
			color_diff = Distance(gi, gj);
			depth_diff = abs(static_cast<int>(di) - static_cast<int>(dj));
			sum += color_diff;
			if (depth_diff > 30) {
				later_update.push_back(Vec3s(pi + j - 1, pj + j - 1, 5));
				later_update.push_back(Vec3s(pi + j, pj + j, 4));
			} else {
				diff_.at<double>(pi + j - 1, pj + j - 1, 5) = diff_.at<double>(pi + j, pj + j, 4) = color_diff;
			}
			gi = gj;
			di = dj;
		}
	}
	mean_diff_ = alpha_ * sum / count;
	//time_print("CPU Compute");
	printf("sum is %lf, count is %d, mean_diff is %lf when alpha=%lf\n", sum, count, mean_diff_, alpha_);
	if (fabs(mean_diff_) >= EPSILON)
		diff_ = diff_ * (1 / mean_diff_) - Scalar::all(1);
	else diff_ -= Scalar::all(1);
	//time_print("CPU mean");
	for (list<Vec3s>::const_iterator it = later_update.begin(); it != later_update.end(); it++)
	{
		diff_.at<double>((*it)[0], (*it)[1], (*it)[2]) = kMaxJ;
	}
	//time_print("CPU Kmaxj");
}

double PottsModel::PixelEnergy(int pi, int pj) const
{
	int ki,kj;
	double energy = 0;
	for(int i = -1; i <= 1; i++)
		for(int j = -1; j <= 1; j++)
		{
			if(0 == i && 0 == j) continue;
			ki = pi + i;
			kj = pj + j;
			if(ki < 0 || ki >= diff_.size[0] || kj < 0 || kj >= diff_.size[1]) continue;
			energy += diff_.at<double>(pi, pj, kPixel(i + 1, j + 1)) * (states_[pi][pj] == states_[ki][kj]);
		}
	return energy;
}

void PottsModel::MetropolisOnce()
{
	//time_print("",0);
	vector<int> min_sequence;
	RNG r;
	int a_count = 0, r_count = 0, d_count = 0;
	int schedule = 0, current = 0;
	for (int i = 0; i < color_.rows; i++)
		{
			schedule = 0;
			for (int j = 0; j < color_.cols; j++)
			{
				min_sequence.clear();
				int s = states_[i][j];
				double e = PixelEnergy(i, j);
				double min_e = 100000000;
				for (int m = 0; m < num_spin_; ++m)
				{
					if (s == m) continue;
					states_[i][j] = m;
					double e_i = PixelEnergy(i, j);
					if (e_i < min_e){
						min_sequence.clear();
						min_sequence.push_back(m);
						min_e = e_i;
					} else if (e_i == min_e) {
						min_sequence.push_back(m);
					}
				}
				double diff_e = min_e - e;
				double p = exp(-1 * abs(diff_e) / (t_ * kK));
				double y = static_cast<double>(r.next() % 1000) / 1000.0;
				if (diff_e <= 0 || y < p)
				{
					if (diff_e > 0) {    //printf("t is %lf move is accepted and diff_e is %lf\n", t, diff_e);
						a_count++;
					} else {
						d_count++;
					}
					int p = r.next() % min_sequence.size();
					states_[i][j] = min_sequence[p];
				}
				else 
				{
					//printf("!!move is not accepted and diff_e is %lf\n", diff_e);
					r_count++;
					states_[i][j] = s;
				}
			}
			current = i * 100 / color_.rows;
			if (current > schedule) {
				schedule = current;
				printf(".");
				fflush(stdout);
			}
		}
		printf("\ntemperature is %lf, %d is accepted, %d is refused, %d is decreased\n",t_, a_count, r_count, d_count);
		t_ *= a_c_;
		num_result_++;
		//time_print("CPU Metropolis Once");
}

void PottsModel::GenStatesResult()
{
	if (num_result_gen_ >= num_result_) return;
	int rows = states_.size();
	int cols = states_[0].size();
	int states;
	Mat hsv(rows, cols, CV_8UC3);
	MatIterator_<Vec3b> it = hsv.begin<Vec3b>(), end = hsv.end<Vec3b>();
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			states = states_[i][j];
			//map the state to the result array
			Vec3b c(static_cast<uchar>(kStatesResult[states] * 0.6), 180, 230);
			//if the state is num_spin_ - 1, the pixel is on the boundry line.
			//so use black
			if (states_[i][j] == num_spin_ - 1) c[2] = 0;
			(*it) = c;
			it++;
		}
	cvtColor(hsv, states_result_, CV_HSV2BGR);
	num_result_gen_ = num_result_;
}

void PottsModel::ShowStates(int milliseconds)
{
	GenStatesResult();
	imshow("PottsModel", states_result_);
	waitKey(milliseconds);
}

void PottsModel::SaveStates(const string &title)
{
	GenStatesResult();
	if (title == "")
	{
		char result_name[20];
		sprintf(result_name, "result%d.jpg", num_result_);
		string s(result_name);
		imwrite(s, states_result_);
	}
	else
	{
		imwrite(title, states_result_);
	}
}

//TODO: optimize the states copy
void PottsModel::UpdateStates(const vector<vector<int> > &states)
{
	CV_Assert(states.size() == color_.rows);
	for (int i = 0; i < states.size(); i++){
		CV_Assert(states[i].size() == color_.cols);
	}
	if (start_frame_)
	{
		for (int i = 0; i < color_.rows; i++)
			for (int j = 0; j < color_.cols; j++)
			{
				states_[i][j] = states[i][j] % num_spin_;
			}
	}
	else
	{
		map<int, int> states_map;
		map<int, vector<int> > states_count;
		map<int, vector<int> >::iterator it;
		int s, si;

		for (int i = 0; i < color_.rows; i++)
			for (int j = 0; j < color_.cols; j++)
			{
				s = states[i][j];
				//if position i,j is the boundry
				if ( s % num_spin_ == num_spin_ - 1) 
				{
					states_[i][j] = -1;
					continue;
				}
				it = states_count.find(s);
				if (it == states_count.end()) 
				{
					states_count.insert(make_pair(s, vector<int>(num_spin_ - 1)));
				}
				else
				{
					si = states_[i][j];
					if (si != num_spin_ - 1)
						(it->second)[si]++;
				}
			}
		for (it = states_count.begin(); it != states_count.end(); it++)
		{
			states_map[it->first] = max_element((it->second).begin(), (it->second).end()) - (it->second).begin();
		}
		for (int i = 0; i < color_.rows; i++)
			for (int j = 0; j < color_.cols; j++)
			{
				if (states_[i][j] != -1)
					states_[i][j] = states_map[states[i][j]];
				else
					states_[i][j] = num_spin_ - 1;
			}
	}
	num_result_++;
}

void PottsModel::UpdateSegmentDepth()
{
	vector<int> segment_depth(num_spin_);
	vector<int> segment_count(num_spin_);
	for (int i = 0; i < color_.rows; i++)
		for (int j = 0; j < color_.cols; j++)
		{
			segment_depth[states_[i][j]] += depth_.at<uchar>(i, j);
			segment_count[states_[i][j]]++;
		}
	for (int i = 0; i < color_.rows; i++)
		for (int j = 0; j < color_.cols; j++)
		{
			segment_depth_.at<uchar>(i, j) = segment_depth[states_[i][j]]/segment_count[states_[i][j]];
		}
}

void PottsModel::GenBoundry()
{
	//initialize the boundry_ matrix
	boundry_ = Scalar::all(255);
	//time_print("", 0);
	int flag = 0;
	for (int i = 0; i < color_.rows; i++)
	{
		flag = 0;
		for (int j = 1; j < color_.cols; j++)
		{
			if (states_[i][j] != states_[i][j-1]) {
				if (flag) {
					boundry_.at<uchar>(i, j) = 0;
				} else {
					flag = 1;
				}
			} else {
				flag = 0;
			}
		}
	}
	for (int i = 0; i < color_.cols; i++)
	{
		flag = 0;
		for (int j = 1; j < color_.rows; j++)
		{
			if (states_[j][i] != states_[j-1][i]) {
				if (flag) {
					boundry_.at<uchar>(j, i) = 0;
				} else {
					flag = 1;
				}
			} else {
				flag = 0;
			}
		}
	}
	//time_print("CPU boundry generate");
}

void PottsModel::ShowDifference() const
{
	HorizontalColor();
	VerticalColor();
	RightDiagColor();
	LeftDiagColor();
	destroyWindow("HorizontalColor");
	destroyWindow("VerticalColor");
	destroyWindow("RightDiagColor");
	destroyWindow("LeftDiagColor");
}

void PottsModel::HorizontalColor() const
{
	Mat md(color_.rows, color_.cols, CV_8UC3);
	double d = 0;
	for(int i = 0; i < color_.rows; i++)
		for(int j = 0; j < color_.cols - 1; j++)
		{
			d = diff_.at<double>(i, j, 1);
			Vec3b c(static_cast<uchar>(d * 0.5), 180, 230);
			if (d < kWhite) c[1] = 0;
			md.at<Vec3b>(i, j) = c;
		}
	Mat result;
	cvtColor(md, result, CV_HSV2BGR);
	namedWindow("HorizontalColor");
	imshow("HorizontalColor", result);
	waitKey();
}

void PottsModel::VerticalColor() const
{
	Mat md(color_.rows, color_.cols, CV_8UC3);
	double d = 0;
	for(int i = 0; i < color_.rows - 1; i++)
		for(int j = 0; j < color_.cols; j++)
		{
			d = diff_.at<double>(i, j, 3);
			Vec3b c(static_cast<uchar>(d * 0.5), 180, 230);
			if (d < kWhite) c[1] = 0;
			md.at<Vec3b>(i, j) = c;
		}
	Mat result;
	cvtColor(md, result, CV_HSV2BGR);
	namedWindow("VerticalColor");
	imshow("VerticalColor", result);
	waitKey();
}

void PottsModel::RightDiagColor() const
{
	Mat md(color_.rows, color_.cols, CV_8UC3);
	double d = 0;
	for(int i = 0; i < color_.rows - 1; i++)
		for(int j = 1; j < color_.cols; j++)
		{
			d = diff_.at<double>(i, j, 7);
			Vec3b c(static_cast<uchar>(d * 0.5), 180, 230);
			if (d < kWhite) c[1] = 0;
			md.at<Vec3b>(i, j) = c;
		}
	Mat result;
	cvtColor(md, result, CV_HSV2BGR);
	namedWindow("RightDiagColor");
	imshow("RightDiagColor", result);
	waitKey();
}

void PottsModel::LeftDiagColor() const
{
	Mat md(color_.rows, color_.cols, CV_8UC3);
	double d = 0;
	for(int i = 0; i < color_.rows - 1; i++)
		for(int j = 0; j < color_.cols - 1; j++)
		{
			d = diff_.at<double>(i, j, 5);
			Vec3b c(static_cast<uchar>(d * 0.5), 180, 230);
			if (d < kWhite) c[1] = 0;
			md.at<Vec3b>(i, j) = c;
		}
	Mat result;
	cvtColor(md, result, CV_HSV2BGR);
	namedWindow("LeftDiagColor");
	imshow("LeftDiagColor", result);
	waitKey();
}

double PottsModel::Distance(const Vec3b &a, const Vec3b &b) const
{
	if (color_space_ == HSV){
		return hsv_distance(a, b);
	}
	else if (color_space_ == RGB) return norm(static_cast<Vec3s>(a) - static_cast<Vec3s>(b));
	else return 0;
}


GpuPottsModel::GpuPottsModel(const cv::Mat &color, const cv::Mat &depth, int color_space)
	:BasicPottsModel(color, depth, color_space), labels_(rows_, cols_, CV_32S),
	INFI(256255), gpu_color_(color_), old_depth_(rows_, cols_, CV_8U),
	gpu_new_color_(rows_, cols_, CV_8UC3), gpu_new_gray_(rows_, cols_, CV_8U),
	BOUNDRY_THRESHOLD(700), MIN_LABEL_COUNT(100), STATES_KEEP_THRESHOLD(0.3), DEPTH_THRESHOLD(30)
{
	diff_ = new float[rows_ * cols_][8];
	states_ = new uchar[rows_ * cols_];
	boundry_ = new uchar[rows_ * cols_];
	cvtColor(gpu_color_, gpu_gray_, CV_BGR2GRAY);
	if(depth.isContinuous())
	{
		memcpy(states_, depth.data, rows_ * cols_);
		depth.copyTo(old_depth_);
	}
	else
	{
		cout<<"depth mat is not continuous! reload the depth image"<<endl;
		exit(0);
	}
	ComputeDifference();
}

void GpuPottsModel::LoadNextFrame(const Mat &color, const Mat &depth, int color_space)
{
	time_print("",0);
	color_ = color;
	depth_ = depth;

	start_frame_++;
	num_result_ = 0;
	num_result_gen_ = -1;
	color_space_ = color_space;

	gpu_new_color_.upload(color);	
	cvtColor(gpu_new_color_, gpu_new_gray_, CV_BGR2GRAY);

	FarneCalc(gpu_gray_, gpu_new_gray_, d_flowx, d_flowy);

	LoadNextFrameWithCuda(states_, depth_.data, PtrStep<float>(d_flowx), PtrStep<float>(d_flowy), rows_, cols_);

	GpuMat temp = gpu_gray_;
	gpu_gray_ = gpu_new_gray_;
	gpu_new_gray_ = temp;

	ComputeDifferenceWithCuda((const unsigned char (*)[3])color_.data, depth_.data, diff_, rows_, cols_);
}

GpuPottsModel::~GpuPottsModel()
{
	delete [] diff_;
	delete [] states_;
	delete [] boundry_;
}

void GpuPottsModel::ComputeDifference()
{
	CV_Assert(color_.type() == CV_8UC3);

	ComputeDifferenceWithCuda((const unsigned char (*)[3])color_.data, depth_.data, diff_, rows_, cols_);
}

void GpuPottsModel::Metropolis()
{
	//time_print("", 0);
	while(iterable())
	{	
		MetropolisOnce();
	}
	//time_print("Metropolis");
}

void GpuPottsModel::MetropolisOnce()
{
	MetropolisOnceWithCuda(t_, states_, rows_, cols_); 
	t_ *= a_c_;
	num_result_++;
}

double GpuPottsModel::PixelEnergy(int pi, int pj) const
{
	return 0;
}

void GpuPottsModel::GenStatesResult()
{
	if (num_result_gen_ >= num_result_) return;
	int states;
	Mat hsv(rows_, cols_, CV_8UC3);
	MatIterator_<Vec3b> it = hsv.begin<Vec3b>(), end = hsv.end<Vec3b>();
	for (int i = 0; i < rows_; i++)
		for (int j = 0; j < cols_; j++)
		{
			states = states_[i * cols_ + j];
			//map the state to the result array
			Vec3b c(static_cast<uchar>(kStatesResult[states] * 0.6), 180, 230);
			//Vec3b c(static_cast<uchar>(states * 0.6), 180, 230);
			//if the state is num_spin_ - 1, the pixel is on the boundry line.
			//so use black
			if (states == num_spin_ - 1) c[2] = 0;
			(*it) = c;
			it++;
		}
	cvtColor(hsv, states_result_, CV_HSV2BGR);
	num_result_gen_ = num_result_;
}

void GpuPottsModel::ShowStates(int milliseconds)
{
	GenStatesResult();
	imshow("GpuPottsModel", states_result_);
	waitKey(milliseconds);
}

void GpuPottsModel::SaveStates(const string &title)
{
	GenStatesResult();
	if (title == "")
	{
		char result_name[20];
		sprintf(result_name, "GPUresult%d_%d.jpg",start_frame_, num_result_);
		string s(result_name);
		imwrite(s, states_result_);
	}
	else
	{
		imwrite(title, states_result_);
	}
}

void GpuPottsModel::GenBoundry()
{
	GenBoundryWithCuda(boundry_, rows_, cols_);
	//eliminate the small boundry use fast label
	memset(label_table, 0, 10000*sizeof(int));
	int m = 1;
	int c1, c3;
	//First Scan
	for(int i = 0; i < rows_; i++)
		for(int j = 0; j < cols_; j++)
		{
			if (boundry_[i * cols_ + j] == 255) labels_.at<int>(i, j) = INFI;
			else if ((c3 = c(i, j, 3)) != INFI)
			{
				labels_.at<int>(i, j) = c3;
				if ((c1 = c(i, j, 1)) != INFI) 
				{ 
					if (c3 != c1) Resolve(c3, c1);
				} 
			}
			else if ((c1 = c(i, j, 1)) != INFI)
			{
				labels_.at<int>(i, j) = c1; 
			}
			else
			{
				//prevent the label set 255
				labels_.at<int>(i, j) = (m % 256 == 255 ? (m+=2) : m);
				m++;
			}
		}
	//label_table
	int label;
	for(int i = 1; i < m; i++)
	{
		label = i;
		while(label_table[label]) label = label_table[label];
		final_label[i] = label;
	}
	//Second Scan
	memset(label_count, 0, 10000*sizeof(int));
	for(int i = 0; i < rows_; i++)
		for(int j = 0; j < cols_; j++)
		{
			label = labels_.at<int>(i, j);
			if (label != INFI)
			{
				label_count[final_label[label]]++;
			}
		}
	for(int i = 0; i < rows_; i++)
		for(int j = 0; j < cols_; j++)
		{
			label = labels_.at<int>(i, j);
			if (label != INFI && label_count[final_label[label]] < BOUNDRY_THRESHOLD)
			{
				boundry_[i * cols_ + j] = 255;
			}
		}
}

void GpuPottsModel::Label()
{
	//time_print("", 0);
	memset(label_table, 0, 10000*sizeof(int));
	memset(label_count, 0, 10000*sizeof(int));
	int m = 1;
	int c1, c2, c3, c4;
	//First Scan
	for(int i = 0; i < rows_; i++)
		for(int j = 0; j < cols_; j++)
		{
			if (boundry_[i * cols_ + j] == 0) labels_.at<int>(i, j) = INFI;
			else if ((c3 = c(i, j, 3)) != INFI) labels_.at<int>(i, j) = c3;
			else if ((c1 = c(i, j, 1)) != INFI)
			{
				labels_.at<int>(i, j) = c1;
				if ((c4 = c(i, j, 4)) != INFI) 
				{ 
					if (c4 != c1) Resolve(c4, c1);
				} 
			}
			else if ((c2 = c(i, j, 2)) != INFI)
			{
				labels_.at<int>(i, j) = c2;
				if ((c4 = c(i, j, 4)) != INFI) 
				{
					if (c2 != c4) Resolve(c2, c4);
				}
			}
			else if ((c4 = c(i, j, 4)) != INFI) { labels_.at<int>(i, j) = c4; }
			else
			{
				//prevent the label set 255
				labels_.at<int>(i, j) = (m % 256 == 255 ? (m+=2) : m);
				m++;
			}
		}
	//time_print("First scan time");
	//label_table
	int label;
	for(int i = 1; i < m; i++)
	{
		label = i;
		while(label_table[label]) label = label_table[label];
		final_label[i] = label;
	}
	//time_print("final_label time");
	//Second Scan
	if (start_frame_ == 1)
	{
		for(int i = 0; i < rows_; i++)
			for(int j = 0; j < cols_; j++)
			{
				label = labels_.at<int>(i, j);
				if (label == INFI) states_[i * cols_ + j] = num_spin_ - 1;
				else 
				{
					label_count[final_label[label]]++;
				}
			}
		uchar last_states = 1;
		for(int i = 0; i < rows_; i++)
			for(int j = 0; j < cols_; j++)
			{
				label = labels_.at<int>(i, j);
				if (label != INFI)
				{
					if (label_count[final_label[label]] > MIN_LABEL_COUNT)
						states_[i * cols_ + j] = last_states = final_label[label] % num_spin_;
					else
						states_[i * cols_ + j] = last_states;
				}
			}
	}
	else
	{
		CopyStatesToHost(states_, rows_, cols_);
		memset(label_table, 0, 10000*sizeof(int));
		memset(label_tmp, 0, 10000*sizeof(int));
		memset(label_count2, 0, 10000*sizeof(int));
		int s, old_s, last_s;
		for (int i = 0; i < color_.rows; i++)
			for (int j = 0; j < color_.cols; j++)
			{
				label = labels_.at<int>(i, j);
				if (label == INFI) continue;
				else if(abs(depth_.at<uchar>(i, j) - old_depth_.at<uchar>(i, j)) > DEPTH_THRESHOLD) continue;
				else
				{
					s = final_label[label];
					label_count2[s]++;
					//states of this position in last frame
					last_s = states_[i * cols_ + j];
					if ((old_s = label_table[s]) != 0)
					{
						if (old_s == last_s) label_count[s]++;
						else if (label_tmp[s] == last_s)
						{
							if (label_count[s] == 0) label_table[s] = last_s;
							else label_count[s]--;
						}
						else
						{
							label_tmp[s] = last_s;
						}
					}
					else
					{
						label_table[s] = states_[i * cols_ + j];
					}
				}
			}
		uchar last_states = 1;
		for (int i = 0; i < color_.rows; i++)
			for (int j = 0; j < color_.cols; j++)
			{
				label = labels_.at<int>(i, j);
				s = final_label[label];
				if (label == INFI) states_[i * cols_ + j] = num_spin_ - 1;
				else if (label_count2[s] < MIN_LABEL_COUNT)
					states_[i * cols_ + j] = last_states;
				else if (label_count[s] > (unsigned int)(label_count2[s] * STATES_KEEP_THRESHOLD))
					states_[i * cols_ + j] = last_states = label_table[s];
				else
				{
					cout << i << " " << j << " " << label_count[s] * 1.0 / label_count2[s] << endl;
					states_[i * cols_ + j] = last_states = (label_table[s] == 1 ? 2 : label_table[s] - 1);
				}
			}
	}
	num_result_++;
	//time_print("Second Scan time");
}

void GpuPottsModel::CopyStates()
{
	CopyStatesToDevice(states_, rows_, cols_);
}

void GpuPottsModel::LoadStates()
{
	CopyStatesToHost(states_, rows_, cols_);
}

void GpuPottsModel::Resolve(int m, int n)
{
	while (label_table[m]) m = label_table[m];
	while (label_table[n]) n = label_table[n];

	if (m < n) label_table[n] = m;
	else if (m > n) label_table[m] = n;
}

int GpuPottsModel::c(int x, int y, int n) const
{
	x += kC[n][0];
	y += kC[n][1];

	if (x < 0 || y < 0 || y == cols_) return INFI;
	return labels_.at<int>(x, y);
}