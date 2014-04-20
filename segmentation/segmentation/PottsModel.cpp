#include <limits>
#include <map>
#include <algorithm>

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
	printf("sum is %lf, count is %d, mean_diff is %lf when alpha=%lf\n", sum, count, mean_diff_, alpha_);
	if (fabs(mean_diff_) >= EPSILON)
		diff_ = diff_ * (1 / mean_diff_) - Scalar::all(1);
	else diff_ -= Scalar::all(1);
	for (list<Vec3s>::const_iterator it = later_update.begin(); it != later_update.end(); it++)
	{
		diff_.at<double>((*it)[0], (*it)[1], (*it)[2]) = kMaxJ;
	}
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
	:BasicPottsModel(color, depth, color_space)
{
	if(depth.isContinuous())
		states_ = depth.data;
	else
	{
		cout<<"depth mat is not continuous! reload the depth image"<<endl;
		exit(0);
	}
	ComputeDifference();
}

GpuPottsModel::~GpuPottsModel()
{
	delete [] diff_;
}

void GpuPottsModel::ComputeDifference()
{
	CV_Assert(color_.type() == CV_8UC3);

	diff_ = new float[rows_ * cols_][8];
	ComputeDifferenceWithCuda((const unsigned char (*)[3])color_.data, depth_.data, diff_, rows_, cols_);
}

void GpuPottsModel::MetropolisOnce()
{
}

double GpuPottsModel::PixelEnergy(int pi, int pj) const
{
	return 0;
}