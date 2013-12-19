#include <cmath>

#include "PottsModel.hpp"

using namespace std;
using namespace cv;

namespace {
//if the color difference < kwhite, show white
const double kWhite = 0;
double hsv_distance(const Vec3b &a, const Vec3b &b)
{
	Vec3d ad(a[1] * a[2] / 65025.0 * cos(double(a[0] * 2)),
			 a[1] * a[2] / 65025.0 * sin(double(a[0] * 2)),
			 a[2] / 255.0);
	Vec3d bd(b[1] * b[2] / 65025.0 * cos(double(b[0] * 2)),
			 b[1] * b[2] / 65025.0 * sin(double(b[0] * 2)),
			 b[2] / 255.0);
	return norm(ad - bd);
	//return norm(static_cast<Vec3s>(a) - static_cast<Vec3s>(b));
}
}

PottsModel::PottsModel(const Mat &color, const Mat &depth)
	:alpha_(1), num_spin_(256), init_t_(1.3806488e+7), min_t_(0.1), a_c_(0.33),
	t_(init_t_), kK(1.3806488e-4), kMaxJ(250), num_result_(0),
	num_result_gen_(-1), color_(color), depth_(depth),
	boundry_(color.rows, color.cols, CV_8U, Scalar::all(255)),
	states_(color.rows, vector<int>(color.cols)),
	kPixel(4, 2, 6, 0, -1, 1, 7, 3 ,5)
{
	for (int i = 0; i < color_.rows; i++)
		for (int j = 0; j < color_.cols; j++)
			states_[i][j] = depth.at<char>(i, j);
	ComputeDifference();
	namedWindow("PottsModel");
}

PottsModel::PottsModel(const Mat &color)
	:alpha_(1), num_spin_(256), init_t_(1.3806488e+7), min_t_(0.1), a_c_(0.33),
	t_(init_t_), kK(1.3806488e-4), kMaxJ(250), num_result_(0),
	num_result_gen_(-1), color_(color), depth_(color.rows, color.cols, CV_8U, Scalar::all(0)),
	boundry_(color.rows, color.cols, CV_8U, Scalar::all(255)),
	states_(color.rows, vector<int>(color.cols)),
	kPixel(4, 2, 6, 0, -1, 1, 7, 3 ,5)
{
	RNG r;
	for (int i = 0; i < color_.rows; i++)
		for (int j = 0; j < color_.cols; j++)
			states_[i][j] = r.next() % num_spin_;
	ComputeDifference();
	namedWindow("PottsModel");
}

PottsModel::~PottsModel()
{
	destroyWindow("PottsModel");
}

void PottsModel::ComputeDifference()
{
	CV_Assert(color_.type() == CV_8UC3);
	CV_Assert(color_.size == depth_.size);

	//initialize diff matrix
	int sz[3] = {color_.rows, color_.cols, 8};
	diff_.create(3, sz, CV_64F);

	double sum = 0, color_diff = 0;
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
			color_diff = hsv_distance(gi, gj);
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
			color_diff = hsv_distance(gi, gj);
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
					color_diff = hsv_distance(gi, gj);
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
			color_diff = hsv_distance(gi, gj);
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
	diff_ = diff_ * (1 / mean_diff_) - 1;
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
	Mat hsv(rows, cols, CV_8UC3);
	MatIterator_<Vec3b> it = hsv.begin<Vec3b>(), end = hsv.end<Vec3b>();
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			Vec3b c(static_cast<uchar>(states_[i][j] * 0.6), 180, 230);
			if (states_[i][j] == 255) c[2] = 0;
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

void PottsModel::SaveStates()
{
	GenStatesResult();
	char result_name[20];
	sprintf(result_name, "result%d.jpg", num_result_);
	string s(result_name);
	imwrite(s, states_result_);
}

void PottsModel::UpdateStates(const vector<vector<int> > &states)
{
	CV_Assert(states.size() == color_.rows);
	for (int i = 0; i < states.size(); i++){
		CV_Assert(states[i].size() == color_.cols);
	}
	states_ = states;
	for (int i = 0; i < color_.rows; i++)
		for (int j = 0; j < color_.cols; j++)
		{
			states_[i][j] %= num_spin_;
		}
	num_result_++;
}

void PottsModel::GenBoundry()
{
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
	imshow("PottsModel", result);
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
	imshow("PottsModel", result);
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
	imshow("PottsModel", result);
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
	imshow("PottsModel", result);
	waitKey();
}