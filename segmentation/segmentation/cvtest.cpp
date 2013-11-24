#include <stdio.h>
#include <cmath>
#include <vector>
#include "cv.h"
#include "highgui.h"

using namespace cv;
using namespace std;

typedef Vec<double, 8> Vec8d;

const double ALPHA = 1;
const int Q = 256;
const int STEP = 1;
const double T = 1.3806488e+8, MIN_T = 1;
const double A_C = 0.553;
const double K = 1.3806488e-4;//Boltzmann constant
const int PIXEL[3][3] = {{4, 2, 6}, {0, -1, 1}, {7, 3 ,5}};
const int MAX_J = 250;

double g_meandiff; 

double PixelEnergy(const Mat &, int, int, const vector<vector<int>> &);
double SumEnergy(const Mat &);
void ShowResult(const vector<vector<int>> &);
double HSVNorm(const Vec3b &, const Vec3b &);
double DifferenceMap(const Mat &, Mat &, int);//compute difference map and return the mean difference
void DifferenceDepth(const Mat &, Mat &);

int main(int, char**)
{
	namedWindow("show");
	/*test
	uchar a = 123;
	uchar b = 245;
	uchar c = a-b;
	cout<< c << endl;
	*/
	Mat img = imread("Color0.png");
	Mat depth, fusion, diff;
	cvtColor(imread("Depth0.png"), depth, CV_BGR2GRAY);
	cvtColor(imread("Fusion0.png"), fusion, CV_BGR2GRAY);
	//compute mean difference and difference map
	g_meandiff =  DifferenceMap(img, diff , 1);
	diff = diff * (1 / g_meandiff) - 1;
	DifferenceDepth(depth, diff);
	//initialize the spin states
	vector<vector<int>> states(img.rows, vector<int>(img.cols));
	RNG r;
	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
			states[i][j] = depth.at<char>(i, j);
	ShowResult(states);
	//initialize the energy matrix
	vector<vector<double>> energy(img.rows, vector<double>(img.cols));
	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
			energy[i][j] = PixelEnergy(diff, i, j, states);
	//Metropolis algorithm
	double t = T;
	while(t > MIN_T)
	{		
		for(int i = 0; i < img.rows; i++)
		{
			for(int j = 0; j < img.cols; j++)
			{
				int s = states[i][j];
				double min_e = 100000000;
				int min_m = 0;
				for(int m = 0; m < Q; m += STEP)
				{
					if(s == m) continue;
					states[i][j] = m;
					double e_i = PixelEnergy(diff, i, j, states);
					if(e_i < min_e)
					{
						min_e = e_i;
						min_m = m;
					}
				}
				double diff_e = min_e - energy[i][j];
				double p = exp(-1 * abs(diff_e) / (t * K));
				double y = (double)(r.next()%1000) / 1000.0;
				if(diff_e < 0 || y < p)
				{
					//if(diff_e > 0) printf("t is %lf move is accepted and diff_e is %lf\n", t, diff_e);
					states[i][j] = min_m;
					energy[i][j] = min_e;
				}
				else 
				{
					//printf("!!move is not accepted and diff_e is %lf\n", diff_e);
					states[i][j] = s;
				}
			}
			printf("t is %lf, row %d is finished\n", t, i);
		}
		ShowResult(states);
		t *= A_C;
	}
	//output the states
	
    return 0;
}

void ShowResult(const vector<vector<int>> &states)
{
	int rows = states.size();
	int cols = states[0].size();
	Mat hsv(rows, cols, CV_8UC3);
	MatIterator_<Vec3b> it = hsv.begin<Vec3b>(), end = hsv.end<Vec3b>();
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++)
		{
			Vec3b c(states[i][j], 255, 255);
			(*it) = c;
			it++;
		}
	Mat result;
	cvtColor(hsv, result, CV_HSV2BGR);
	imwrite("result.jpg", result);
	imshow("show", result);
	waitKey(5000);
}

double PixelEnergy(const Mat &img, 
				   int pi,
				   int pj,
				   const vector<vector<int>> &spin)
{
	int ki,kj;
	double energy = 0;
	for(int i = -1; i <= 1; i++)
		for(int j = -1; j <= 1; j++)
		{
			if(0 == i && 0 == j) continue;
			ki = pi + i;
			kj = pj + j;
			if(ki < 0 || ki >= img.size[0] || kj < 0 || kj >= img.size[1]) continue;
			energy += img.at<double>(pi, pj, PIXEL[i + 1][j + 1]) * (spin[pi][pj] == spin[ki][kj]);
		}
	return energy;
}

double HSVNorm(const Vec3b &a, const Vec3b &b)
{
	Vec3d ad = Vec3d(a[1] * a[2] / 65025.0 * cos(double(a[0] * 2)), 
					 a[1] * a[2] / 65025.0 * sin(double(a[0] * 2)),
					 a[2] / 255.0);
	Vec3d bd = Vec3d(b[1] * b[2] / 65025.0 * cos(double(b[0] * 2)), 
					 b[1] * b[2] / 65025.0 * sin(double(b[0] * 2)),
					 b[2] / 255.0);
	return 0;
}

void DifferenceDepth(const Mat &depth, Mat &dst)
{
	//horizontal diff
	uchar gi, gj;
	for(int i = 0; i < depth.rows; i++)
	{
		const uchar *p = depth.ptr<uchar>(i);
		gi = p[0];
		for(int j = 1; j < depth.cols; j++)
		{
			gj = p[j];
			int diff = abs(int(gi) - int(gj));
			if(diff > 30)
				dst.at<double>(i, j-1, 1) = dst.at<double>(i, j, 0) = MAX_J;
			gi = gj;
		}
	}
	//vertical diff
	for(int j = 0; j < depth.cols; j++)
	{
		gi = depth.at<uchar>(0, j);
		for(int i = 1; i < depth.rows; i++)
		{
			gj = depth.at<uchar>(i, j);
			int diff = abs(int(gi) - int(gj));
			if(diff > 30)
				dst.at<double>(i-1, j, 3) = dst.at<double>(i, j, 2) = MAX_J;
			gi = gj;
		}
	}
	//right diagonal diff
	for(int d = 0; d < depth.cols + depth.rows - 1 ; d++)
	{
		int flag = 0;
		for(int i = 0; i <= d; i++)
		{
			int j = d - i;
			if(i < depth.rows && j < depth.cols)
			{
				gj = depth.at<uchar>(i, j);
				if(!flag) flag = 1;
				else 
				{
					int diff = abs(int(gi) - int(gj));
					if(diff > 30)
						dst.at<double>(i-1, d - i + 1, 7) = dst.at<double>(i, j, 6) = MAX_J;
				}
				gi = gj;
			}
		}
	}
	//left diagonal diff
	for(int n = -1 * depth.rows + 2; n < depth.cols - 1; n++)
	{
		Mat d = depth.diag(n);
		int pi,pj;
		if(n<0) pi = -n, pj=0;
		else pi = 0, pj = n;
		const uchar *p = d.ptr<uchar>(0);
		gi = p[0];
		for(int j = 1; j < d.rows; j++)
		{
			gj = *d.ptr<uchar>(j);
			int diff = abs(int(gi) - int(gj));
			if(diff > 30)
				dst.at<double>(pi + j - 1, pj + j - 1, 5) = dst.at<double>(pi + j, pj + j, 4) = MAX_J;
			gi = gj;
		}
	}
}

double DifferenceMap(const Mat &src, Mat &dst, int code)
{
	//input: RGB Mat src, destination Difference Mat
	CV_Assert(src.type() == CV_8UC3);
	//init dst mat
	if(dst.type() != CV_64F || dst.size != src.size)
	{
		int sz[3] = {src.rows, src.cols, 8};
		dst.create(3, sz, CV_64F);
	}
	for(MatIterator_<double> it = dst.begin<double>(); it != dst.end<double>(); it++)
		*it = -1;
	//some veriable
	double sum = 0;
	int count = 0;
	Vec3b gi, gj;
	//horizontal diff
	for(int i = 0; i < src.rows; i++)
	{
		const Vec3b *p = src.ptr<Vec3b>(i);
		gi = p[0];
		for(int j = 1; j < src.cols; j++)
		{
			count++;
			gj = p[j];
			double diff = norm(Vec3s(gi) - Vec3s(gj));
			sum += diff;
			dst.at<double>(i, j-1, 1) = dst.at<double>(i, j, 0) = diff;
			gi = gj;
		}
	}
	//vertical diff
	for(int j = 0; j < src.cols; j++)
	{
		gi = src.at<Vec3b>(0, j);
		for(int i = 1; i < src.rows; i++)
		{
			count++;
			gj = src.at<Vec3b>(i, j);
			double diff = norm(Vec3s(gi) - Vec3s(gj));
			sum += diff;
			dst.at<double>(i-1, j, 3) = dst.at<double>(i, j, 2) = diff;
			gi = gj;
		}
	}
	//right diagonal diff
	for(int d = 0; d < src.cols + src.rows - 1 ; d++)
	{
		int flag = 0;
		for(int i = 0; i <= d; i++)
		{
			int j = d - i;
			if(i < src.rows && j < src.cols)
			{
				gj = src.at<Vec3b>(i, j);
				if(!flag) flag = 1;
				else 
				{
					count++;
					double diff = norm(Vec3s(gi) - Vec3s(gj));
					sum += diff;
					dst.at<double>(i-1, d - i + 1, 7) = dst.at<double>(i, j, 6) = diff;
				}
				gi = gj;
			}
		}
	}
	//left diagonal diff
	for(int n = -1 * src.rows + 2; n < src.cols - 1; n++)
	{
		Mat d = src.diag(n);
		int pi,pj;
		if(n<0) pi = -n, pj=0;
		else pi = 0, pj = n;
		const Vec3b *p = d.ptr<Vec3b>(0);
		gi = p[0];
		for(int j = 1; j < d.rows; j++)
		{
			count++;
			gj = *d.ptr<Vec3b>(j);
			double diff = norm(Vec3s(gi) - Vec3s(gj));
			sum += diff;
			dst.at<double>(pi + j - 1, pj + j - 1, 5) = dst.at<double>(pi + j, pj + j, 4) = diff;
			gi = gj;
		}
	}
	printf("sum is %lf, count is %d\n", sum, count);
	return ALPHA * sum / count;
}

double SumEnergy(const IplImage &img)
{ 
	return 0;
}