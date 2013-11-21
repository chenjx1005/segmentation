#include <stdio.h>
#include <cmath>
#include <vector>
#include "cv.h"
#include "highgui.h"

using namespace cv;
using namespace std;

const double ALPHA = 1;
const int Q = 128;
const int STEP = 1;
const double T = 1.3806488e+8, MIN_T = 1;
const double A_C = 0.553;
const double K = 1.3806488e-4;//Boltzmann constant

double g_meandiff; 

double PixelEnergy(const Mat &, const Mat &, int, int, const vector<vector<int>> &, double mean = g_meandiff);
double SumEnergy(const Mat &);
double MeanDiff(const Mat &);
void ShowResult(const vector<vector<int>> &);
double HSVNorm(const Mat &, Mat &);

int main(int, char**)
{
	namedWindow("show");
	Mat img = imread("Color0.png");
	Mat depth, fusion;
	cvtColor(imread("Depth0.png"), depth, CV_BGR2GRAY);
	cvtColor(imread("Fusion0.png"), fusion, CV_BGR2GRAY);
	//compute mean difference
	g_meandiff =  MeanDiff(img);
	//initialize the spin states
	vector<vector<int>> states(img.rows, vector<int>(img.cols));
	RNG r;
	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
			states[i][j] = fusion.at<char>(i, j);
	ShowResult(states);
	//initialize the energy matrix
	vector<vector<double>> energy(img.rows, vector<double>(img.cols));
	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
			energy[i][j] = PixelEnergy(img, depth, i, j, states);
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
					double e_i = PixelEnergy(img, depth, i, j, states);
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

double MeanDiff(const Mat &img)
{
	CV_Assert(img.type() == CV_8UC3);

	int w = img.cols;
	int h = img.rows;
	double sum = 0;
	int count = 0;
	Vec3b gi, gj;
	//horizontal diff
	for(int i = 0; i < h; i++)
	{
		const Vec3b *p = img.ptr<Vec3b>(i);
		gi = p[0];
		for(int j = 1; j < w; j++)
		{
			count++;
			gj = p[j];
			sum += norm(Vec3s(gi) - Vec3s(gj));
			gi = gj;
		}
	}
	//vertical diff
	for(int j = 0; j < w; j++)
	{
		gi = img.at<Vec3b>(0, j);
		for(int i = 1; i < h; i++)
		{
			count++;
			gj = img.at<Vec3b>(i, j);
			sum += norm(Vec3s(gi) - Vec3s(gj));
			gi = gj;
		}
	}
	//right diagonal diff
	for(int d = 0; d < w+h-1 ; d++)
	{
		int flag = 0;
		for(int i = 0; i <= d; i++)
		{
			int j = d - i;
			if(i < h && j < w)
			{
				gj = img.at<Vec3b>(i, j);
				if(!flag) flag = 1;
				else 
				{
					count++;
					sum += norm(Vec3s(gi) - Vec3s(gj));
				}
				gi = gj;
			}
		}
	}
	//left diagonal diff
	for(int n = -1*h+2; n < w-1; n++)
	{
		Mat d = img.diag(n);
		const Vec3b *p = d.ptr<Vec3b>(0);
		gi = p[0];
		for(int j = 1; j < d.rows; j++)
		{
			count++;
			gj = *d.ptr<Vec3b>(j);
			sum += norm(Vec3s(gi) - Vec3s(gj));
			gi = gj;
		}
	}
	printf("sum is %lf, count is %d\n", sum, count);
	return ALPHA * sum / count;
}

double PixelEnergy(const Mat &img, 
				   const Mat &depth,
				   int pi,
				   int pj,
				   const vector<vector<int>> &spin,
				   double mean)
{
	int ki,kj;
	double Jij, energy = 0;
	int dp = depth.at<char>(pi, pj);
	int dk;
	for(int i = -1; i <= 1; i++)
		for(int j = -1; j <= 1; j++)
		{
			if(0 == i && 0 == j) continue;
			ki = pi + i;
			kj = pj + j;
			if(ki < 0 || ki >= img.rows || kj < 0 || kj >= img.cols) continue;
			dk = depth.at<char>(ki,kj);
			if(30 < abs(dk - dp))
				Jij = 250;
			else
				Jij = norm(Vec3s(img.at<Vec3b>(pi,pj)) - Vec3s(img.at<Vec3b>(ki,kj))) / mean - 1;
			energy += Jij * (spin[pi][pj] == spin[ki][kj]);
		}
	return energy;
}

double HSVNorm(const Mat &src, Mat &dst)
{
	return 0;
}

double SumEnergy(const IplImage &img)
{
	return 0;
}