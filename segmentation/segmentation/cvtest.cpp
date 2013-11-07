#include <stdio.h>
#include <cmath>
#include <vector>
#include "cv.h"
#include "highgui.h"

using namespace cv;
using namespace std;

const double ALPHA = 1;

double g_meandiff; 

double PixelEnergy(const Mat &, int, int, const vector<vector<int>> &, double mean = g_meandiff);
double SumEnergy(const Mat &);
double MeanDiff(const Mat &);

int main(int, char**)
{
	RNG r;
	for(int i = 0; i < 10; i++) printf("%d ", r.next()%100);
	namedWindow("show");
	Mat img = imread("lena.jpg");
	Mat m = (Mat_<Vec3b>(2,2) << Vec3b(1,0,0), Vec3b(0,1,0), Vec3b(0,0,1), Vec3b(0,0,0));
	g_meandiff =  MeanDiff(img);
	vector<vector<int>> states(2,vector<int>(2,1));
	double e = PixelEnergy(m,0,1,states);
    return 0;
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
	for(int n = -1*h+1; n < w-1; n++)
	{
		Mat d = img.diag(n);
		const Vec3b *p = d.ptr<Vec3b>(0);
		gi = p[0];
		for(int j = 1; j < d.rows; j++)
		{
			count++;
			gj = p[j];
			sum += norm(Vec3s(gi) - Vec3s(gj));
			gi = gj;
		}
	}
	printf("sum is %lf, count is %d\n", sum, count);
	return ALPHA * sum / count;
}

double PixelEnergy(const Mat &img,
				   int pi,
				   int pj,
				   const vector<vector<int>> &spin,
				   double mean)
{
	int ki,kj;
	double Jij, energy = 0;
	for(int i = -1; i <= 1; i++)
		for(int j = -1; j <= 1; j++)
		{
			if(0 == i && 0 == j) continue;
			ki = pi + i;
			kj = pj + j;
			if(ki < 0 || ki >= img.rows || kj < 0 || kj >= img.cols) continue;
			Jij = norm(Vec3s(img.at<Vec3b>(pi,pj)) - Vec3s(img.at<Vec3b>(ki,kj))) / mean - 1;
			energy += Jij * (spin[pi][pj] == spin[ki][kj]);
		}
	return energy;
}

double SumEnergy(const IplImage &img)
{
	return 0;
}