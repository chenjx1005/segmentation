#ifndef SEGMENTATION_FASTLABEL_H
#define SEGMENTATION_FASTLABEL_H

#include <cstdio>
#include <set>
#include <list>

#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

typedef Matx<int, 5, 2> Matx52i;

const int INFI = 0x7FFFFFFF;

class FastLabel
{
public:
	FastLabel(const Mat &boundry);
	virtual ~FastLabel() {};
	void Resolve(const Point &p, int a, int b);
	void FirstScan();
	void SecondScan();
	const vector<vector<int> > & get_labels() const { return labels_; }

private:
	//return the label of c[n] of the current pixel
	//no pixel && boundry: INFI
	int c(const Point &p,int n)
	{
		CV_Assert(n >= 0 && n < 5);
		Point position(p.x + kC_(n, 0), p.y + kC_(n, 1));
		if (position.x < 0 || position.x >= boundry_.rows ||
			position.y < 0 || position.y >= boundry_.cols) return INFI;
		return labels_[position.x][position.y];
	}

	//the min label number
	int m_;
	list<set<int> > label_tables_;
	Mat boundry_;
	//label of every pixel
	vector<vector<int> > labels_;
	const Matx52i kC_;
};
#endif

