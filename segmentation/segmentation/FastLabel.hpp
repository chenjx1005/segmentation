#ifndef SEGMENTATION_FASTLABEL_H
#define SEGMENTATION_FASTLABEL_H

#include <cstdio>
#include <set>
#include <list>

#include "opencv2\opencv.hpp"

class FastLabel
{
public:
	typedef cv::Matx<int, 5, 2> Matx52i;
	FastLabel(const cv::Mat &boundry);
	virtual ~FastLabel() {};
	void Resolve(const cv::Point &p, int a, int b);
	void FirstScan();
	void SecondScan();
	const std::vector<std::vector<int> > & get_labels() const { return labels_; }

private:
	//set the state of the pixel on the boundry line INFI
	//INFI % 256 = 255.  
	const int INFI;
	//return the label of c[n] of the current pixel
	//no pixel && boundry: INFI
	int c(const cv::Point &p, int n)
	{
		CV_Assert(n >= 0 && n < 5);
		cv::Point position(p.x + kC_(n, 0), p.y + kC_(n, 1));
		if (position.x < 0 || position.x >= boundry_.rows ||
			position.y < 0 || position.y >= boundry_.cols) return INFI;
		return labels_[position.x][position.y];
	}

	//the min label number
	int m_;
	std::list<std::set<int> > label_tables_;
	cv::Mat boundry_;
	//label of every pixel
	std::vector<std::vector<int> > labels_;
	const Matx52i kC_;
};
#endif

