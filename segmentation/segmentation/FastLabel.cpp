#include "FastLabel.hpp"

using namespace std;
using namespace cv;

FastLabel::FastLabel(const Mat &boundry)
	:m_(1), boundry_(boundry), kC_(0, 0, 0, -1, -1, -1, -1, 0, -1, 1),
	labels_(boundry.rows, vector<int>(boundry.cols)), INFI(256255)
	{
		uchar u;
		for(int i = 0; i < boundry.rows; i++)
			for(int j = 0; j < boundry.cols; j++)
			{
				u = boundry.at<uchar>(i, j); 
				if (u < 10) {
					labels_[i][j] = INFI;
				}
				else if (u > 245) labels_[i][j] = INFI - 1;
				else printf("boundry map error at%d, %d\n", i, j);
			}
	}

void FastLabel::Resolve(const Point &p, int a, int b)
{
	int label_a = c(p, a);
	int label_b = c(p, b);
	list<set<int> >::iterator i = label_tables_.end();
	for(list<set<int> >::iterator it = label_tables_.begin();
		it != label_tables_.end();){
		if ( it->count(label_a) || it->count(label_b)){
			if ( i == label_tables_.end()) {
				i = it++;
			} else {
				i->insert(it->begin(), it->end());
				label_tables_.erase(it);
				break;
			}
		} else {
			it++;
		}
	}
}

void FastLabel::FirstScan()
{
	int c1, c2, c3, c4;
	Point p;
	set<int> tmp_s;
	for (int i = 0; i < boundry_.rows; i++)
	{
		for (int j = 0; j < boundry_.cols; j++)
		{
			p.x = i, p.y = j;
			if (c(p, 0) == INFI) continue;
			if ((c3 = c(p, 3)) != INFI) { labels_[i][j] = c3; }
			else if (((c1 = c(p, 1)) != INFI)) {
				labels_[i][j] = c1;
				if (c(p, 4) != INFI) { Resolve(p, 4, 1); } 
			}
			else if ((c2 = c(p, 2)) != INFI) {
				labels_[i][j] = c2;
				if (c(p, 4) != INFI) { Resolve(p, 2, 4); }
			}
			else if ((c4 = c(p, 4)) != INFI) { labels_[i][j] = c4; }
			else {
				//prevent the label set 255
				labels_[i][j] = m_ % 256 == 255 ? ++m_ : m_;
				tmp_s.clear();
				tmp_s.insert(m_++);
				label_tables_.push_back(tmp_s);
			}
		}
	}
	cout<<m_<<endl;
}

void FastLabel::SecondScan()
{
	int label;
	for (int i = 0; i < boundry_.rows; i++)
		for (int j = 0; j < boundry_.cols; j++)
		{
			label = labels_[i][j];
			for (list<set<int> >::iterator it = label_tables_.begin();
				it != label_tables_.end();
				it++)
			{
				if (it->count(label)) 
				{
					labels_[i][j] = *(it->begin());
					break;
				}
			}
		}
}