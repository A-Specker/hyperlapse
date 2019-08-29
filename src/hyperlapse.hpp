#include <iostream>
#include <opencv2/opencv.hpp>
#include "utilities.hpp"

using namespace cv;
using namespace std;


class Hyperlapse{


public:
	VideoCapture cap;
	vector<Mat> video;
	vector<int> path;

	int done = 0;
	int skip = 0;



	int num_pix(){
		return cap.get(3) * cap.get(4);}

	float norm_pix(){
		return 1/(float)num_pix();}

};
