#include "utilities.hpp"

using namespace cv;
using namespace std;



template <typename T>
inline T mapVal(T x, T a, T b, T c, T d) {
	x = ::max(::min(x, b), a);
	return c + (d-c) * (x-a) / (b-a);
}

void Utility::colorize_flow(const Mat &u, const Mat &v, string filename)
{
	Mat dst;
	double uMin, uMax;
	cv::minMaxLoc(u, &uMin, &uMax, 0, 0);
	double vMin, vMax;
	cv::minMaxLoc(v, &vMin, &vMax, 0, 0);
	uMin = ::abs(uMin); uMax = ::abs(uMax);
	vMin = ::abs(vMin); vMax = ::abs(vMax);
	float dMax = static_cast<float>(::max(::max(uMin, uMax), ::max(vMin, vMax)));

	dst.create(u.size(), CV_8UC3);
	for (int y = 0; y < u.rows; ++y){
		for (int x = 0; x < u.cols; ++x){
			dst.at<uchar>(y,3*x) = 0;
			dst.at<uchar>(y,3*x+1) = (uchar)mapVal(-v.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
			dst.at<uchar>(y,3*x+2) = (uchar)mapVal(u.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
		}
	}
	imshow("flow", dst);
	waitKey(0);
	cv::imwrite(filename, dst);
}

void Utility::play_video(vector<Mat> &video, float fps){
	for (unsigned int i=0; i<video.size(); ++i) {
		imshow("Frame", video[i]);
		waitKey(1/fps * 1000);
	}
}

void Utility::save_video(vector<Mat> &video, float fps, string name, string path){
	string dest = path + name;
	VideoWriter vid_writer(dest, CV_FOURCC('M','J','P','G'), fps, video[0].size());
	for(unsigned int i=0; i<video.size(); i++){
		vid_writer.write(video[i]);
	}
	vid_writer.release();
	cout << "Video written at " << dest << endl;
}

