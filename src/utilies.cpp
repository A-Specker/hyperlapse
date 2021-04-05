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

vector<int> get_frms(){
	vector<int> fr{
		2100,
		2111,
		2123,
		2136,
		2149,
		2162,
		2175,
		2188,
		2201,
		2214,
		2227,
		2240,
		2251,
		2264,
		2277,
		2289,
		2302,
		2315,
		2327,
		2340,
		2352,
		2365,
		2378,
		2389,
		2402,
		2415,
		2428,
		2440,
		2450,
		2463,
		2476,
		2488,
		2501,
		2514,
		2527,
		2539,
		2552,
		2558,
		2571,
		2584,
		2597,
		2609,
		2622,
		2635,
		2648,
		2661,
		2673,
		2686,
		2699,
		2710,
		2720,
		2733,
		2746,
		2759,
		2772,
		2785,
		2793,
		2806,
		2819,
		2832,
		2845,
		2858,
		2871,
		2884,
		2897,
		2910,
		2923,
		2936,
		2949,
		2959,
		2969,
		2977,
		2989,
		2999,
		3012,
		3025,
		3038,
		3051,
		3063,
		3076,
		3089,
		3099,
		3112
	};
	return fr;
}

void Utility::frms2vis(string path){
	vector<int> frms = get_frms();
	cout << "Save Video with " << frms.size() << " frames." << endl;

	string dest = "./short_vid .avi";

	VideoCapture cap(path);
	if( !cap.isOpened() )
		cout << "Bitch" << endl;
	vector<Mat> video;

	for(unsigned int i=0; i<frms.size(); i++){
		Mat tmp;
		cap.set(CV_CAP_PROP_POS_FRAMES, (double)frms[i]);
		cap >> tmp;
		video.push_back(tmp);

	}
	VideoWriter vid_writer(dest, CV_FOURCC('M','J','P','G'), 24.0f, video[0].size());

	for(unsigned int i=0; i<video.size(); i++){
		vid_writer.write(video[i]);
	}
	vid_writer.release();
	cout << "Video written at " << dest << endl;
}

void Utility::frms2vis(string path, vector<int>& frms){
	cout << "Save Video with " << frms.size() << " frames." << endl;

	string dest = "./15000_vpp .avi";

	VideoCapture cap(path);
	if( !cap.isOpened() )
		cout << "Bitch" << endl;
	vector<Mat> video;

	for(unsigned int i=0; i<frms.size(); i++){
		Mat tmp;
		cap.set(CV_CAP_PROP_POS_FRAMES, (double)frms[i]);
		cap >> tmp;
		video.push_back(tmp);

	}

	VideoWriter vid_writer(dest, CV_FOURCC('M','J','P','G'), 24.0f, video[0].size());

	for(unsigned int i=0; i<video.size(); i++){
		vid_writer.write(video[i]);
	}
	vid_writer.release();
	cout << "Video written at " << dest << endl;
}
/*
 * Doesnt work
 */
void Utility::side_two_vids(string p1, string p2, string dest){
	VideoCapture c1(p1);
	VideoCapture c2(p2);

	if (!(c1.isOpened() && c2.isOpened())){
		cout << "Error loading videos" << endl;
		return;
	}
	//2*w, h
	Size siz(c1.get(3)+c2.get(3), c1.get(4));

	VideoWriter writer(dest, CV_FOURCC('M','J','P','G'), 24.0, siz);

	int more = c1.get(7) > c2.get(7) ?  c1.get(7) :  c2.get(7);

	for(int i=0; i<more; i++){
		Mat m, m1, m2;

		// if one vid is longer, make the other black
		if(i >= c1.get(7))
			m1 = (c1.get(4), c1.get(3), CV_8UC3, Scalar(0,0,0));
		else
			c1 >> m1;

		if(i >= c2.get(7))
			m2 = (c2.get(4), c2.get(3), CV_8UC3, Scalar(0,0,0));
		else
			c2 >> m2;

		hconcat(m1, m2, m);
		writer.write(m);
	}
	writer.release();
	c1.release();
	c2.release();
}
