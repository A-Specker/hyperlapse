#ifndef UTILITIES_HPP    // To make sure you don't declare the function more than once by including the header multiple times.
#define UTILITIES_HPP
#include <opencv2/opencv.hpp>


class Utility{

//	Utility();
public:
	static void colorize_flow(const cv::Mat &u, const cv::Mat &v, std::string filename="img.png");
	static void play_video(std::vector<cv::Mat> &video, float fps);
	static void save_video(std::vector<cv::Mat> &video, float fps, std::string name="outcpp.avi", std::string path="./");

	const static int FRST = 2100; // Start frame
	const static int LAST=  2701; // Last Frame

};


#endif
