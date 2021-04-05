#ifndef UTILITIES_HPP    // To make sure you don't declare the function more than once by including the header multiple times.
#define UTILITIES_HPP
#include <opencv2/opencv.hpp>


class Utility{

//	Utility();
public:
	static void colorize_flow(const cv::Mat &u, const cv::Mat &v, std::string filename="img.png");
	static void play_video(std::vector<cv::Mat> &video, float fps);
	static void save_video(std::vector<cv::Mat> &video, float fps, std::string name="outcpp.avi", std::string path="./");
	static void frms2vis(std::string path);
	static void frms2vis(std::string path, std::vector<int>& frms);
	static void side_two_vids(std::string path_1, std::string path_2, std::string dest="./sided.avi");


	const static int FRST = 2100; // Start frame
	const static int LAST =  15000; // Last Frame

	// weights
	static double ALPHA(){ return 3.0;} // opic
	static double BETA() { return 1.0;} // block
	static double GAMMA(){ return 0.1;} // EDM



};


#endif
