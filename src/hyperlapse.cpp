#ifndef GPU
#define GPU true

#include <iostream>
#include <opencv2/opencv.hpp>
#include "utilities.hpp"
//#include "opencv2/features2d.hpp"
//#include "opencv2/cudafeatures2d.hpp"
//#include "opencv2/xfeatures2d/cuda.hpp"

using namespace cv;
using namespace std;
using namespace cv::cuda;


VideoCapture load_video(String path, vector<Mat>& vid, int start_frame=Utility::FRST, int end_frame=Utility::LAST){
	VideoCapture cap(path);
	if( !cap.isOpened() )
		throw "Can't open video! Too bad...";
	// hyperlaps only from frame 2100 to 15000, because compare with others
	//	int num_frames = (15000-2100)/12; // thats 12900, with ~2.8Mb --> 36gb RAM (its 32)
	int num_frames = end_frame - start_frame;
	vid.clear();
	vid.reserve(num_frames);
	cap.set(CV_CAP_PROP_POS_FRAMES, start_frame);
	for (int i=0; i<num_frames; i++) {
		Mat tmp;
		cap >> tmp;
		vid.push_back(tmp);
	}
	return cap;
}


void surf_detection(Mat i1, Mat i2){

	// init stuff
	Mat i1_grey, i2_grey;
	cv::cvtColor(i1, i1_grey, CV_BGR2GRAY);
	cv::cvtColor(i2, i2_grey, CV_BGR2GRAY);

	GpuMat img1, img2;
	img1.upload(i1_grey);
	img2.upload(i2_grey);

	// detect features
	SURF_CUDA surf(10000);
	GpuMat keypoints1GPU, keypoints2GPU;
	GpuMat descriptors1GPU, descriptors2GPU;
	surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
	surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);
	//	cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
	//	cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

	//	// match features
	Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
	vector<DMatch> matches;
	matcher->match(descriptors1GPU, descriptors2GPU, matches);

	// get em on cpu
	vector<KeyPoint> keypoints1, keypoints2;
	vector<float> descriptors1, descriptors2;
	surf.downloadKeypoints(keypoints1GPU, keypoints1);
	surf.downloadKeypoints(keypoints2GPU, keypoints2);
	surf.downloadDescriptors(descriptors1GPU, descriptors1);
	surf.downloadDescriptors(descriptors2GPU, descriptors2);



	vector<Point2f> p1, p2;
	vector<DMatch> ms;

	for(unsigned int i=0; i<matches.size(); i++){
		DMatch m(matches[i].queryIdx, matches[i].trainIdx, matches[i].distance);
		ms.push_back(m);

		//		cout << matches[i].queryIdx << "; " << matches[i].trainIdx << endl;
		//		cout << keypoints1[matches[i].queryIdx].pt << endl;
		//		cout << keypoints2[matches[i].trainIdx].pt << endl;
		//		cout << matches[i].distance << endl;


		p1.push_back(keypoints1[matches[i].queryIdx].pt);
		p2.push_back(keypoints2[matches[i].trainIdx].pt);
	}

	Mat fund_mat = findFundamentalMat(p1, p2, FM_RANSAC, 3, 0.99);
	cout << fund_mat << endl;


	Mat img_matches;
	drawMatches(Mat(img1), keypoints1, Mat(img2), keypoints2, matches, img_matches);
	namedWindow("matches", 0);
	imshow("matches", img_matches);
	waitKey(0);

}


double optical_flow_farneback(Mat& i1, Mat& i2){
	Mat i1_grey, i2_grey;
	cv::cvtColor(i1, i1_grey, CV_BGR2GRAY);
	cv::cvtColor(i2, i2_grey, CV_BGR2GRAY);

	GpuMat d_frameL(i1_grey), d_frameR(i2_grey);
	GpuMat d_flow;
	Ptr<cuda::FarnebackOpticalFlow> d_calc = cuda::FarnebackOpticalFlow::create();
	Mat flowxy, flowx, flowy;

	if (GPU) {
		d_calc->calc(d_frameL, d_frameR, d_flow);

		GpuMat planes[2];
		cuda::split(d_flow, planes);

		planes[0].download(flowx);
		planes[1].download(flowy);
	}
	else {
		calcOpticalFlowFarneback(
				i1_grey, i2_grey, flowxy, d_calc->getPyrScale(), d_calc->getNumLevels(), d_calc->getWinSize(),
				d_calc->getNumIters(), d_calc->getPolyN(), d_calc->getPolySigma(), d_calc->getFlags());

		Mat planes[] = {flowx, flowy};
		split(flowxy, planes);
		flowx = planes[0]; flowy = planes[1];
	}

	double diff = cv::sum(cv::abs(flowx))[0];
	diff += cv::sum(cv::abs(flowy))[0];
	//	cout << diff << endl;
	//	Utility::colorize_flow(flowx, flowy);
	return diff;
}

double optical_flow_pyr(Mat& i1, Mat& i2){
	//shit
	Mat i1_grey, i2_grey;
	vector<Point2f> prevPts, nextPts, diffPts;
	cv::cvtColor(i1, i1_grey, CV_BGR2GRAY);
	cv::cvtColor(i2, i2_grey, CV_BGR2GRAY);
	goodFeaturesToTrack(i1_grey, prevPts, 100, 0.01, 2.0);

	Size winSize(21, 21);
	vector<Mat> prevPyr, nextPyr;
	buildOpticalFlowPyramid(i1_grey, prevPyr, winSize, 3, true, 0, 0, true);
	buildOpticalFlowPyramid(i2_grey, nextPyr, winSize, 3, true, 0, 0, true);

	vector<Mat> p1, p2;
	for (unsigned int i=0; i<prevPyr.size(); ++i) {
		p1.push_back(prevPyr[i]);
		p2.push_back(nextPyr[i]);
	}
	Mat status, err; // 0 = found
	calcOpticalFlowPyrLK(p1, p2, prevPts, nextPts, status, err, winSize, 3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 2);

	for (unsigned int i=0; i<prevPyr.size(); ++i) {
		diffPts.push_back(nextPts[i] - prevPts[i]);


	}
	//	cout << diffPts.size() << " points" << endl;
	//	cout << status <<  endl;
	double d = cv::norm(diffPts) * 1/(double)diffPts.size();
	cout << d << endl;
	//	imshow("flow", i1_grey);
	//	waitKey(0);
	//	imshow("flow", i2_grey);
	//	waitKey(0);
	int ct=0;
	for(int i = 0;i < status.size().height;i++){
		if ((int)status.at<unsigned char>(1,i) == 0)
			ct++;
	}

	//	cout << cv::norm(err) << endl;
	//	cout << err << endl;
//	cout << "######" << endl;

	return d;

}

double optical_flow_brox(Mat& i1, Mat& i2){
	Mat i1_grey, i2_grey;
	cv::cvtColor(i1, i1_grey, CV_BGR2GRAY);
	cv::cvtColor(i2, i2_grey, CV_BGR2GRAY);

	i1_grey.convertTo(i1_grey, CV_32FC1, 1.0/255.0);
	i2_grey.convertTo(i2_grey, CV_32FC1, 1.0/255.0);
	// CV_32FC1


	Ptr<cuda::BroxOpticalFlow> brx = cuda::BroxOpticalFlow::create();
	GpuMat mat_1(i1_grey), mat_2(i2_grey);
	GpuMat gpu_flow;
	Mat flowxy, flowx, flowy;

	if(GPU){
		brx->calc(mat_1, mat_2, gpu_flow);
		GpuMat planes[2];
		cuda::split(gpu_flow, planes);
		planes[0].download(flowx);
		planes[1].download(flowy);
	}


	double diff = cv::sum(cv::abs(flowx))[0];
	diff += cv::sum(cv::abs(flowy))[0];
//	cout << diff << endl;
//	Utility::colorize_flow(flowx, flowy, to_string(diff) + ".png");
//	imwrite("frame0.png", i1);
//	imwrite("org_" + to_string(diff) + ".png", i2);
	return diff;
}

double optical_flow_tvl1(Mat& i1, Mat& i2){
	// slow
	Mat i1_grey, i2_grey;
	cv::cvtColor(i1, i1_grey, CV_BGR2GRAY);
	cv::cvtColor(i2, i2_grey, CV_BGR2GRAY);

	GpuMat d_frameL(i1_grey), d_frameR(i2_grey);
	GpuMat d_flow;
	Ptr<cuda::OpticalFlowDual_TVL1> d_calc = cuda::OpticalFlowDual_TVL1::create();
	Mat flowxy, flowx, flowy;

	if (GPU) {
		d_calc->calc(d_frameL, d_frameR, d_flow);

		GpuMat planes[2];
		cuda::split(d_flow, planes);

		planes[0].download(flowx);
		planes[1].download(flowy);
	}

	double diff = cv::sum(cv::abs(flowx))[0];
	diff += cv::sum(cv::abs(flowy))[0];
//	cout << diff << endl;
	//	Utility::colorize_flow(flowx, flowy);
	return diff;
}

float earthmover_distance_grey(Mat& i1, Mat& i2){
	Mat i1_grey, i2_grey;
	cv::cvtColor(i1, i1_grey, CV_BGR2GRAY);
	cv::cvtColor(i2, i2_grey, CV_BGR2GRAY);
	int histSize = 256;

	Mat hist_1, hist_2;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	cv::calcHist(&i1_grey, 1, 0, Mat(), hist_1, 1, &histSize, &histRange, true, false);
	cv::calcHist(&i2_grey, 1, 0, Mat(), hist_2, 1, &histSize, &histRange, true, false);
	float diff = EMDL1(hist_1, hist_2);
	//	cout << diff << endl;
	return diff;
}

float earthmover_distance(Mat& i1, Mat& i2){
	int histSize = 256;
	Mat bgr_1[3];
	Mat bgr_2[3];
	split(i1, bgr_1);
	split(i2, bgr_2);

	Mat hist_1[3], hist_2[3];
	float diff[3];
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	for(int i=0; i<3; i++){
		cv::calcHist(&bgr_1[i], 1, 0, Mat(), hist_1[i], 1, &histSize, &histRange, true, false);
		cv::calcHist(&bgr_2[i], 1, 0, Mat(), hist_2[i], 1, &histSize, &histRange, true, false);
		diff[i] = EMDL1(hist_1[i], hist_2[i]);
		//		cout << diff[i]<< "; ";
	}

	float diff_sum = 0.0f;
	for(int i=0; i<3; i++){
		diff_sum += diff[i];
	}
	return diff_sum;
}

double block_matching(Mat& i1, Mat& i2){
	Mat i1_grey, i2_grey;
	cv::cvtColor(i1, i1_grey, CV_BGR2GRAY);
	cv::cvtColor(i2, i2_grey, CV_BGR2GRAY);
	int num_disps = 64;
	int block_size = 21;
	double diff;

	Mat disp;
	GpuMat d_frameL(i1_grey), d_frameR(i2_grey), d_disp;

	Ptr<cuda::StereoBM> bm;
	Ptr<cuda::StereoBeliefPropagation> bp;
	Ptr<cuda::StereoConstantSpaceBP> csbp;

	if (GPU) {
		bm = cuda::createStereoBM(64, block_size); //
		bp = cuda::createStereoBeliefPropagation(num_disps);
		csbp = cv::cuda::createStereoConstantSpaceBP(num_disps);

		bm->compute(d_frameL, d_frameR, d_disp);
		diff = cuda::absSum(d_disp)[0];
	}
	else {
		cout << "missing method" << endl;
	}
	d_disp.download(disp);
	//	imshow("disparity", disp);
	//	waitKey(0);

	return diff;
}



void test_farneback(string path){
	vector<Mat> output;

	int idx = 2100;
	int least_offset = 5;
	int max_offset = 25;
	int best_idx = -1;

	double min_diff = std::numeric_limits<double>::max();
	vector<Mat> vid;
	VideoCapture capture;

	while(true){
		capture = load_video(path, vid, idx, idx+max_offset+1);
		for(int i=least_offset; i<=max_offset; i++){
			double diff = optical_flow_farneback(vid[0], vid[i]);
			//			cout << min_diff<< "; " << diff << "; " << i << endl;

			if(diff<min_diff){
				min_diff = diff;
				best_idx = i;
			}
		}
		idx += best_idx;
		output.push_back(vid[best_idx]);
		min_diff = std::numeric_limits<double>::max();
		if(idx > 5000)
			break;
		cout << idx << endl;
	}
	Utility::save_video(output, 24.0f, "out_5000.avi");
	Utility::play_video(output, 24.0f);

}

void naiv_hyper(string path){
	vector<Mat> vid;
	VideoCapture cap("out_5000.avi");
	int frms = cap.get(CV_CAP_PROP_FRAME_COUNT);
	int skip = 2900/frms;


	VideoCapture ca(path);
	for (int i=0; i<frms; i++) {
		ca.set(CV_CAP_PROP_POS_FRAMES, 2100+i*skip);
		Mat tmp;
		ca >> tmp;
		vid.push_back(tmp);
	}

	Utility::save_video(vid, 24.0f, "naiv_5000.avi");

}

void other_tests(string path){
	int idx = 2100;
	int end = 3000;
	int least_offset = 5;
	int max_offset = 25;

	int best_idx[6] = {-1};
	double min_diff[6] = {std::numeric_limits<double>::max()};
	double diff[6];
	vector<Mat> vid;
	VideoCapture capture;

	while(true){
		capture = load_video(path, vid, idx, idx+max_offset+1);
		for(int i=least_offset; i<=max_offset; i++){

			diff[0] = optical_flow_farneback(vid[0], vid[i]);
//			cout << " 1 " << endl;
			diff[1] = optical_flow_farneback(vid[0], vid[i]);
//			cout << " 2 " << endl;
			diff[2] = optical_flow_brox(vid[0], vid[i]);
//			cout << " 3 " << endl;
//			diff[3] = optical_flow_tvl1(vid[0], vid[i]);
			diff[3] = diff[2];

//			cout << " 4 " << endl;
			diff[4] = earthmover_distance(vid[0], vid[i]);
//			cout << " 5 " << endl;
			diff[5] = block_matching(vid[0], vid[i]);
//			cout << " 6 " << endl;

			for(int j=0; j<6; j++){
				if(diff[j]<min_diff[j]){
					min_diff[j] = diff[j];
					best_idx[j] = i;
				}

			}
		}
		idx += 10;
		//		output.push_back(vid[best_idx]);
		if(idx > end)
			break;
		cout << "Index: " << idx << endl;
		for(int j=0; j<6; j++){
			min_diff[j] = std::numeric_limits<double>::max();
			cout << j << ": " << best_idx[j] << endl;
		}
		cout << endl;

	}

}

// TODO: Im Paper EgoSampling machen die nen Graphen dessen Gweicht aus 3 Komponenten besteht:
// TODO: 1. FOE(opipole)	2. Velocity Cost (optflow)	3. EMD
// TODO: FOE fehlt bis jetzt komplett, da muss ich mich vielleicht noch ins zeug legen
// TODO:

int main(int argc, char** argv) {

	vector<Mat> vid;
	VideoCapture capture = load_video(argv[1], vid);
	int num_pix = capture.get(3) * capture.get(4); //width and height
	float norm_pix = 1/(float)num_pix;

	// use fundamental matrix to determine the distance
	// alternativly use FOE/optical flow
	//	surf_detection(vid[0], vid[0]);

	//todo: use optical flow

	//	test_farneback(argv[1]);
	//	optical_flow_pyr(vid[0], vid[1]);
	//	optical_flow_pyr(vid[0], vid[500]);
	//	naiv_hyper(argv[1]);

//	cout << block_matching(vid[0], vid[0]) * norm_pix<< endl;
//	cout << block_matching(vid[0], vid[1])* norm_pix << endl;
//	cout << block_matching(vid[0], vid[2]) * norm_pix<< endl;
//	cout << block_matching(vid[0], vid[500]) * norm_pix<< endl;

	other_tests(argv[1]);

}
#endif // GPUMODE
