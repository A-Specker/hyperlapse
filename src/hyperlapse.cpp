#ifndef GPU
#define GPU true

#include "hyperlapse.hpp"
#include "graph.hpp"
#include "stabilize.hpp"

#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace std;
using namespace cv::cuda;


VideoCapture load_cap(String path){
	VideoCapture cap(path);
	if( !cap.isOpened() )
		throw "Can't open video! Too bad...";
	// hyperlaps only from frame 2100 to 15000, because compare with others
	//	int num_frames = (15000-2100)/12; // thats 12900, with ~2.8Mb --> 36gb RAM (its 32)
	cout << "Cap loaded." << endl;
	return cap;
}

void load_frame(Hyperlapse h, Mat& frame, int pos){
	h.cap.set(CV_CAP_PROP_POS_FRAMES, pos);
	h.cap >> frame;
}

void load_frames(Hyperlapse h, std::vector<Mat>& vid, int start_frame=Utility::FRST, int end_frame=Utility::LAST){
	int num_frames = end_frame - start_frame;
	vid.clear();
	vid.reserve(num_frames);
	h.cap.set(CV_CAP_PROP_POS_FRAMES, start_frame);
	for (int i=0; i<num_frames; i++) {
		Mat tmp;
		h.cap >> tmp;
		vid.push_back(tmp);
	}
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
	GpuMat descr_1_GPU, descr_2_GPU;
	surf(img1, GpuMat(), keypoints1GPU, descr_1_GPU);
	surf(img2, GpuMat(), keypoints2GPU, descr_2_GPU);
	//	cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
	//	cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

	//	// match features
	Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
	vector<DMatch> matches;
	matcher->match(descr_1_GPU, descr_2_GPU, matches);

	// get em on cpu
	vector<KeyPoint> keypoints1, keypoints2;
	vector<float> descr_1, descr_2;
	surf.downloadKeypoints(keypoints1GPU, keypoints1);
	surf.downloadKeypoints(keypoints2GPU, keypoints2);
	surf.downloadDescriptors(descr_1_GPU, descr_1);
	surf.downloadDescriptors(descr_2_GPU, descr_2);



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

void sift_detection(Mat& i1, Mat& i2){
	Mat i1_grey, i2_grey;
	cv::cvtColor(i1, i1_grey, CV_BGR2GRAY);
	cv::cvtColor(i2, i2_grey, CV_BGR2GRAY);


	Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descr_1, descr_2;

	detector->detect(i1_grey, keypoints_1);
	detector->detect(i2_grey, keypoints_2);

	detector->compute(i1_grey, keypoints_1, descr_1);
	detector->compute(i2_grey, keypoints_2, descr_2);


	// now, whats happening here?
	vector<DMatch> matches;
	vector<Point2f> p1, p2;
	vector<DMatch> ms;

	auto matcher = cv::DescriptorMatcher::create(2);
	matcher->match(descr_1, descr_2, matches);

	for(unsigned int i=0; i<matches.size(); i++){
		DMatch m(matches[i].queryIdx, matches[i].trainIdx, matches[i].distance);
		ms.push_back(m);

		//		cout << matches[i].queryIdx << "; " << matches[i].trainIdx << endl;
		//		cout << keypoints1[matches[i].queryIdx].pt << endl;
		//		cout << keypoints2[matches[i].trainIdx].pt << endl;
		//		cout << matches[i].distance << endl;


		p1.push_back(keypoints_1[matches[i].queryIdx].pt);
		p2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	cout << matches.size() << endl;

	// and the fundamental matrix F...
	cout << "Find fundamental Matrix F" << endl;

	Mat fund_mat = findFundamentalMat(p1, p2, FM_RANSAC, 3, 0.99);
	cout << fund_mat << endl;


	Mat img_matches;
	drawMatches(i1_grey, keypoints_1, i2_grey, keypoints_2, matches, img_matches);
	namedWindow("matches", 0);
	imshow("matches", img_matches);
	waitKey(0);

	return;
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
	// sparse but shit
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

double vpp_optflow(Mat& i1, Mat& i2, Hyperlapse h){



	vpp::box2d domain = vpp::make_box2d(h.cap.get(CV_CAP_PROP_FRAME_HEIGHT),
			h.cap.get(CV_CAP_PROP_FRAME_WIDTH));
	vpp::video_extruder_ctx ctxx(domain);

	vpp::image2d<vpp::vuchar3> fr1 = vpp::from_opencv<vpp::vuchar3>(i1);
	vpp::image2d<vpp::vuchar3> fr2 = vpp::from_opencv<vpp::vuchar3>(i2);
	auto f1 = vpp::clone(fr1, vpp::_border = 3);
	auto f2 = vpp::clone(fr2, vpp::_border = 3);
	fill_border_mirror(f1);
	fill_border_mirror(f2);
	auto f1_grey = vpp::rgb_to_graylevel<unsigned char>(f1);
	auto f2_grey = vpp::rgb_to_graylevel<unsigned char>(f2);
	vpp::image2d<unsigned char> prev_frame(domain);
	vpp::video_extruder_update(ctxx, f1_grey, f1_grey,
			vpp::_detector_th = 10,
			vpp::_keypoint_spacing = 5,
			vpp::_detector_period = 1,
			vpp::_max_trajectory_length = 15);

	vpp::video_extruder_update(ctxx, f1_grey, f2_grey,
			vpp::_detector_th = 10,
			vpp::_keypoint_spacing = 5,
			vpp::_detector_period = 1,
			vpp::_max_trajectory_length = 15);



//	cout << "Hist size: " <<  ctxx.trajectories[0].history_.size() <<endl;
//	cout << ctxx.trajectories[0].history_[0][0] << ", " << ctxx.trajectories[0].history_[0][1] <<  endl;
//	cout << ctxx.trajectories[0].history_[1][0] << ", " <<  ctxx.trajectories[0].history_[1][1] <<  endl;
//	cout << "Veclocity: " << ctxx.keypoints[0].velocity[0] <<  ", " <<  ctxx.keypoints[0].velocity[1] << endl;
	double sum = 0;
	for(int i=0; i< ctxx.keypoints.size(); i++){
		sum += hypot(ctxx.keypoints[i].velocity[0], ctxx.keypoints[i].velocity[1]);
//		cout << ctxx.trajectories[i].history_.size() << endl;
	}
//	cout << sum << ", " << sum/ctxx.keypoints.size() << endl;
	//	auto display = clone(f1);
	//	vpp::draw::draw_trajectories(display, ctxx.trajectories, 200);
	//	cv::imshow("Trajectories", to_opencv(display));
	//	cv::waitKey(0);
	return sum/ctxx.keypoints.size();


}




void get_path(Hyperlapse& h, HGraph& g, int idx = 2100){
	g.add_node(idx);
	unsigned int end = 2200;
	unsigned int least_offset = 6;
	unsigned int max_offset = 18;
	Mat fr;
	vector<Mat> video;
	for(unsigned int k=0; k<12900; k++){
		load_frame(h, fr, idx);
		load_frames(h, video, idx+least_offset, idx+max_offset);

		for(unsigned int i=0; i<video.size(); i++){
			if(g.edge_exists(idx, idx+least_offset+i)){
				h.skip++;
				continue;
			}
//			double c1 = optical_flow_farneback(fr, video[i]) * h.norm_pix();
			double c1 = vpp_optflow(fr, video[i], h);
			double c2 = block_matching(fr, video[i])* h.norm_pix();
			double c3 = earthmover_distance(fr, video[i])* h.norm_pix();

			//		cout << i << ": " << c1 << "; " << c2 << "; " << c3 << endl;

			double cost = c1*Utility::ALPHA() + c2*Utility::BETA() + c3* Utility::GAMMA();
			g.add_edge(idx, idx+least_offset+i, cost);
			h.done++;
			cout << idx << ": " << c1*Utility::ALPHA() << ", " << c2*Utility::BETA() << ", " << c3*Utility::GAMMA() << endl;

			//		cout << idx+least_offset+i << endl;
			//		g.print_nodes();
			if (idx+least_offset+i < end){
				//			another_test(h, g, idx+least_offset+i);
			}
		}
		idx++;
	}
	g.Dijkstra(h.path);

}

void test_dijkstra(Hyperlapse h, HGraph g){
	vector<int> shortest_path;
	g.add_node(1);
	g.add_node(2);
	g.add_node(3);
	g.add_node(4);
	g.add_node(5);

	g.add_edge(1, 2, 1.0);
	g.add_edge(1, 4, 2.0);
	g.add_edge(1, 3, 7.0);
	g.add_edge(2, 3, 3.0);
	g.add_edge(3, 4, 5.0);
	g.add_edge(3, 5, 1.0);
	g.add_edge(4, 5, 7.0);
	g.Dijkstra(shortest_path);
}

void test_farneback(Hyperlapse h){
	vector<Mat> output;

	int idx = 2100;
	int least_offset = 5;
	int max_offset = 25;
	int best_idx = -1;

	double min_diff = std::numeric_limits<double>::max();
	vector<Mat> vid;
	VideoCapture capture;

	while(true){
		load_frames(h, vid, idx, idx+max_offset+1);
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
//	VideoCapture cap("out_15000.avi");
//	int frms = cap.get(CV_CAP_PROP_FRAME_COUNT);
	int skip = 10;


	VideoCapture ca(path);
	for (int i=0; i<(15000-2100)/10; i++) {
		ca.set(CV_CAP_PROP_POS_FRAMES, 2100+i*skip);
		Mat tmp;
		ca >> tmp;
		vid.push_back(tmp);
	}

	Utility::save_video(vid, 24.0f, "naiv_15000.avi");

}

void other_tests(Hyperlapse h){
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
		load_frames(h, vid, idx, idx+max_offset+1);
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

// TODO: Im Paper EgoSampling machen die nen Graphen dessen Gewicht aus 3 Komponenten besteht:
// TODO: 1. FOE(opipole)	2. Velocity Cost (optflow)	3. EMD
// TODO: FOE fehlt bis jetzt komplett, da muss ich mich vielleicht noch ins zeug legen
// TODO: Ich will eigetlich auch einen Graphen und Daijstra, weil das besser is als iterativ

int main(int argc, char** argv) {
	cout << "Start." << endl;
	Hyperlapse h;
	HGraph g;
	h.cap = load_cap(argv[1]);



//	string vid = "/home/specker/Documents/hyperlapse/opencv/hyperlapse/Release/15000_vpp.avi";
//	Stabilize::stabilize(h.cap, vid);


//	get_path(h, g);
//	Utility::frms2vis(argv[1], h.path);



}

#endif // GPUMODE
