#include <jni.h>
#include <pthread.h>
#include <semaphore.h>
#include <queue>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
int bufferSize = 2;

int patternSet = 0, threaded = 0, count1 = 0, count2 = 0;
//semaphores & mutex for synchronisation between two threads
sem_t full; //0
sem_t empty; //buffersize
pthread_mutex_t critic; //1
sem_t new_full;
sem_t new_empty;
pthread_mutex_t new_critic;
sem_t out_full;
sem_t out_empty;
pthread_mutex_t out_critic;
//thread declarations
pthread_t thread_level1, thread_level2, thread_level3, thread_level4;
//OpenCv variables that are used multiple times
Mat patternGray;
Size size_pattern;
vector<Point2f> initialPoints;
vector<KeyPoint> features;

queue<Mat> new_img_buffer, inter_img_buffer, inter_hmg_buffer,
		inter_warped_buffer, out_img_buffer;
queue<vector<KeyPoint> > inter_key_buffer;
Ptr<FeatureDetector> l1_detector = Ptr<FeatureDetector>(
		new OrbFeatureDetector(500, 1.5, 8, 31, 0, 2, ORB::FAST_SCORE, 31));
Ptr<FeatureDetector> l2_detector = Ptr<FeatureDetector>(
		new OrbFeatureDetector(400, 1.5, 8, 31, 0, 2, ORB::FAST_SCORE, 31));
Ptr<DescriptorExtractor> l1_extractor = Ptr<DescriptorExtractor>(
		new OrbDescriptorExtractor());
Ptr<DescriptorExtractor> l2_extractor = Ptr<DescriptorExtractor>(
		new OrbDescriptorExtractor());
Ptr<DescriptorMatcher> l1_matcher = Ptr<DescriptorMatcher>(
		new FlannBasedMatcher(new flann::LshIndexParams(6, 12, 2)));
Ptr<DescriptorMatcher> l2_matcher = Ptr<DescriptorMatcher>(
		new BFMatcher(NORM_HAMMING, false));

void draw2dContour(cv::Mat& image, cv::Scalar color, vector<Point2f> points2d) {
	for (size_t i = 0; i < points2d.size(); i++) {
		line(image, (Point2f) points2d[i],
				(Point2f) points2d[(i + 1) % points2d.size()], color, 4,
				CV_AA);
	}
}

void doRatioTest(vector<vector<DMatch> > knnMatches, vector<DMatch>& out,
		int minRatio = 2) {
	for (size_t i = 0; i < knnMatches.size(); i++) {
		const cv::DMatch& bestMatch = knnMatches[i][0];
		const cv::DMatch& betterMatch = knnMatches[i][1];
		float distanceRatio = bestMatch.distance / betterMatch.distance;
		if (distanceRatio < minRatio) {
			out.push_back(bestMatch);
		}
	}
}
bool refineMatchesWithHomography(const vector<KeyPoint> queryKeypoints,
		const vector<KeyPoint> trainKeypoints, float reprojectionThreshold,
		vector<DMatch> matched, Mat& homography) {
	const int minNumberMatchesAllowed = 8;
	if (matched.size() < minNumberMatchesAllowed)
		return false;
// Prepare data for cv::findHomography
	vector<Point2f> srcPoints(matched.size());
	vector<Point2f> dstPoints(matched.size());
	for (size_t i = 0; i < matched.size(); i++) {
		DMatch temp_match = matched[i];
		srcPoints[i] = ((KeyPoint) trainKeypoints[temp_match.trainIdx]).pt;
		dstPoints[i] = ((KeyPoint) queryKeypoints[temp_match.queryIdx]).pt;
	}
// Find homography matrix and get inliers mask
	vector<unsigned char> inliersMask(srcPoints.size());
	homography = findHomography(srcPoints, dstPoints,
	CV_FM_RANSAC, reprojectionThreshold, inliersMask);
	vector<DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++) {
		if (inliersMask[i])
			inliers.push_back((DMatch) matched[i]);
	}
	matched.swap(inliers);
	return matched.size() > minNumberMatchesAllowed;
}
void l1MatchProcedure() {
	//this_thread::__sleep_for(chrono::seconds(5), chrono::nanoseconds(0)); //fake produce time
	if(!patternSet)
		return;
	vector<KeyPoint> imn_keypoints;
	Mat imn_descriptors, rough_homography, warped, imageColor;
	vector<DMatch> matches;
	Mat imageGray;
	sem_wait(&new_full);
	pthread_mutex_lock(&new_critic);
	imageColor = new_img_buffer.front();
	new_img_buffer.pop();
	pthread_mutex_unlock(&new_critic);
	sem_post(&new_empty);
	cvtColor(imageColor, imageGray, CV_RGBA2GRAY);
	l1_detector->detect(imageGray, imn_keypoints);
	l1_extractor->compute(imageGray, imn_keypoints, imn_descriptors);
	l1_matcher->match(imn_descriptors, matches);
	bool homographyfound = refineMatchesWithHomography(imn_keypoints, features,
			2, matches, rough_homography);
	if (homographyfound) {
		warpPerspective(imageGray, warped, rough_homography, size_pattern,
				cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);
		sem_wait(&empty);
		pthread_mutex_lock(&critic);
		inter_hmg_buffer.push(rough_homography);
		inter_img_buffer.push(imageColor.clone());
		inter_warped_buffer.push(warped);
		pthread_mutex_unlock(&critic);
		sem_post(&full);
	}
}
void *l1MatchProcedurethreaded(void *void_ptr) {
	while (1) {
		l1MatchProcedure();
	}
}
void l2MatchProcedure() {
	if (!patternSet)
		return;
	sem_wait(&full);
	pthread_mutex_lock(&critic);
	Mat inter_hmg = inter_hmg_buffer.front();
	inter_hmg_buffer.pop();
	Mat image = inter_img_buffer.front();
	inter_img_buffer.pop();
	Mat warped = inter_warped_buffer.front();
	inter_warped_buffer.pop();
	pthread_mutex_unlock(&critic);
	sem_post(&empty);
	Mat imageGray, desc, hmgr, final_hmg;
	//cvtColor(image,imageGray,CV_RGBA2GRAY);
	vector<KeyPoint> warped_keypoints;
	vector<DMatch> matched;
	l2_detector->detect(warped, warped_keypoints);
	l2_extractor->compute(warped, warped_keypoints, desc);
	l2_matcher->match(desc, matched);
	// Estimate new refinement homography
	bool homographyFound = refineMatchesWithHomography(warped_keypoints,
			features, 2, matched, hmgr);
	if (homographyFound) {
		final_hmg = inter_hmg * hmgr;
		vector<Point2f> draw;
		perspectiveTransform(initialPoints, draw, final_hmg);
		draw2dContour(image, Scalar(255, 0, 0), draw);
		sem_wait(&out_empty);
		pthread_mutex_lock(&out_critic);
		out_img_buffer.push(image);
		pthread_mutex_unlock(&out_critic);
		sem_post(&out_full);
	}
}
void *l2MatchProcedurethreaded(void *void_ptr) {
	while (1) {
		l2MatchProcedure();
	}
}
int threadStarted = 0;
extern "C" {
JNIEXPORT void JNICALL Java_com_example_nativethreadtest_MainActivity_initializeSystem(
		JNIEnv *env, jobject obj) {
	sem_init(&full, 1, 0); //0
	sem_init(&empty, 1, bufferSize); //buffersize
	sem_init(&new_full, 1, 0);
	sem_init(&new_empty, 1, bufferSize);
	sem_init(&out_full, 1, 0);
	sem_init(&out_empty, 1, bufferSize);
}
JNIEXPORT void JNICALL Java_com_example_nativethreadtest_MainActivity_updateImage(
		JNIEnv *env, jobject obj, jlong addrIn) {
	Mat& mRgb = *(Mat*) addrIn;
	if (patternSet == 0)
		return;
	//add new image into buffer
	sem_wait(&new_empty);
	pthread_mutex_lock(&new_critic);
	new_img_buffer.push(mRgb.clone());
	pthread_mutex_unlock(&new_critic);
	sem_post(&new_full);
}

JNIEXPORT void JNICALL Java_com_example_nativethreadtest_MainActivity_getOutPutImage(
		JNIEnv *env, jobject obj, jlong addrIn) {
	Mat& mRgb = *(Mat*) addrIn;
	int ret = 0;
	pthread_mutex_lock(&out_critic);
	if (out_img_buffer.empty()) {
		ret = 1;
	}
	pthread_mutex_unlock(&out_critic);
	if (patternSet == 0 || ret == 1)
		return;
	//take from out queue if not empty yet
	sem_wait(&out_full);
	pthread_mutex_lock(&out_critic);
	mRgb = out_img_buffer.front();
	out_img_buffer.pop();
	pthread_mutex_unlock(&out_critic);
	sem_post(&out_empty);
}
JNIEXPORT void JNICALL Java_com_example_nativethreadtest_MainActivity_capturePattern(
		JNIEnv *env, jobject obj, jlong addrIn) {
	Mat& mRgb = *(Mat*) addrIn;
	cvtColor(mRgb, mRgb, CV_RGBA2GRAY);
	resize(mRgb, mRgb, Size(mRgb.size().width / 2, mRgb.size().height / 2));
	size_pattern = mRgb.size();
	Mat desc;
	Ptr<FeatureDetector> detector = Ptr<FeatureDetector>(
			new OrbFeatureDetector(1000));
	Ptr<DescriptorExtractor> extractor = Ptr<DescriptorExtractor>(
			new OrbDescriptorExtractor());
	detector->detect(mRgb, features);
	extractor->compute(mRgb, features, desc);
	l1_matcher->clear();
	l2_matcher->clear();
	std::vector<cv::Mat> descriptors(1);
	descriptors[0] = desc.clone();
	l1_matcher->add(descriptors);
	l2_matcher->add(descriptors);
	l1_matcher->train();
	l2_matcher->train();
	initialPoints.push_back(Point2f(0, 0));
	initialPoints.push_back(Point2f(mRgb.cols, 0));
	initialPoints.push_back(Point2f(mRgb.cols, mRgb.rows));
	initialPoints.push_back(Point2f(0, mRgb.rows));
	patternSet = 1;
}
JNIEXPORT void JNICALL Java_com_example_nativethreadtest_MainActivity_singleThread(
		JNIEnv *env, jobject obj) {
	threaded = 0;
	threadStarted = 0;
	l1MatchProcedure();
	l2MatchProcedure();
}
JNIEXPORT void JNICALL Java_com_example_nativethreadtest_MainActivity_MultiThread(
		JNIEnv *env, jobject obj) {
	if (threadStarted == 0) {
		threaded = 1;
		threadStarted = 1;
		pthread_create(&thread_level1, NULL, l1MatchProcedurethreaded, NULL);
		pthread_create(&thread_level1, NULL, l2MatchProcedurethreaded, NULL);
	}
}
JNIEXPORT jint JNICALL Java_com_example_nativethreadtest_MainActivity_check1(
		JNIEnv *env, jobject obj) {
	return count1;
}
JNIEXPORT jint JNICALL Java_com_example_nativethreadtest_MainActivity_check2(
		JNIEnv *env, jobject obj) {
	return count2;
}

}
