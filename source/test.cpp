#include "common.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main( int argc, char* argv[] )
{
  
  std::cout << getBuildInformation() << "\n";
  
  //Crop
  double crop_factor = 1.0;
  //1.0, 0.5, 0.25
  
  double scale_factor = 1.0;
  //1.0, 0.5, 0.25, 0.125, 0.0625
  
  bool debayer = true;
  //true, false
  
  bool real = true;
  //true, false  Phase coorelation needs real valued image
  
  int interpolation = cv::INTER_NEAREST;
  //cv::INTER_NEAREST, cv::INTER_CUBIC, cv::INTER_AREA, cv::INTER_LANCZOS4, cv::INTER_LINEAR_EXACT,
  //cv::INTER_LINEAR_EXACT, cv::INTER_NEAREST_EXACT
  
  //Note that setting parameters will override this setting
  int features = pathCam::_ORB;
  //pathCam::_AKAZE, *pathCam::_BRISK, pathCam::_KAZE,
  //pathCam::_MSER, *pathCam::_ORB, *pathCam::_SIFT, pathCam::_BOOST,
  //pathCam::_DAISY, pathCam::_LATCH, pathCam::_LUCID,
  //pathCam::_MSD, *pathCam::_SURF, pathCam::_VGG

  //Not working
  bool use_FREAK = false;
  
  //not really working: pathCam::_GFFT
  
  pathCam::FeatureDetector::SIFTParameters SIFT_params;
  pathCam::FeatureDetector::SURFParameters SURF_params = pathCam::FeatureDetector::SURFParameters(1000, 1, 1, false, false);
  pathCam::FeatureDetector::AKAZEParameters AKAZE_params;
  pathCam::FeatureDetector::BRISKParameters BRISK_params;
  pathCam::FeatureDetector::ORBParameters ORB_params = pathCam::FeatureDetector::ORBParameters(500, 1.0, 1, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
  
  cv::DescriptorMatcher::MatcherType matcher_type = cv::DescriptorMatcher::BRUTEFORCE_HAMMING;
  //cv::DescriptorMatcher::FLANNBASED
  //cv::DescriptorMatcher::BRUTEFORCE
  //cv::DescriptorMatcher::BRUTEFORCE_L1
  //cv::DescriptorMatcher::BRUTEFORCE_HAMMING
  //cv::DescriptorMatcher::BRUTEFORCE_HAMMINGLUT
  //cv::DescriptorMatcher::BRUTEFORCE_SL2
  
  int estimator_type = cv::RANSAC;
  //cv::LMEDS, cv::RANSAC, cv::RHO
  
  string file1 = "/Users/bsumma/source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1196.Raw";
  string file2 = "/Users/bsumma/source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1198.Raw";
  float dx = -4.57613;
  float dy = 0.324399;
  
//  string file1 = "/Users/bsumma/source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1196.Raw";
//  string file2 = "/Users/bsumma/source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1197.Raw";
//  float dx = -3.92382;
//  float dy = 0.041635;
  
  pathCam::Image *image_1 = new pathCam::Image();
  image_1->set_disk_file(file1);
  
  image_1->load_raw_from_disk();
  
  pathCam::Image *image_2 = new pathCam::Image();
  image_2->set_disk_file(file2);
  
  image_2->load_raw_from_disk();
  
  if(!image_1->in_memory() || !image_2->in_memory()){ return -1; }
  
  auto begin = std::chrono::high_resolution_clock::now();

  image_1->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);
  image_2->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);

  pathCam::MotionEstimator *mot = new pathCam::MotionEstimator();
  
//  pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(features, use_FREAK);
//
//  switch(features){
//    case pathCam::_SURF:
//      detector->set_SURF_params(SURF_params);
//      break;
//    case pathCam::_SIFT:
//      detector->set_SIFT_params(SIFT_params);
//      break;
//    case pathCam::_AKAZE:
//      detector->set_AKAZE_params(AKAZE_params);
//      break;
//    case pathCam::_BRISK:
//      detector->set_BRISK_params(BRISK_params);
//      break;
//    case pathCam::_ORB:
//      detector->set_ORB_params(ORB_params);
//      break;
//  }
//
//  detector->detect_and_compute(image_1);
//  detector->detect_and_compute(image_2);
//
//  pathCam::Match *m = new pathCam::Match(image_1,image_2);
//  pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(matcher_type);
//  matcher->match(m);
//
//  mot->findHomography(m, estimator_type);
//  
//  auto end = std::chrono::high_resolution_clock::now();
//  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
//  
//  std::cout << scale_factor << "\t";
//  std::cout << crop_factor << "\t";
//  std::cout << debayer << "\t";
//  std::cout << (image_1->keypoints.size()+image_2->keypoints.size())/2 << "\t";
//  std::cout << sqrt((dx-mot->t_x)*(dx-mot->t_x) + (dy-mot->t_y)*(dy-mot->t_y)) << "\t";
//  printf("%.3fs\n", elapsed.count() * 1e-9);
  
  mot->matchTemplate(image_1, image_2);
  

//  Mat T_M = Mat(2,3,CV_32F);
//
//  T_M.at<float>(0,0) = 1.0;
//  T_M.at<float>(0,1) = 0.0;
//  T_M.at<float>(0,2) = -dx;
//  T_M.at<float>(1,0) = 0.0;
//  T_M.at<float>(1,1) = 1.0;
//  T_M.at<float>(1,2) = -dy;
//
//  image_1->load_raw_from_disk();
//  image_2->load_raw_from_disk();
//
//
//  Mat image_1_Mat = cv::Mat(Size(6464,4852), CV_8UC1, image_1->get_Raw(), Mat::AUTO_STEP);
//  cvtColor(image_1_Mat,image_1_Mat,COLOR_BayerBG2BGR);
//
//  Mat image_2_Mat = cv::Mat(Size(6464,4852), CV_8UC1, image_2->get_Raw(), Mat::AUTO_STEP);
//  cvtColor(image_2_Mat,image_2_Mat,COLOR_BayerBG2BGR);
//
//  Mat ground_truth = image_2_Mat.clone();
//  warpAffine(ground_truth, ground_truth,  T_M, Size(6464,4852));
//  absdiff(ground_truth, image_1_Mat, ground_truth);
//
//  imwrite("ground_truth.png", ground_truth);
//
//
//  T_M.at<float>(0,2) = -mot->t_x;
//  T_M.at<float>(1,2) = -mot->t_y;
//
//  Mat test = image_2_Mat.clone();
//  warpAffine(test, test,  T_M, Size(6464,4852));
//  absdiff(test, image_1_Mat, test);
//
//  imwrite("test.png", ground_truth);
  
  delete image_1;
  delete image_2;
//  delete detector;
//  delete m;
  delete mot;
  
  return 0;
}
