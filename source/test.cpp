#include "pathCam.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


int main( int argc, char* argv[] )
{
  
  std::cout << getBuildInformation() << "\n";
  
  //Crop
  double crop_factor = 0.5;
  //1.0, 0.5, 0.25
  
  double scale_factor = 1.0;
  //1.0, 0.5, 0.25, 0.125, 0.0625
  
  bool debayer = true;
  //true, false
  
  bool real = false;
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
  pathCam::FeatureDetector::ORBParameters ORB_params = pathCam::FeatureDetector::ORBParameters(500, 1, 1, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
  
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
    string file2 = "/Users/bsumma/Library/CloudStorage/Box-Box/PathCam/test_data/raw/temp-07282022114117-1230.Raw";
    float dx = -4.57613;
    float dy = 0.324399;
  
//  string file1 = "/Users/bsumma/Library/CloudStorage/Box-Box/PathCam/test_data/raw/temp-07282022113848-356.Raw";
//  string file2 = "/Users/bsumma/Library/CloudStorage/Box-Box/PathCam/test_data/raw/temp-07282022113849-358.Raw";
//  float dx = -0.236034;
//  float dy = 0.581168;
  
  
  
  //  string file1 = "/Users/bsumma/source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1196.Raw";
  //    string file2 = "/Users/bsumma/source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1197.Raw";
  //    float dx = -3.92382;
  //    float dy = 0.041635;
  
  pathCam::Image *image_1 = new pathCam::Image();
  image_1->set_disk_file(file1);
  
  image_1->load_raw_from_disk();
  
  pathCam::Image *image_2 = new pathCam::Image();
  image_2->set_disk_file(file2);
  
  image_2->load_raw_from_disk();
  
  if(!image_1->in_memory() || !image_2->in_memory()){ return -1; }
  
  image_1->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);
  image_2->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);
  
  
  
  pathCam::MotionEstimator *mot = new pathCam::MotionEstimator();
  
  pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(features, use_FREAK);
  
  switch(features){
    case pathCam::_SURF:
      detector->set_SURF_params(SURF_params);
      break;
    case pathCam::_SIFT:
      detector->set_SIFT_params(SIFT_params);
      break;
    case pathCam::_AKAZE:
      detector->set_AKAZE_params(AKAZE_params);
      break;
    case pathCam::_BRISK:
      detector->set_BRISK_params(BRISK_params);
      break;
    case pathCam::_ORB:
      detector->set_ORB_params(ORB_params);
      break;
  }
  
  detector->detect_and_compute(image_1);
  detector->detect_and_compute(image_2);
  
  if(image_1->keypoints.size() < 200 || image_2->keypoints.size() < 200){
    std::cout << "Too little features detected.  Going back to defaults\n";
    delete detector;
    detector = new pathCam::FeatureDetector(features, use_FREAK);
    detector->detect_and_compute(image_1);
    detector->detect_and_compute(image_2);
  }
  
  
  
  pathCam::Match *m = new pathCam::Match(image_1,image_2);
  
  pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(matcher_type);
  matcher->match(m);
  
  
  
  mot->findHomography(m, estimator_type);
  
  
  //    std::cout << scale_factor << "\t";
  //    std::cout << crop_factor << "\t";
  //    std::cout << debayer << "\t";
  //std::cout << (image_1->keypoints.size()+image_2->keypoints.size())/2 << "\t";
  std::cout << m->t_x << "\t" << m->t_y << "\n";
  //std::cout << sqrt((dx-mot->t_x)*(dx-mot->t_x) + (dy-mot->t_y)*(dy-mot->t_y)) << "\t";
  //printf("%.3fs\n", elapsed.count() * 1e-9);
  
  //    Mat out_im;
  //    Mat match_im;
  //
  //    cv::drawKeypoints(image_1->get_reg_image(), image_1->keypoints, out_im, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  //    cv::drawMatches(image_1->get_reg_image(), image_1->keypoints, image_2->get_reg_image(), image_2->keypoints,
  //                    m->good_matches, match_im);
  //
  //    imwrite("keypoints.png", out_im);
  //    imwrite("matches.png", match_im);
  //    imshow("keypoints.png", out_im);
  //    imshow("matches.png", match_im);
  //
  //    waitKey(0);                                          // Wait for a keystroke in the window
  //    destroyAllWindows();
  //    return 0;
  
  
  
  image_1->load_raw_from_disk();
  image_2->load_raw_from_disk();
  
  
  Mat image_1_Mat = cv::Mat(Size(6464,4852), CV_8UC1, image_1->get_Raw(), Mat::AUTO_STEP);
  cvtColor(image_1_Mat,image_1_Mat,COLOR_BayerBG2BGR);
  
  Mat image_2_Mat = cv::Mat(Size(6464,4852), CV_8UC1, image_2->get_Raw(), Mat::AUTO_STEP);
  cvtColor(image_2_Mat,image_2_Mat,COLOR_BayerBG2BGR);
    
  double image_1_Box[4] = {0.0f, 0.0f, (float)image_1_Mat.cols, (float)image_1_Mat.rows};
  double image_2_Box[4] = {0.0+-m->t_x, 0.0+-m->t_y, (double)image_2_Mat.cols+-m->t_x, (double)image_2_Mat.rows+-m->t_y};
  
  std::cout << "image_1 bbox: ";
  std::cout << image_1_Box[0] << "\t" << image_1_Box[1] << "\t";
  std::cout << image_1_Box[2] << "\t" << image_1_Box[3] << "\n";

  
  std::cout << "image_2 bbox: ";
  std::cout << image_2_Box[0] << "\t" << image_2_Box[1] << "\t";
  std::cout << image_2_Box[2] << "\t" << image_2_Box[3] << "\n";

  
  double combined_RECT[4] = {min(image_1_Box[0] , image_2_Box[0]),
    min(image_1_Box[1] , image_2_Box[1]),
    max(image_1_Box[2] , image_2_Box[2]),
    max(image_1_Box[3] , image_2_Box[3])};
  
  std::cout << "combined bbox: ";
  std::cout << combined_RECT[0] << "\t" << combined_RECT[1] << "\t";
  std::cout << combined_RECT[2] << "\t" << combined_RECT[3] << "\n";

  
  if(combined_RECT[0] < 0.0){
    image_1_Box[0] -= combined_RECT[0];
    image_1_Box[2] -= combined_RECT[0];
    image_2_Box[0] -= combined_RECT[0];
    image_2_Box[2] -= combined_RECT[0];
  }
  
  if(combined_RECT[1] < 0.0){
    image_1_Box[1] -= combined_RECT[1];
    image_1_Box[3] -= combined_RECT[1];
    image_2_Box[1] -= combined_RECT[1];
    image_2_Box[3] -= combined_RECT[1];
   }
  
  combined_RECT[0] = min(image_1_Box[0] , image_2_Box[0]);
  combined_RECT[1] =  min(image_1_Box[1] , image_2_Box[1]);
  combined_RECT[2] =  max(image_1_Box[2] , image_2_Box[2]);
  combined_RECT[3] =  max(image_1_Box[3] , image_2_Box[3]);

  
  std::cout << "combined bbox: ";
  std::cout << combined_RECT[0] << "\t" << combined_RECT[1] << "\t";
  std::cout << combined_RECT[2] << "\t" << combined_RECT[3] << "\n";

  std::cout << "width height: ";
  std::cout << "[ " << combined_RECT[2]-combined_RECT[0] << ", ";
  std::cout << combined_RECT[3]-combined_RECT[1] << "]\n";
  
  std::cout << "image 1: ";
  std::cout << image_1_Box[0] << "\t" << image_1_Box[1] << "\t";
  std::cout << image_1_Mat.cols<< "\t" << image_1_Mat.rows << "\n";


  
  Mat3b combined(combined_RECT[3]-combined_RECT[1],combined_RECT[2]-combined_RECT[0], Vec3b(0,0,0));
  std::cout << combined.size()  << "\n";
  image_1_Mat.copyTo(combined(Rect(image_1_Box[0], image_1_Box[1], image_1_Mat.cols, image_1_Mat.rows)));
  image_2_Mat.copyTo(combined(Rect(image_2_Box[0], image_2_Box[1], image_2_Mat.cols, image_2_Mat.rows)));
  imwrite("registration.png", combined);
  
  
//
//  Mat ground_truth = image_2_Mat.clone();
//  warpAffine(ground_truth, ground_truth,  T_M, Size(6464,4852));
//  absdiff(ground_truth, image_1_Mat, ground_truth);
//
//  imwrite("ground_truth.png", ground_truth);


//  T_M.at<float>(0,2) = -m->t_x;
//  T_M.at<float>(1,2) = -m->t_y;


//  image_1_Mat.copyTo(combined(Rect(0, 0, image_1_Mat.cols, image_1_Mat.rows)));
//  //image_2_Mat.copyTo(combined(Rect(-dy, -dx, image_2_Mat.cols, image_2_Mat.rows)));
//
 
  
  delete image_1;
  delete image_2;
//  delete detector;
//  delete m;
  //delete mot;
  
  return 0;
}
