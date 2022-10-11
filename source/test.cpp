#include "common.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main( int argc, char* argv[] )
{
  
  std::cout << getBuildInformation() << "\n";
  
  //Crop
  double crop_factor = 0.5;
  //1.0, 0.5, 0.25
  
  double scale_factor = 0.25;
  //1.0, 0.5, 0.25, 0.125, 0.0625
  
  bool debayer = true;
  //true, false
  
  int interpolation = cv::INTER_NEAREST;
  //INTER_NEAREST, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4, INTER_LINEAR_EXACT,
  //INTER_LINEAR_EXACT, INTER_NEAREST_EXACT
  
  //Note that setting parameters will override this setting
  int features = pathCam::_SURF;
  //pathCam::_AKAZE, *pathCam::_BRISK, pathCam::_KAZE,
  //pathCam::_MSER, pathCam::_ORB, *pathCam::_SIFT, pathCam::_BOOST,
  //pathCam::_DAISY, pathCam::_LATCH, pathCam::_LUCID,
  //pathCam::_MSD, *pathCam::_SURF, pathCam::_VGG

  //Not working
  bool use_FREAK = true;
  
  //not really working: pathCam::_GFFT
  
  pathCam::FeatureDetector::SIFTParameters SIFT_params;
  pathCam::FeatureDetector::SURFParameters SURF_params = pathCam::FeatureDetector::SURFParameters(1000, 1, 3, false, false);
  pathCam::FeatureDetector::AKAZEParameters AKAZE_params;
  pathCam::FeatureDetector::BRISKParameters BRISK_params;

  
  cv::DescriptorMatcher::MatcherType matcher_type = cv::DescriptorMatcher::BRUTEFORCE_HAMMING;
  //cv::DescriptorMatcher::FLANNBASED
  //cv::DescriptorMatcher::BRUTEFORCE
  //cv::DescriptorMatcher::BRUTEFORCE_L1
  //cv::DescriptorMatcher::BRUTEFORCE_HAMMING
  //cv::DescriptorMatcher::BRUTEFORCE_HAMMINGLUT
  //cv::DescriptorMatcher::BRUTEFORCE_SL2
  
  int estimator = RANSAC;
  //LMEDS,RANSAC, RHO, USAC_DEFAULT, USAC_PARALLEL, USAC_FM_8PTS,
  //USAC_FAST, USAC_ACCURATE, USAC_PROSAC, USAC_MAGSAC
  
  string file1 = "/Users/bsumma/source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1196.Raw";
  string file2 = "/Users/bsumma/source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1198.Raw";
  float dx = -4.57613;
  float dy = 0.324399;
  
  pathCam::Image *image_1 = new pathCam::Image();
  image_1->set_disk_file(file1);
  
  image_1->load_raw_from_disk();
  
  pathCam::Image *image_2 = new pathCam::Image();
  image_2->set_disk_file(file2);
  
  image_2->load_raw_from_disk();
  
  if(!image_1->in_memory() || !image_2->in_memory()){ return -1; }
  
  auto begin = std::chrono::high_resolution_clock::now();

  image_1->create_reg_image(scale_factor,crop_factor,debayer,interpolation);
  image_2->create_reg_image(scale_factor,crop_factor,debayer,interpolation);

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
  }

  detector->detect_and_compute(image_1);
  detector->detect_and_compute(image_2);
  
  pathCam::Match *m = new pathCam::Match(image_1,image_2);
  pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(matcher_type);
  matcher->match(m);
  
    //-- Localize the object
    std::vector<Point2f> image_1_pts;
    std::vector<Point2f> image_2_pts;
    for( size_t i = 0; i < m->good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
      image_1_pts.push_back( image_1->keypoints[ m->good_matches[i].queryIdx ].pt );
      image_2_pts.push_back( image_2->keypoints[ m->good_matches[i].trainIdx ].pt );
    }
  
  Mat H = findHomography( image_1_pts, image_2_pts, estimator );
  double t_x = H.at<double>(0,0)*H.at<double>(0,2)*(1.0/scale_factor);
  double t_y = H.at<double>(1,1)*H.at<double>(1,2)*(1.0/scale_factor);

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  
  std::cout << scale_factor << "\t";
  std::cout << crop_factor << "\t";
  std::cout << debayer << "\t";
  std::cout << (image_1->keypoints.size()+image_2->keypoints.size())/2 << "\t";
//  std::cout << "(" << dx << "," << dy <<  ")" << "\t";
//  std::cout << "(" << t_x << "," << t_y <<  ")" << "\t";
  std::cout << sqrt((dx-t_x)*(dx-t_x) + (dy-t_y)*(dy-t_y)) << "\t";
  printf("%.3fs\n", elapsed.count() * 1e-9);

  
//  //-- Draw matches
//  Mat img_matches;
//  drawMatches( image_1->get_reg_image(), image_1->keypoints, image_2->get_reg_image(), image_2->keypoints, good_matches, img_matches, Scalar::all(-1),
//               Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//
//    //-- Get the corners from the image_1 ( the object to be "detected" )
//    std::vector<Point2f> obj_corners(4);
//    obj_corners[0] = Point2f(0, 0);
//    obj_corners[1] = Point2f( (float)image_1->get_reg_image().cols, 0 );
//    obj_corners[2] = Point2f( (float)image_1->get_reg_image().cols, (float)image_1->get_reg_image().rows );
//    obj_corners[3] = Point2f( 0, (float)image_1->get_reg_image().rows );
//    std::vector<Point2f> scene_corners(4);
//    perspectiveTransform( obj_corners, scene_corners, H);
//    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//    line( img_matches, scene_corners[0] + Point2f((float)image_1->get_reg_image().cols, 0),
//          scene_corners[1] + Point2f((float)image_1->get_reg_image().cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches, scene_corners[1] + Point2f((float)image_1->get_reg_image().cols, 0),
//          scene_corners[2] + Point2f((float)image_1->get_reg_image().cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[2] + Point2f((float)image_1->get_reg_image().cols, 0),
//          scene_corners[3] + Point2f((float)image_1->get_reg_image().cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[3] + Point2f((float)image_1->get_reg_image().cols, 0),
//          scene_corners[0] + Point2f((float)image_1->get_reg_image().cols, 0), Scalar( 0, 255, 0), 4 );
//    //-- Show detected matches
//    imshow("Good Matches & Object detection", img_matches );
//    waitKey();
  
  delete image_1;
  delete image_2;
  delete detector;
  delete m;
  
  return 0;
}
