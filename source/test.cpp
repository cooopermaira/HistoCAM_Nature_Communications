#include "common.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#ifdef HAVE_OPENCV_XFEATURES2D

int main( int argc, char* argv[] )
{
  
  int factor = 16;

  string file1 = "/Users/bsumma/Source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1196.Raw";
  string file2 = "/Users/bsumma/Source/tulane/pathcam/opencv-testing/feature_extraction_test/images/temp-07282022114108-1197.Raw";
  
  pathCam::Image *image_1 = new pathCam::Image();
  image_1->set_disk_file(file1);
  
  image_1->load_raw_from_disk();
  image_1->create_reg_image(factor);
  
  pathCam::Image *image_2 = new pathCam::Image();
  image_2->set_disk_file(file2);
  
  image_2->load_raw_from_disk();
  image_2->create_reg_image(factor);
  
  
  auto begin = std::chrono::high_resolution_clock::now();
  
  pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(pathCam::_SURF);
  detector->set_SURF_params(5000);
  detector->detect_and_compute(image_1, image_2);
  
//    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
//    int minHessian = 5000;
//    Ptr<SURF> detector = SURF::create( minHessian );
//    //Mat descriptors_object, descriptors_scene;
//  detector->detectAndCompute( image_1->get_reg_image(), noArray(), image_1->keypoints, image_1->descriptors );
//    std::cout << image_1->keypoints.size() << "\n";
//  detector->detectAndCompute( image_2->get_reg_image(), noArray(), image_2->keypoints, image_2->descriptors );
//    std::cout << image_2->keypoints.size() << "\n";
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( image_1->descriptors, image_2->descriptors, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Localize the object
    std::vector<Point2f> image_1_pts;
    std::vector<Point2f> image_2_pts;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
      image_1_pts.push_back( image_1->keypoints[ good_matches[i].queryIdx ].pt );
      image_2_pts.push_back( image_2->keypoints[ good_matches[i].trainIdx ].pt );
    }
  
  //0, RANSAC, LMEDS, RHO
  Mat H = findHomography( image_1_pts, image_2_pts, RANSAC );
  double t_x = H.at<double>(0,0)*H.at<double>(0,2)*factor;
  double t_y = H.at<double>(1,1)*H.at<double>(1,2)*factor;
  std::cout << t_x << "\t" << t_y << "\n";

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  
  printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);

  
  //-- Draw matches
  Mat img_matches;
  drawMatches( image_1->get_reg_image(), image_1->keypoints, image_2->get_reg_image(), image_2->keypoints, good_matches, img_matches, Scalar::all(-1),
               Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)image_1->get_reg_image().cols, 0 );
    obj_corners[2] = Point2f( (float)image_1->get_reg_image().cols, (float)image_1->get_reg_image().rows );
    obj_corners[3] = Point2f( 0, (float)image_1->get_reg_image().rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f((float)image_1->get_reg_image().cols, 0),
          scene_corners[1] + Point2f((float)image_1->get_reg_image().cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f((float)image_1->get_reg_image().cols, 0),
          scene_corners[2] + Point2f((float)image_1->get_reg_image().cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f((float)image_1->get_reg_image().cols, 0),
          scene_corners[3] + Point2f((float)image_1->get_reg_image().cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f((float)image_1->get_reg_image().cols, 0),
          scene_corners[0] + Point2f((float)image_1->get_reg_image().cols, 0), Scalar( 0, 255, 0), 4 );
    //-- Show detected matches
    imshow("Good Matches & Object detection", img_matches );
    waitKey();
  
  delete image_1;
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
