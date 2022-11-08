//
//  MotionEstimator.cpp
//  pathCam
//
//  Created by Brian Summa on 11/1/22.
//

#include "common.h"


namespace pathCam{

void MotionEstimator::findHomography(pathCam::Match *m, int estimator_type,
                                     double ransacReprojThreshold,
                                     int maxIters, double confidence){
  
  //-- Localize the object
  std::vector<Point2f> image_1_pts;
  std::vector<Point2f> image_2_pts;
  for( size_t i = 0; i < m->good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    image_1_pts.push_back( m->image_1->keypoints[ m->good_matches[i].queryIdx ].pt );
    image_2_pts.push_back( m->image_2->keypoints[ m->good_matches[i].trainIdx ].pt );
  }
  
  Mat H = cv::findHomography(image_1_pts, image_2_pts, estimator_type,
                             ransacReprojThreshold, noArray(), maxIters,
                             confidence);
  t_x = H.at<double>(0,0)*H.at<double>(0,2)*(1.0/m->image_2->get_reg_scale());
  t_y = H.at<double>(1,1)*H.at<double>(1,2)*(1.0/m->image_2->get_reg_scale());
  
}

void MotionEstimator::phaseCorrelate(Image *image_1, Image *image_2){

  Point2d p = cv::phaseCorrelate(image_1->get_reg_image(), image_2->get_reg_image());
  t_x = p.x*(1.0/image_2->get_reg_scale());
  t_y = p.y*(1.0/image_2->get_reg_scale());
  
}

void MotionEstimator::matchTemplate(Image *image_1, Image *image_2){
  
  int match_method = cv::TM_CCORR ;
  
  int result_cols = image_1->get_reg_image().cols - image_2->get_reg_image().cols + 1;
  int result_rows = image_1->get_reg_image().rows - image_2->get_reg_image().rows + 1;

  
  Mat result;
  result.create( result_rows, result_cols, CV_32FC1 );
  
  cv::matchTemplate(image_1->get_reg_image(),image_2->get_reg_image(),result, match_method);
  
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;
  cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
  

  
  if( match_method  == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED )
      { matchLoc = minLoc; }
    else
      { matchLoc = maxLoc; }
  

  std::cout << matchLoc << " " << minVal << " " << maxVal << " " << minLoc << " " << maxLoc << "\n";

}





};
