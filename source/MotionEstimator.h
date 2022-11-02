//
//  MotionEstimator.h
//  pathCam
//
//  Created by Brian Summa on 11/1/22.
//

#ifndef MotionEstimator_h
#define MotionEstimator_h

#include "common.h"

namespace pathCam{
    
  class MotionEstimator{
  public:
    
    double t_x;
    double t_y;
    
    MotionEstimator(): t_x(0.0), t_y(0.0) {};
    
    void findHomography(pathCam::Match *m, int estimator_type){
      
      //-- Localize the object
      std::vector<Point2f> image_1_pts;
      std::vector<Point2f> image_2_pts;
      for( size_t i = 0; i < m->good_matches.size(); i++ )
      {
        //-- Get the keypoints from the good matches
        image_1_pts.push_back( m->image_1->keypoints[ m->good_matches[i].queryIdx ].pt );
        image_2_pts.push_back( m->image_2->keypoints[ m->good_matches[i].trainIdx ].pt );
      }
      
      Mat H = cv::findHomography( image_1_pts, image_2_pts, estimator_type );
      t_x = H.at<double>(0,0)*H.at<double>(0,2)*(1.0/m->image_2->get_reg_scale());
      t_y = H.at<double>(1,1)*H.at<double>(1,2)*(1.0/m->image_2->get_reg_scale());
      
    }
    
    
    void phaseCorrelate(Image *image_1, Image *image_2){
    
      Point2d p = cv::phaseCorrelate(image_1->get_reg_image(), image_2->get_reg_image());
      t_x = p.x*(1.0/image_2->get_reg_scale());
      t_y = p.y*(1.0/image_2->get_reg_scale());
      
    }
    

    
  };
  
}

#endif /* MotionEstimator_h */
