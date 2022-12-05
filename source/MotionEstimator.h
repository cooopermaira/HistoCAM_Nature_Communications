//
//  MotionEstimator.h
//  pathCam
//
//  Created by Brian Summa on 11/1/22.
//

#ifndef MotionEstimator_h
#define MotionEstimator_h

#include "pathCam.h"

namespace pathCam{
    
  class MotionEstimator{
  public:
    
    double t_x;
    double t_y;
    
    MotionEstimator(): t_x(0.0), t_y(0.0) {};
    
    int findHomography(pathCam::Match *m, int estimator_type,
                        double ransacReprojThreshold = 3, int maxIters = 2000,
                        double confidence = 0.995);
    
    void phaseCorrelate(Image *image_1, Image *image_2);
    
    
  };
  
}

#endif /* MotionEstimator_h */
