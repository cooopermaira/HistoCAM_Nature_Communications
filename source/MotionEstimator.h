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
    
    MotionEstimator(){};
    
    int findHomography(pathCam::Match *m, int estimator_type,
                        double ransacReprojThreshold = 3, int maxIters = 2000,
                        double confidence = 0.995);
    
    void phaseCorrelate(pathCam::Match *m, Image *image_1, Image *image_2);
    
    
  };
  
}

#endif /* MotionEstimator_h */
