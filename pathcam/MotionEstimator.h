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

class RegInfo{
public:
  bool successful,root;
  Vec2 vec;
  RegInfo(bool successful=false, Vec2 vec=Vec2(0.0, 0.0),bool root = false):
  successful(successful), vec(vec) {};
  
  std::string toString(){
    std::stringstream ss;
    ss << ((successful) ? "good" : "bad") << "\t";
    ss << vec.toString();
    return ss.str();
  }
  
};

}

#endif /* MotionEstimator_h */
