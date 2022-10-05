//
//  FeatureDetector.hpp
//  pathCam
//
//  Created by Brian Summa on 10/5/22.
//

#ifndef FeatureDetector_h
#define FeatureDetector_h

#include "common.h"

using namespace cv;
using namespace cv::xfeatures2d;

namespace pathCam{

enum{_AKAZE, _BRISK, _GFFT, _KAZE, _MSER, _ORB, _SIFT, //features2d
  _BOOST, _DAISY, _FREAK, _LATCH, _LUCID, _MSD, _SURF, _VGG}; //xfeatures2d


class FeatureDetector{
  
  int feature_type;
  
  Ptr<Feature2D> detector;
  
  void create_default_detector(){
    switch(feature_type){
      case _SURF:
        detector = SURF::create();
        break;
      default:
        break;
    }
  }
  
  
public:
  
  FeatureDetector(int feature_type=_SURF): feature_type(feature_type) {
    create_default_detector();
  };
  
  //SURF
  struct{
    int minHessian;
  } SURF_params;
  
  
  inline void set_SURF_params(int minHessian){
    SURF_params.minHessian = minHessian;
    if(detector != nullptr){ detector.release(); }
    detector = SURF::create(SURF_params.minHessian);
  }
  
  inline void detect_and_compute(Image *image_1, Image *image_2){
    detector->detectAndCompute(image_1->get_reg_image(),
                               noArray(), image_1->keypoints,
                               image_1->descriptors );
    detector->detectAndCompute(image_2->get_reg_image(),
                               noArray(), image_2->keypoints,
                               image_2->descriptors );
  }
  
  
};

}

#endif /* FeatureDetector_hpp */
