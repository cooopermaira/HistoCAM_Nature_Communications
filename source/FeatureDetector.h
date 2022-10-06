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
      case _SIFT:
        detector = SIFT::create();
        break;
      default:
        break;
    }
  }
  
  
public:
  
  FeatureDetector(int feature_type=_SURF): feature_type(feature_type) {
    create_default_detector();
  };
  
  ~FeatureDetector(){ detector.release(); };

  
  //SURF
  struct{
    int minHessian;
  } SURF_params;
  
  
  inline void set_SURF_params(int minHessian){
    SURF_params.minHessian = minHessian;
    feature_type=_SURF;
    if(detector != nullptr){ detector.release(); }
    detector = SURF::create(SURF_params.minHessian);
  }
  
  struct{
    int nfeatures;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;
  } SIFT_params;
  
  inline void set_SIFT_params(int   nfeatures,
                              int   nOctaveLayers,
                              double   contrastThreshold,
                              double   edgeThreshold,
                              double   sigma){
    SIFT_params.nfeatures = nfeatures;
    SIFT_params.nOctaveLayers = nOctaveLayers;
    SIFT_params.contrastThreshold = contrastThreshold;
    SIFT_params.edgeThreshold = edgeThreshold;
    SIFT_params.sigma = sigma;
    
    feature_type=_SIFT;

    if(detector != nullptr){ detector.release(); }
    detector = SIFT::create(SIFT_params.nfeatures,
                            SIFT_params.nOctaveLayers,
                            SIFT_params.contrastThreshold,
                            SIFT_params.edgeThreshold,
                            SIFT_params.sigma);
  }
  
  
  inline void detect_and_compute(Image *image){
    detector->detectAndCompute(image->get_reg_image(),
                               noArray(), image->keypoints,
                               image->descriptors );
  }
  
  
};

}

#endif /* FeatureDetector_hpp */
