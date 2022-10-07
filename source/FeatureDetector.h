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
  class SURFParameters{
  public:
    double hessianThreshold;
    int nOctaves;
    int nOctaveLayers;
    bool extended;
    bool upright;
    
    SURFParameters(double hessianThreshold=100,
                int nOctaves=4,
                int nOctaveLayers=3,
                bool extended=false,
                bool upright=false):
                    hessianThreshold(hessianThreshold),
                    nOctaves(nOctaves),
                    nOctaveLayers(nOctaveLayers),
                    extended(extended),
                    upright(upright) {};
  };
  
  SURFParameters SURF_params;
  
  inline void set_SURF_params(SURFParameters _SURF_params){
    SURF_params = _SURF_params;
    feature_type=_SURF;
    if(detector != nullptr){ detector.release(); }
    detector = SURF::create(SURF_params.hessianThreshold,
                            SURF_params.nOctaves,
                            SURF_params.nOctaveLayers,
                            SURF_params.extended,
                            SURF_params.upright );
  }
  
  class SIFTParameters{
  public:
    int nfeatures;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;
    
    SIFTParameters(int   nfeatures=0,
                   int   nOctaveLayers=3,
                   double   contrastThreshold=0.04,
                   double   edgeThreshold=10,
                   double   sigma=1.6):
                          nfeatures(nfeatures),
                          nOctaveLayers(nOctaveLayers),
                          contrastThreshold(contrastThreshold),
                          edgeThreshold(edgeThreshold),
                          sigma(sigma) {};
  };
  
  SIFTParameters SIFT_params;
  
  inline void set_SIFT_params(SIFTParameters _SIFT_params){
    SIFT_params = _SIFT_params;
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
