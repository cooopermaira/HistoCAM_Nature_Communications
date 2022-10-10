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
  _BOOST, _DAISY, _LATCH, _LUCID, _MSD, _SURF, _VGG}; //xfeatures2d

class FeatureDetector{
  
  int feature_type;
  bool use_FREAK;
  
  Ptr<FREAK> extractor;
  
  Ptr<Feature2D> detector;
  
  void create_default_detector(){
    switch(feature_type){
      case _SURF:
        detector = SURF::create();
        break;
      case _SIFT:
        detector = SIFT::create();
        break;
      case _AKAZE:
        detector = AKAZE::create();
        break;
      case _BRISK:
        detector = BRISK::create();
        break;
      case _GFFT:
        detector = GFTTDetector::create();
        break;
      default:
        break;
    }
  }
  
  
public:
  
  FeatureDetector(int feature_type=_SURF, bool use_FREAK=false):
                            feature_type(feature_type), use_FREAK(use_FREAK) {
    create_default_detector();
    if(use_FREAK){extractor = FREAK::create();}
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
  
    
  class AKAZEParameters{
  public:
                             
    AKAZE::DescriptorType descriptor_type;
    int descriptor_size;
    int descriptor_channels;
    float threshold;
    int nOctaves;
    int nOctaveLayers;
    KAZE::DiffusivityType diffusivity = KAZE::DIFF_PM_G2;
    
    
    AKAZEParameters(AKAZE::DescriptorType descriptor_type = AKAZE::DESCRIPTOR_MLDB,
                    int descriptor_size = 0, int descriptor_channels = 3,
                    float threshold = 0.001f, int nOctaves = 4,
                    int nOctaveLayers = 4, KAZE::DiffusivityType diffusivity = KAZE::DIFF_PM_G2):
    descriptor_type(descriptor_type),
    descriptor_size(descriptor_size),
    descriptor_channels(descriptor_channels),
    threshold(threshold),
    nOctaves(nOctaves),
    nOctaveLayers(nOctaveLayers),
    diffusivity(diffusivity)
    {};
  };
  
  AKAZEParameters AKAZE_params;
  
  inline void set_AKAZE_params(AKAZEParameters _AKAZE_params){
    AKAZE_params = _AKAZE_params;
    feature_type=_AKAZE;

    if(detector != nullptr){ detector.release(); }
    detector = AKAZE::create(AKAZE_params.descriptor_type,
                             AKAZE_params.descriptor_size,
                             AKAZE_params.descriptor_channels,
                             AKAZE_params.threshold,
                             AKAZE_params.nOctaves,
                             AKAZE_params.nOctaveLayers,
                             AKAZE_params.diffusivity);
  }

  
  
  class BRISKParameters{
  public:
                             
    int thresh;
    int octaves=3;
    float patternScale;
    
    BRISKParameters(int thresh=30, int octaves=3, float patternScale=1.0f):
    thresh(thresh),
    octaves(octaves),
    patternScale(patternScale)
    {};
  };
  
  BRISKParameters BRISK_params;
  
  inline void set_BRISK_params(BRISKParameters _BRISK_params){
    BRISK_params = _BRISK_params;
    feature_type=_BRISK;

    if(detector != nullptr){ detector.release(); }
    detector = BRISK::create(BRISK_params.thresh,
                             BRISK_params.octaves,
                             BRISK_params.patternScale);
  }

  
  
  inline void detect_and_compute(Image *image){
    if(use_FREAK){
      detector->detect(image->get_reg_image(),image->keypoints);
      extractor->compute( image->get_reg_image(), image->keypoints, image->descriptors  );
    }else{
      detector->detectAndCompute(image->get_reg_image(),
                                 noArray(), image->keypoints,
                                 image->descriptors );
    }
  }
  
  
};

}

#endif /* FeatureDetector_hpp */
