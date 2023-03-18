//
//  BatchCam.hp
//  pathCamLib
//
//  Created by Brian on 3/15/23.
//

#ifndef BatchCam_hp
#define BatchCam_hp

#include "pathCam.h"

using Poco::MemoryPool;
using Poco::Path;

namespace pathCam{

class BatchCam{
private:
  MemoryPool *imagePool;
  
  Poco::Path input_images;
  Poco::Path output_log;
  Poco::Path out_image;
  
  //Registration Params
  double crop_factor;
  double scale_factor;
  bool debayer;
  bool real;
  int interpolation;
  int feature_type;
  bool use_FREAK;
  cv::DescriptorMatcher::MatcherType matcher_type;
  int estimator_type;
  
  pathCam::FeatureDetector::SIFTParameters SIFT_params;
  pathCam::FeatureDetector::SURFParameters SURF_params;
  pathCam::FeatureDetector::AKAZEParameters AKAZE_params;
  pathCam::FeatureDetector::BRISKParameters BRISK_params;
  pathCam::FeatureDetector::ORBParameters ORB_params;

  
  pathCam::MotionEstimator *mot;
  pathCam::FeatureDetector *detector;
  pathCam::DescriptorMatcher *matcher;
  
  std::vector < Image *> images;
  std::vector < RegInfo > reg_results;


public:
  BatchCam(Poco::Path xml_config);
  
  ~BatchCam(){
    delete imagePool;
    
    for(unsigned int i=0; i < images.size(); i++){
      delete images[i];
    }
    images.clear();
  }
  
  bool run();

private:
  bool parseXML(Poco::Path xml_config);
  bool loadFileList();
  
  bool registration();
  bool compositing();
  
};

}

#endif /* BatchCam_hp */
