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
  pathCam::FeatureDetector::ORBParameters ORB_params;
  cv::DescriptorMatcher::MatcherType matcher_type;
  int estimator_type;
  
  pathCam::MotionEstimator *mot;
  pathCam::FeatureDetector *detector;
  pathCam::DescriptorMatcher *matcher;


public:
  BatchCam(Poco::Path xml_config);
  
  ~BatchCam(){
    delete imagePool;
  }

private:
  bool parseXML(Poco::Path xml_config);
  
  
};

}

#endif /* BatchCam_hp */
