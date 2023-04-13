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
  
  friend class RegRunnable;
  
private:
  std::vector <MemoryPool *> mempool;
  Poco::Logger *logger;
  unsigned int threads;
  
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

    
  std::vector < Image *> images;
  MatchMatrix *matchM;
  std::vector < RegInfo > reg_results;


public:
  BatchCam(Poco::Util::LayeredConfiguration::Ptr config, Poco::Logger &Applogger);
  
  ~BatchCam(){
    
    for(unsigned int i=0; i < threads; i++){
      delete mempool[i];
    }
    mempool.clear();
   
    for(unsigned int i=0; i < images.size(); i++){
      delete images[i];
    }
    images.clear();
    

  }
  
  bool run();

private:
  bool parseConfig(Poco::Util::LayeredConfiguration::Ptr pConf);
  bool loadFileList();
  
  bool registration(unsigned int thread_id=0);
  bool compositing();
  
};

}

#endif /* BatchCam_hp */
