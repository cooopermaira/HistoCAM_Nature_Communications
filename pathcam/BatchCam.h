//
//  BatchCam.h
//  pathCamLib
//
//  Created by Brian on 3/15/23.
//

#ifndef BatchCam_hp
#define BatchCam_hp

#include "pathCam.h"

using Poco::MemoryPool;
using Poco::Path;
using Poco::Logger;

namespace pathCam{

class BatchCam{
  
  friend class PairRegRunnable;
  
protected:
  std::vector <MemoryPool *> mempool;
  Logger *logger;
  Logger::Ptr results_logger;
  unsigned int threads;
  
  
  Poco::Path input_images;
  Poco::Path out_image;
  Poco::Path flat_field_file;

  
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
  std::vector < RegInfo > reg_results;
  std::vector < Bbox > box;
  
  MatchMatrix matchM;
  OverlapMatrix overlapM;
  Bbox combined_box;


public:
  BatchCam(Poco::Util::LayeredConfiguration::Ptr config);
  
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
  
  virtual bool run();
  
protected:
  bool parseConfig(Poco::Util::LayeredConfiguration::Ptr pConf);
  bool loadFileList();

  virtual bool resolve_bboxes();
  void find_overlaps();
  bool compositing();
  
};

}

#endif /* BatchCam_hp */
