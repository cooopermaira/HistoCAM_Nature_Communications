//
//  Loader.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam {
  void LoaderLogicRunnable::run() {


    if (!image->in_memory()) {
      image->load_raw_from_disk();
    }

    if (!image->in_memory()) {
      successful = false;
      image->label = Image::_BAD_FILE;
      std::cout << "Image failed to load" << std::endl;
      parent->loaderCount--;
      jobComplete.set();
      return;
    }
    if(image->check_blur() < 100){
      successful = true;
      parent->loaderCount--;
      jobComplete.set();
      return;
    }

    image->find_label();

//    if (image->label == Image::_2X || image->label == Image::_4X){
//      if (image->blurVariance < 300){
//        successful = true;
//        parent->loaderCount--;
//        jobComplete.set();
//        return;
//      }
//    }

//    if (!image->decide_label_and_blur()){
//      successful = true;
//      parent->loaderCount--;
//      jobComplete.set();
//      return;
//    }

    if (image->is_good()) {

      image->create_reg_image(parent->scale_factor, parent->crop_factor, parent->debayer, parent->interpolation,
                              parent->real);
      image->reg_scale_initial = parent->scale_factor;
      image->reg_crop_initial = parent->crop_factor;

      image->build_whitebalance_Mat(parent);




      //image->free_memory_RAW();
      pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(parent->feature_type, parent->use_FREAK);
      //pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(7, parent->use_FREAK);

      switch (parent->feature_type) {
        case _SIFT:
          detector->set_SIFT_params(parent->SIFT_params);
          break;
        case _SURF:
          detector->set_SURF_params(parent->SURF_params);
          break;
        case _AKAZE:
          detector->set_AKAZE_params(parent->AKAZE_params);
          break;
        case _BRISK:
          detector->set_BRISK_params(parent->BRISK_params);
          break;
        case _ORB:
          detector->set_ORB_params(parent->ORB_params);
          break;
      }

      detector->detect_and_compute(image,0);

      if (image->keypoints.size() < 200) {
        detector->set_ORB_params();
        detector->detect_and_compute(image,0);
      }
      delete detector;

      if (image->keypoints.size() < 200) {
        successful = true;
        image->label = Image::_LOWFEAT;
        parent->loaderCount--;
        jobComplete.set();
        return;
      }

      if(additionalSiftReg){
        pathCam::FeatureDetector *detector2 = new pathCam::FeatureDetector(7, parent->use_FREAK);
        detector2->detect_and_compute(image,1);
      }

      image->release_reg_image();
      image->free_memory_RAW();

      image->index = image_index;
      parent->add_image(image, image_index);
      auto matchjob = new MatchRunnable(parent, image_index);
      parent->matchableCount++;
      parent->JobQ->add_runnable(matchjob);
    } else {
      image->free_memory_RAW();
    }

    parent->loaderCount--;
    jobComplete.set();
    successful = true;
  }

  bool FeatureDetector::detect_and_compute(pathCam::Image *image, int flag) {

    std::vector<cv::KeyPoint> *points;
    Mat descriptors;
    switch (flag){
      case 0:
        points = &image->keypoints;
        break;
      case 1:
        points = &image->keypointsMultilevel;
        break;
      case 2:
        points = &image->keypointsFull;
        break;
    }

    if(use_FREAK){
      detector->detect(image->get_reg_image(), *points);
      extractor->compute( image->get_reg_image(), *points, descriptors  );
    }else{
      detector->detectAndCompute(image->get_reg_image(),
                                 noArray(), *points,
                                 descriptors );
    }
    switch (flag){
      case 0:
        image->descriptors = descriptors;
        break;
      case 1:
        image->descriptorsMultilevel = descriptors;
        break;
      case 2:
        image->descriptorsFull = descriptors;
        break;
    }
    return (points->size() > 0);

  }
}
