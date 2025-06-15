//
//  Loader.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam {
  void LoaderLogicRunnable::run() {


    image->parent = parent;
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


    // if(image->check_blur() < 100){
    //   successful = true;
    //   parent->loaderCount--;
    //   image->free_memory_RAW();
    //   jobComplete.set();
    //   return;
    // }
    image->index = image_index;
    //image->find_label();

//    if (image->label == Image::_2X || image->label == Image::_4X){
//      if (image->blurVariance < 300){
//        successful = true;
//        parent->loaderCount--;
//        jobComplete.set();
//        return;
//      }
//    }



    if (!image->is_mostly_black()) {

      image->create_reg_image(parent->scale_factor, parent->crop_factor, parent->debayer, parent->interpolation,
                              parent->real);


      image->reg_scale_initial = parent->scale_factor;
      image->reg_crop_initial = parent->crop_factor;


      pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(parent->feature_type, parent->use_FREAK);

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
        image->release_reg_image();
        image->free_memory_RAW();
        return;
      }

      // if(additionalSiftReg){
      //   //image->create_reg_image(1,1,parent->debayer,parent->interpolation,parent->real);
      //   auto detector2 = new pathCam::FeatureDetector(7, parent->use_FREAK);
      //   detector2->detect_and_compute(image,1);
      // }

      image->release_reg_image();

      //parent->add_image(image, image_index);
      auto matchjob = new MatchRunnable(parent, image_index);
      ++parent->matchableCount;
      parent->JobQ->add_runnable(matchjob);
      successful = true;
    } else {
      parent->mark_neighbors_as_underexposed(image_index);
      image->free_memory_RAW();
    }

    parent->loaderCount--;
    jobComplete.set();
    successful = true;
  }

  bool FeatureDetector::detect_and_compute(pathCam::Image *image, int flag) {

    std::vector<cv::KeyPoint> *points;
    Mat reg_image;
    Mat descriptors;
    switch (flag){
      case 0:
        points = &image->keypoints;
        reg_image = image->get_reg_image();
        break;
      case 1:
        points = &image->keypointsMultilevel;
        reg_image = image->get_reg_image();
        break;
      case 2:
        points = &image->keypointsFull;
        break;
    }

    if(use_FREAK){
      detector->detect(reg_image, *points);
      extractor->compute( reg_image, *points, descriptors  );
    }else{
      if(image->label == Image::_2X){

        detector->detectAndCompute(reg_image,
                                   image->parent->regCircleMask, *points,
                                   descriptors );
      }
      detector->detectAndCompute(reg_image,
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
