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

    image->find_label();

    if (image->is_good()) {
      image->create_reg_image(parent->scale_factor, parent->crop_factor, parent->debayer, parent->interpolation,
                              parent->real);

//      if (image->check_blur() < 800.0) {
//        successful = true;
//        parent->loaderCount--;
//        jobComplete.set();
//        return;
//      }

      image->free_memory_RAW();
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

      detector->detect_and_compute(image);

      if (image->keypoints.size() < 200) {
        detector->set_ORB_params();
        detector->detect_and_compute(image);
      }

      delete detector;

      if (image->keypoints.size() < 200) {
        successful = true;
        image->label = Image::_LOWFEAT;
        parent->loaderCount--;
        jobComplete.set();
        return;
      }

      image->release_reg_image();

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
}
