//
//  Loader.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam {
  void LoaderLogicRunnable::self_cancel(int label) {
    successful = true;
    image->label = label;
    --parent->loaderCount;
    jobComplete.set();
    image->release_reg_image();
    parent->JobQ->cancel_job(2, image_index);
    parent->JobQ->update_job_readiness(2, image_index);
    image->free_memory_RAW();
  }

  void LoaderLogicRunnable::run() {
    image->parent = parent;
    if (!image->in_memory()) {
      image->load_raw_from_disk();
    }


    if (!image->in_memory()) {
      successful = false;
      image->label = Image::_BAD_FILE;
      std::cout << "Image failed to load" << std::endl;
      --parent->loaderCount;
      jobComplete.set();
      return;
    }
    // if (parent->recordingMode) {
    //   image->write_to_path(true);
    // }


    image->index = image_index;


    if (!image->is_mostly_black()) {
      image->check_blur_async(); //this is computationally very expensive even for small windows
      // image->motionBlur = 0.5;

      image->create_reg_image(parent->scale_factor, parent->crop_factor, parent->debayer, parent->interpolation,
                              parent->real);


      auto *detector = new FeatureDetector(parent->feature_type, parent->use_FREAK);

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

      if (image->keypoints.size() < 250) {
        detector->set_ORB_params();
        detector->detect_and_compute(image);
      }
      delete detector;


      if (image->keypoints.size() < 250) {
        std::cout << "LOW FT: " << image_index << std::endl;
        self_cancel(Image::_LOWFEAT);
        return;
      }

      image->release_reg_image();

      //parent->add_image(image, image_index);
      auto matchjob = new MatchRunnable(parent, image_index);
      ++parent->matchableCount;
      parent->JobQ->add_runnable(matchjob);
      successful = true;


      // parent->increment_match_counter(true,image_index);
    } else {
      //std::cout<<"Too Black: "+std::to_string(image_index)<<std::endl;
      parent->mark_neighbors_as_underexposed(image_index);
      image->free_memory_RAW();
    }

    --parent->loaderCount;
    jobComplete.set();
    successful = true;
  }

  bool FeatureDetector::detect_and_compute(pathCam::Image *image) const {
    if (use_FREAK) {
      detector->detect(image->get_reg_image(), image->keypoints);
      extractor->compute(image->get_reg_image(), image->keypoints, image->descriptors);
    } else {
      //if (image->label == Image::_2X) {
      if (false) {
        detector->detectAndCompute(image->get_reg_image(),
                                   image->parent->circleMask, image->keypoints,
                                   image->descriptors);
      } else {
        detector->detectAndCompute(image->get_reg_image(),
                                   noArray(), image->keypoints,
                                   image->descriptors);
      }
    }

    return (image->keypoints.size() > 0);
  }
}
