//
//  Loader.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam {
    void LoaderLogicRunnable::run() {

        if(!image->in_memory()){
            image->load_raw_from_disk();
        }

        if (!image->in_memory()) {
            successful = false;
            image->label = Image::_BAD_FILE;
            std::cout << "Image failed to load" << std::endl;
            parent->loaderCount--;
            return;
        }

        image->find_label();

        if (image->is_good()) {
            image->create_reg_image(parent->scale_factor, parent->crop_factor, parent->debayer, parent->interpolation,
                                    parent->real);
            /*
            double blurVal = image->check_blur();
            parent->variancesForDebug[sort_order] = blurVal;
            if(blurVal < 5000.0){
                parent->loaderCount--;
                return;
            }
             */
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
                successful = false;
                image->label = Image::_LOWFEAT;
                parent->loaderCount--;
                return;
            }

            //unsigned long image_index = parent->add_image(image);
            unsigned long image_index = sort_order;
            image->index = image_index;
            parent->add_image(image, image_index);
            auto matchjob = new MatchRunnable(parent, image_index, sort_order + 20);
            parent->matchableCount++;
            parent->JobQ->add_runnable(matchjob);
        } else {
            image->free_memory_RAW();
        }

        parent->loaderCount--;
    }
}
