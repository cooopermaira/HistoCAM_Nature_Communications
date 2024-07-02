//
//  Runnables.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include "pathCam.h"

namespace pathCam {
  MatchRunnable::MatchRunnable(StreamCam *parent, unsigned long image_idx,
                               unsigned long sort_order) : RunnableIntermediate(sort_order), parent(parent),
                                                           image_idx(image_idx),
                                                           successful(false) {
  };


  void MatchRunnable::run() {
    pathCam::Image *image = parent->get_image_ref(image_idx);


    if (!image->is_good()) {
      return;
    }

    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);

    pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();

    for (long int prev_idx = image_idx - 1; prev_idx >= 0; prev_idx--) {
      pathCam::Image *previous = parent->get_image_ref(prev_idx);
      if (previous == nullptr) {
        continue;
      }

      if (!previous->is_good()) { continue; }
      parent->matchM.match[prev_idx][image_idx] = new Match(previous, image);
      Match *m = parent->matchM.match[prev_idx][image_idx];
      matcher->match(m);
      int result = motion_est->findHomography(m, parent->estimator_type);
      if (result == 1) {
        if (std::abs(parent->matchM.match[prev_idx][image_idx]->t_x) < image->width / 2 && std::abs(
            parent->matchM.match[prev_idx][image_idx]->t_y) < image->height / 2) {
          //parent->matchM.match[image_idx][prev_idx] = new Match(parent->matchM.match[prev_idx][image_idx]);
          parent->set_match(image_idx, prev_idx);
          auto tempReg = RegInfo(true, Vec2(0.0, 0.0), false, 0);
          tempReg.index = image_idx;
          tempReg.resolved = false;
          tempReg.matchedTo = prev_idx;
          tempReg.relativeCoords.x = parent->matchM.match[image_idx][prev_idx]->t_x;
          tempReg.relativeCoords.y = parent->matchM.match[image_idx][prev_idx]->t_y;
          parent->reg_results[image_idx] = tempReg;
          successful = true;
          auto rj = new RegistrationRunnable(parent, image_idx, sort_order + 20);
          parent->regCount++;
          parent->JobQ->add_runnable(rj);
          break;
        } else {
          parent->matchM.match[prev_idx][image_idx] = nullptr;
        }
      } else {
        // if(result == -1 || result == -2){
        parent->matchM.match[prev_idx][image_idx] = nullptr;
      }
      delete m;
    }

    if (!successful) {
      parent->add_new_component_Q(image_idx, cv::Size(image->width, image->height));
    }
    //parent->RegistrationConsecQ.add_index(image_idx);

    delete matcher;
    delete motion_est;
    parent->matchableCount--;
  } //end run

  void SingleMatchRunnable::run() {
    pathCam::Image *image1 = parent->get_image_ref(image_idx1);
    pathCam::Image *image2 = parent->get_image_ref(image_idx2);

    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);

    pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();

    parent->matchM.match[image_idx2][image_idx1] = new Match(image2, image1);
    Match *m = parent->matchM.match[image_idx2][image_idx1];
    matcher->match(m);
    int result = motion_est->findHomography(m, parent->estimator_type);
    if (result == 1) {
      if (std::abs(parent->matchM.match[image_idx2][image_idx1]->t_x) < image1->width / 2 && std::abs(
          parent->matchM.match[image_idx2][image_idx1]->t_y) < image1->height / 2) {
        parent->matchM.match[image_idx1][image_idx2] = new Match(parent->matchM.match[image_idx2][image_idx1]);
        parent->composites[component_membership]->matchedEdges[edgeNumber] = {image_idx1, image_idx2};
      } else {
        parent->matchM.match[image_idx2][image_idx1] = nullptr;
      }
    } else {
      // if(result == -1 || result == -2){
      parent->matchM.match[image_idx2][image_idx1] = nullptr;
    }
    delete m;
    delete matcher;
    delete motion_est;
    parent->composites[component_membership]->matchableCount--;
  }

  void XCompRunnable::extract_multilevel_keypoints(pathCam::Image *image) {
    auto *detector = new pathCam::FeatureDetector(parent->feature_type, parent->use_FREAK);

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
        detector->ORB_params.nlevels = 8;
        break;
    }

    image->create_reg_image(parent->scale_factor, parent->crop_factor, parent->debayer, parent->interpolation,
                            parent->real);

    detector->detect_and_compute_multilevel(image);

    image->release_reg_image();

    delete detector;
  }

  void XCompRunnable::run() {
    //extract multilevel features from self image
    pathCam::Image *image = parent->get_image_ref(image_idx);
    extract_multilevel_keypoints(image);


  }

  SingleMatchRunnable::SingleMatchRunnable(StreamCam *parent, unsigned long image_idx1, unsigned long image_idx2,
                                           unsigned int component_membership, int edgeNumber,
                                           unsigned long sort_order) : RunnableIntermediate(sort_order), parent(parent),
                                                                       image_idx1(image_idx1), image_idx2(image_idx2),
                                                                       component_membership(component_membership),
                                                                       edgeNumber(edgeNumber) {
  };
}; //end namespace
