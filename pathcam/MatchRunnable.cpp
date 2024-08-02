//
//  Runnables.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include "pathCam.h"

namespace pathCam {
  MatchRunnable::MatchRunnable(StreamCam *parent, unsigned long image_idx) : RunnableIntermediate(image_idx, 2),
                                                                             parent(parent),
                                                                             image_idx(image_idx){};

  bool XCompRunnable::match_to_images(Image *selfImage, std::vector<Image *> otherCompImages) {
    Image *matchedTo;
    double scale = 0;
    double mtoScale = 0;
    Point2f offset, offset2;

    //loop through and get homography
    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(
        cv::DescriptorMatcher::MatcherType::BRUTEFORCE);
    pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();
    for (int ii = otherCompImages.size() - 1; ii >= 0; ii--) {
      auto imgCompare = otherCompImages[ii];
      extract_multilevel_keypoints(imgCompare);
      parent->resize_mmatch_mutex->readLock();
      parent->matchM.match[image_idx][imgCompare->index] = new Match(selfImage, imgCompare);
      Match *m = parent->matchM.match[image_idx][imgCompare->index];
      parent->resize_mmatch_mutex->unlock();

      matcher->match(m, 1);
      int result = motion_est->findHomography(m, parent->estimator_type, 500, 1);
      if (result == 1) {

        matchedTo = otherCompImages[ii];
        parent->reg_results_mutex->readLock();
        auto regInfo = parent->reg_results[matchedTo->index];
        parent->reg_results_mutex->unlock();

        if (!regInfo.resolved) {
          parent->JobQ->jobRefs[matchedTo->index * 3 + 2]->waitOnThisGuy();
          parent->reg_results_mutex->readLock();
          auto regInfo = parent->reg_results[matchedTo->index];
          parent->reg_results_mutex->unlock();
          if (!regInfo.resolved) {
            continue;
          }
        }

        mtoScale = parent->composites[matchedTo->component_membership]->imagePyramid->scale;
        if (mtoScale == 0 ){
          parent->composites[matchedTo->component_membership]->imagePyramid->scaleSet.wait();
          mtoScale = parent->composites[matchedTo->component_membership]->imagePyramid->scale;
        }
        assert(mtoScale != 0);

        auto mtoOffset = parent->composites[matchedTo->component_membership]->imagePyramid->offset;
        scale = (m->H.at<double>(0, 0) + m->H.at<double>(1, 1)) / 2.0;
        //scale = 2.0139375;
//        offset = Point2f((((m->t_x / scale + regInfo.absoluteCoords.x) / scale) / mtoScale + mtoOffset.x) / mtoScale,
//                         (((m->t_y / scale + regInfo.absoluteCoords.y) / scale) / mtoScale + mtoOffset.y) / mtoScale);

        offset = Point2f((m->t_x / scale + regInfo.absoluteCoords.x + mtoOffset.x) / scale ,
                         (m->t_y / scale + regInfo.absoluteCoords.y + mtoOffset.y) / scale) ;

        break;
      }

    }
    if (scale > 0) {
      parent->composites[componentMembership]->imagePyramid->set_scale(scale * mtoScale);
      parent->composites[componentMembership]->imagePyramid->set_offset(offset);
      parent->composites[componentMembership]->imagePyramid->scaleSet.set();
      return true;
    }
    return false;
  }

  void XCompRunnable::run() {
    //extract multilevel features from self image
    pathCam::Image *image = parent->get_image_ref(image_idx);
    extract_multilevel_keypoints(image);

    //get references to other component images
    std::vector<unsigned long> indexes;

    for (int i = 0; i < parent->composites.size(); i++) {
      if (parent->composites[i]->componentIndex != componentMembership) {
        for (auto item: parent->composites[i]->delaunayMembers) {
          indexes.push_back(item.second);
        }
      }
    }

    auto otherImages = parent->get_image_refs(indexes);
    if(!match_to_images(image, otherImages)){
      indexes.clear();
      for (int i = 0; i < image->index; i++){
        indexes.push_back((unsigned long) i);
      }
      auto secondTry = parent->get_image_refs(indexes);
      assert(match_to_images(image,secondTry));
    }

    int k = 0;
  }

  void MatchRunnable::run() {
    pathCam::Image *image = parent->get_image_ref(image_idx);


    if (!image->is_good()) {
      return;
    }

    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);

    pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();
    int mostMatches = 0;
    long bestMatch = -1;
    bool tryWaiting = true;
    for (long int prev_idx = image_idx - 1; prev_idx >= 0; prev_idx--) {
      pathCam::Image *previous = parent->get_image_ref(prev_idx);
      if (previous == nullptr) {
        if (max(5, int(image_idx)) <= prev_idx + 5 && tryWaiting) {
          auto res = parent->JobQ->getSortOrderAndJobRefs(1,prev_idx);
          auto waitFor = parent->JobQ->jobRefs[res.first];
          waitFor->waitOnThisGuy();
          parent->debugMatchSuspendThread++;
          tryWaiting = false;
          previous = parent->get_image_ref(prev_idx);
          if (previous == nullptr) { continue; }

          //could remain null if !image->isGood()

        } else { continue; }
      }

      if (!previous->is_good()) { continue; }

      parent->resize_mmatch_mutex->readLock();
      parent->matchM.match[prev_idx][image_idx] = new Match(previous, image);
      Match *m = parent->matchM.match[prev_idx][image_idx];
      parent->resize_mmatch_mutex->unlock();

      matcher->match(m);
      int result = motion_est->findHomography(m, parent->estimator_type, 100, 0);
      if (m->good_matches.size() > mostMatches) {
        mostMatches = m->good_matches.size();
        bestMatch = prev_idx;
      }
      if (result == 1) {
        if (std::abs(m->t_x) < image->width / 2 && std::abs(
            m->t_y) < image->height / 2) {
          //parent->matchM.match[image_idx][prev_idx] = new Match(parent->matchM.match[prev_idx][image_idx]);
          parent->set_match(image_idx, prev_idx);
          auto tempReg = RegInfo(true, Vec2(0.0, 0.0), false, 0);
          tempReg.index = image_idx;
          tempReg.resolved = false;
          tempReg.matchedTo = prev_idx;
          tempReg.relativeCoords.x = -1 * m->t_x;
          tempReg.relativeCoords.y = -1 * m->t_y;
          parent->add_registration(tempReg);
          parent->regCount++;
          auto rj = new RegistrationRunnable(parent, image_idx);
          parent->JobQ->add_runnable(rj);
          successful = true;
          break;
        } else {
          parent->resize_mmatch_mutex->readLock();
          parent->matchM.match[prev_idx][image_idx] = nullptr;
          parent->resize_mmatch_mutex->unlock();
        }
      } else {
        // if(result == -1 || result == -2){
        parent->resize_mmatch_mutex->readLock();
        parent->matchM.match[prev_idx][image_idx] = nullptr;
        parent->resize_mmatch_mutex->unlock();

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
    jobComplete.set();
    successful = true;
  } //end run

  void SingleMatchRunnable::run() {
    pathCam::Image *image1 = parent->get_image_ref(image_idx1);
    pathCam::Image *image2 = parent->get_image_ref(image_idx2);

    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);

    pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();

    parent->matchM.match[image_idx2][image_idx1] = new Match(image2, image1);

    //it's ok to set this match directly without the mutex because the mutex is already locked in
    //perform_global_alignment() and all image_idx values are less than matchM.match size
    Match *m = parent->matchM.match[image_idx2][image_idx1];
    matcher->match(m);
    int result = motion_est->findHomography(m, parent->estimator_type, 100, 0);
//    m->t_x *= image1->get_reg_scale();
//    m->t_y *= image2->get_reg_scale();
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

    auto *detector = new pathCam::FeatureDetector(7, parent->use_FREAK);

    switch (7) {
      case _SIFT:
        //detector->set_SIFT_params();
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
        detector->set_ORB_params();
        //detector->ORB_params.nlevels = 8;
        break;
    }

    image->create_reg_image(parent->scale_factor, 1, parent->debayer, parent->interpolation,
                            parent->real);


    detector->detect_and_compute(image,1);

    image->release_reg_image();

    delete detector;
  }


  SingleMatchRunnable::SingleMatchRunnable(StreamCam *parent, unsigned long image_idx1, unsigned long image_idx2,
                                           unsigned int component_membership, int edgeNumber,
                                           unsigned long sort_order) : RunnableIntermediate(sort_order, 0), parent(parent),
                                                                       image_idx1(image_idx1), image_idx2(image_idx2),
                                                                       component_membership(component_membership),
                                                                       edgeNumber(edgeNumber) {
  };
}; //end namespace
