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

  ReverseMatchRunnable::ReverseMatchRunnable(pathCam::StreamCam *parent, unsigned long image_idx,
                                             unsigned long start_from_idx) : RunnableIntermediate(image_idx, 2),
                                                                             parent(parent),
                                                                             image_idx(image_idx),
                                                                             start_from_idx(start_from_idx){};

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


  bool XCompRunnable::match_to_images(Image *selfImage, std::vector<Image *> otherCompImages) {
    Image *matchedTo;
    double scale = 0;
    double mtoScale = 0;
    Point2f offset;

    //loop through and get homography
    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(
        cv::DescriptorMatcher::MatcherType::BRUTEFORCE);
    pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();
    for (int ii = otherCompImages.size() - 1; ii >= 0; ii--) {
      auto imgCompare = otherCompImages[ii];
      //extract_multilevel_keypoints(imgCompare);
      parent->resize_mmatch_mutex->readLock();
      parent->matchM.match[image_idx][imgCompare->index] = new Match(selfImage, imgCompare);
      Match *m = parent->matchM.match[image_idx][imgCompare->index];
      parent->resize_mmatch_mutex->unlock();

      //debug afb

      if(imgCompare->index == 1725 && selfImage->index == 1726){
        scale = 0.497557;
        matchedTo = imgCompare;
        auto mtoRegInfo = parent->get_registration(140);
        if(!mtoRegInfo->resolved){
          mtoRegInfo->waitOnResolve.wait();
        }
        Point2f mtoOffset;
        while(parent->composites.size() <= matchedTo->component_membership || !parent->get_scale_and_offset(mtoRegInfo->component_membership,mtoScale,mtoOffset)) {
          Poco::Thread::sleep(50);
        }
        offset = Point2f((1904.69  + mtoRegInfo->absoluteCoords.x + mtoOffset.x) / scale ,
                         (2508.67  + mtoRegInfo->absoluteCoords.y + mtoOffset.y) / scale) ;
        parent->set_scale_and_offset(componentMembership, scale * mtoScale, offset);
        return true;
      }else if(imgCompare->index == 314 && selfImage->index == 315){
        scale = 0.246104;
        matchedTo = imgCompare;
        auto mtoRegInfo = parent->get_registration(314);
        if(!mtoRegInfo->resolved){
          mtoRegInfo->waitOnResolve.wait();
        }
        Point2f mtoOffset;
        while(parent->composites.size() <= matchedTo->component_membership || !parent->get_scale_and_offset(mtoRegInfo->component_membership,mtoScale,mtoOffset)) {
          Poco::Thread::sleep(50);
        }
        offset = Point2f((2515.841  + mtoRegInfo->absoluteCoords.x + mtoOffset.x) / scale ,
                         (1830.419  + mtoRegInfo->absoluteCoords.y + mtoOffset.y) / scale) ;
        parent->set_scale_and_offset(componentMembership, scale * mtoScale, offset);
        return true;
      }


      matcher->match(m, 1);
      int result = motion_est->findHomography(m, parent->estimator_type, 3, 1);
      if (result == 1) {

        matchedTo = otherCompImages[ii];
        auto mtoRegInfo = parent->get_registration(matchedTo->index);

        mtoRegInfo->accessMutex->lock();
        if (!mtoRegInfo->resolved) {
          mtoRegInfo->set_waiting_component(componentMembership,m);
          mtoRegInfo->accessMutex->unlock();
          return true;
        }
        mtoRegInfo->accessMutex->unlock();

        //if the component hasnt been added yet, wait
        Point2f mtoOffset;
        while(parent->composites.size() <= matchedTo->component_membership || !parent->get_scale_and_offset(mtoRegInfo->component_membership,mtoScale,mtoOffset)) {
          Poco::Thread::sleep(50);
        }

        scale = (m->H.at<double>(0, 0) + m->H.at<double>(1, 1)) / 2.0;
        offset = Point2f((m->t_x / scale + mtoRegInfo->absoluteCoords.x + mtoOffset.x) / scale ,
                         (m->t_y / scale + mtoRegInfo->absoluteCoords.y + mtoOffset.y) / scale) ;

        parent->set_scale_and_offset(componentMembership, scale * mtoScale, offset);
        return true;
      }
    }
    return false;
  }

  void XCompRunnable::run() {
    //extract multilevel features from self image
    pathCam::Image *image = parent->get_image_ref(image_idx);
    //extract_multilevel_keypoints(image);

    //get references to other component images
    std::vector<unsigned long> indexes;
    for (int i = 0; i < image_idx; i++){
      indexes.push_back(i);
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
  }

  void ReverseMatchRunnable::run() {
    pathCam::Image *image = parent->get_image_ref(image_idx);

    if (!image->is_good()) {
      return;
    }

    //pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);
    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(
        cv::DescriptorMatcher::MatcherType::BRUTEFORCE);
    pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();
    int mostMatches = 0;
    long bestMatch = -1;
    bool tryWaiting = true;
    std::vector<unsigned int> skipComponents;


    for (unsigned long prev_idx = start_from_idx; prev_idx < image_idx; prev_idx++) {
      pathCam::Image *previous = parent->get_image_ref(prev_idx);

      if (previous == nullptr) {
        continue;
      }

      if (!previous->is_good()) { continue; }

      Match *m = new Match(previous, image);
      matcher->match(m);

      int result = motion_est->findHomography(m, parent->estimator_type, 20, 0);

      if (m->good_matches.size() > mostMatches) {
        mostMatches = m->good_matches.size();
        bestMatch = prev_idx;
      }
      if (result == 1) {

        if (std::abs(m->t_x) < image->width / 1 && std::abs(m->t_y) < image->height / 1) {

          parent->set_match(image_idx, prev_idx, m);

          auto tempReg = parent->get_registration(image_idx);
          tempReg->accessMutex->lock();
          tempReg->index = image_idx;
          tempReg->root = false;
          tempReg->matchedTo = prev_idx;
          tempReg->relativeCoords.x = -1 * m->t_x;
          tempReg->relativeCoords.y = -1 * m->t_y;
          auto newAbC = Vec2(tempReg->relativeCoords.x + previous->regInfo->absoluteCoords.x,tempReg->relativeCoords.y + previous->regInfo->absoluteCoords.y);
          tempReg->absoluteCoords = newAbC;
          if(abs(tempReg->absoluteCoords.x - newAbC.x) > 300 || abs(tempReg->absoluteCoords.y - newAbC.y) > 300){
            //parent->composites[image->component_membership]->save_pyramid_as_image("20x.png");
            int k = 0;
          }
          tempReg->accessMutex->unlock();

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


    delete matcher;
    delete motion_est;
    parent->composites[image->regInfo->component_membership]->notify_job_complete();
    jobComplete.set();
  } //end run

  void MatchRunnable::run() {
    pathCam::Image *image = parent->get_image_ref(image_idx);

    if (!image->is_good()) {
      return;
    }

    //pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);
    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(
        parent->matcher_type);
    pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();
    int mostMatches = 0;
    long bestMatch = -1;
    bool tryWaiting = true;
    std::vector<unsigned int> skipComponents;


    for (long int prev_idx = image_idx - 1; prev_idx >= 0; prev_idx--) {
      pathCam::Image *previous = parent->get_image_ref(prev_idx);
      
      if (previous == nullptr) {
          continue;
      }
      /*
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
      */
      if (!previous->is_good()) { continue; }

      Match *m = new Match(previous, image);
      matcher->match(m,0);

      int result = motion_est->findHomography(m, parent->estimator_type, 25, 0);

      if (m->good_matches.size() > mostMatches) {
        mostMatches = m->good_matches.size();
        bestMatch = prev_idx;
      }

      if (result == 1) {

        if (std::abs(m->t_x) < image->width / 1 && std::abs(m->t_y) < image->height / 1) {

          parent->set_match(image_idx, prev_idx, m);

          //this should all be in the damn constructor
          auto tempReg = parent->get_registration(image_idx);
          tempReg->accessMutex->lock();
          tempReg->index = image_idx;
          tempReg->root = false;
          tempReg->matchedTo = prev_idx;
          tempReg->relativeCoords.x = -1 * m->t_x;
          tempReg->relativeCoords.y = -1 * m->t_y;
          tempReg->accessMutex->unlock();

          parent->regCount++;
          auto rj = new RegistrationRunnable(parent, tempReg);
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
      parent->JobQ->update_job_readiness(jobTypeFlag,image_idx);
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

    auto matcher = new pathCam::DescriptorMatcher(cv::DescriptorMatcher::MatcherType::BRUTEFORCE);

    auto motion_est = new pathCam::MotionEstimator();

    auto m = new Match(image2, image1);
    matcher->match(m,1);
    int result = motion_est->findHomography(m, parent->estimator_type, 100, 1);

    if (result == 1) {
      if (std::abs(m->t_x) < float(parent->scope_radius / 1.2) && std::abs(
          m->t_y) < float(parent->scope_radius / 1.2)) {
        parent->resize_mmatch_mutex->writeLock();
        parent->matchM.match[image_idx1][image_idx2] = new Match(m);
        parent->resize_mmatch_mutex->unlock();
        parent->composites[component_membership]->matchedEdges[edgeNumber] = {image_idx1, image_idx2};
      }
    }

    delete m;
    delete matcher;
    delete motion_est;
    parent->composites[component_membership]->matchableCount--;
  }



  SingleMatchRunnable::SingleMatchRunnable(StreamCam *parent, unsigned long image_idx1, unsigned long image_idx2,
                                           unsigned int component_membership, int edgeNumber,
                                           unsigned long sort_order) : RunnableIntermediate(sort_order, 0), parent(parent),
                                                                       image_idx1(image_idx1), image_idx2(image_idx2),
                                                                       component_membership(component_membership),
                                                                       edgeNumber(edgeNumber) {
  };
}; //end namespace
