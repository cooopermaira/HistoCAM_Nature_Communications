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
    std::vector<unsigned int> skipComponents;

    auto tempReg = parent->get_reg_ref(image_idx);

    for (long int prev_idx = image_idx - 1; prev_idx >= 0; prev_idx--) {
      Image *previous = parent->get_image_ref(prev_idx);

      if (previous == nullptr) {
          continue;
      }

      if (!previous->is_good()) { continue; }

      Match *m = new Match(previous, image);
      matcher->match(m,0);

      int result = motion_est->findHomography(m, parent->estimator_type, 25, 0);

      if (m->good_matches.size() > mostMatches) {
        mostMatches = m->good_matches.size();
        bestMatch = prev_idx;
        tempReg->bestMatch = bestMatch;
        tempReg->numBestMatches = mostMatches;
      }

      if (result == 1) {

        if (std::abs(m->t_x) < image->width / 1 && std::abs(m->t_y) < image->height / 1) {

          parent->set_match(image_idx, prev_idx, m);

          //this should all be in the damn constructor

          tempReg->accessMutex->lock();
          tempReg->index = image_idx;
          tempReg->root = false;
          tempReg->matchedTo = prev_idx;
          tempReg->relativeCoords.x = -1 * m->t_x;
          tempReg->relativeCoords.y = -1 * m->t_y;
          tempReg->accessMutex->unlock();
          tempReg->image = image;

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
      std::unique_lock lock(image->blurMutex);
      image->cudaBufferConVar.wait(lock, [&] { return image->blurSet; });
      parent->add_new_component_Q(image_idx, cv::Size(image->width, image->height));
    }
    //parent->RegistrationConsecQ.add_index(image_idx);


    delete matcher;
    delete motion_est;
    parent->matchableCount--;
    jobComplete.set();
    successful = true;
  } //end run








}; //end namespace
