//
//  Runnables.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include "pathCam.h"

namespace pathCam {


  cuda::GpuMat &getThreadConvertSpace(int width, int height) {
    thread_local cuda::GpuMat buffer;

    if (buffer.cols != width || buffer.rows != height) {
      buffer.create(height, width,CV_32FC1);
    }
    return buffer;
  }

  void ComponentMatchSearch::run() {
    auto myComp = reinterpret_cast<MetricComposite *>(parent->composites[image->regInfo->component_membership]);
    auto matcher = DescriptorMatcher(parent->matcher_type);
    std::vector<Match *> matches;

    //image->siftMutex.lock();
    image->extract_sift(parent->siftPoints,4,0,0.4f,0.1f,
                        getThreadConvertSpace(parent->siftWindow,parent->siftWindow), true);
    //image->siftMutex.unlock();
    image->free_memory_RAW();

    for (long int prev_idx = image_index - 1; prev_idx >= 0; prev_idx--) {
      Image *previous = parent->get_image_ref(prev_idx);

      if (previous == nullptr) {
        continue;
      }

      if (!previous->is_good()) { continue; }

      auto m = new Match(previous, image);
      matcher.match(m);

      if (1 == MotionEstimator::findHomography(m, parent->estimator_type, 10)) {
        //forward match to feature track generator (ftg)
        matches.push_back(m);
      } else {
        delete m;
      }
    }

    myComp->ftg->accessMutex.lock();
    for (auto &match: matches) {
      if (match->image_1->regInfo->component_membership != match->image_2->regInfo->component_membership) {
        auto theirComp = reinterpret_cast<MetricComposite *>
            (parent->composites[match->image_1->regInfo->component_membership]);

        if (myComp->componentMagLabel == theirComp->componentMagLabel) {
          //these two components should actually be the same component. we will suspend one and join to the other
          myComp->componentJoinMatches.push_back(match);
          theirComp->componentJoinMatches.push_back(match);
        }else {
          delete match;
        }
      } else {
        myComp->ftg->store_match(match);
      }
    }
    myComp->ftg->accessMutex.unlock();
    --myComp->outstandingCMS_jobs;
  }

  void MatchRunnable::run() {
    Image *image = parent->get_image_ref(image_idx);


    if (!image->is_good()) {
      return;
    }

    //pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);
    auto matcher = DescriptorMatcher(parent->matcher_type);
    int mostMatches = 0;
    long bestMatch = -1;

    auto tempReg = parent->get_reg_ref(image_idx);

    for (long int prev_idx = image_idx - 1; prev_idx >= 0; prev_idx--) {
      Image *previous = parent->get_image_ref(prev_idx);

      if (previous == nullptr) {
        continue;
      }

      if (!previous->is_good()) { continue; }

      Match *m = new Match(previous, image);
      matcher.match(m);

      int result = MotionEstimator::findHomography(m, parent->estimator_type, 10);

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
          parent->resize_mmatch_mutex.readLock();
          parent->matchM.match[prev_idx][image_idx] = nullptr;
          parent->resize_mmatch_mutex.unlock();
        }
      } else {
        // if(result == -1 || result == -2){
        parent->resize_mmatch_mutex.readLock();
        parent->matchM.match[prev_idx][image_idx] = nullptr;
        parent->resize_mmatch_mutex.unlock();
      }

      delete m;
    }


    if (!successful) {
      parent->add_new_component_Q(image_idx, cv::Size(image->width, image->height));
    }
    //parent->RegistrationConsecQ.add_index(image_idx);

    --parent->matchableCount;
    jobComplete.set();
    successful = true;
  } //end run
}; //end namespace
