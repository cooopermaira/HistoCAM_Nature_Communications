//
//  Runnables.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include "pathCam.h"

namespace pathCam {
  struct SiftScratchTLS {
    float*  d_ptr        = nullptr;
    size_t  numFloats    = 0;

    ~SiftScratchTLS() {
      if (d_ptr) {
        cudaFree(d_ptr);
        d_ptr = nullptr;
        numFloats = 0;
      }
    }
  };

  static thread_local SiftScratchTLS tlsSiftScratch;

  inline size_t RequiredSiftScratchFloats(
    int imgW, int imgH,
    int numOctaves,
    bool scaleUp)
  {
    //const int nd = NUM_SCALES + 3;
    const int nd = 8;

    int w = imgW * (scaleUp ? 2 : 1);
    int h = imgH * (scaleUp ? 2 : 1);

    auto align128 = [](int x) { return (x + 127) & ~127; };

    int p = align128(w);
    size_t size    = static_cast<size_t>(h) * p;
    size_t sizeTmp = static_cast<size_t>(nd) * h * p;

    for (int i = 0; i < numOctaves; ++i) {
      w >>= 1;
      h >>= 1;
      int p2 = align128(w);
      size    += static_cast<size_t>(h) * p2;
      sizeTmp += static_cast<size_t>(nd) * h * p2;
    }

    return size + sizeTmp; // total floats
  }

  inline float* EnsureSiftScratch(
    int imgW, int imgH,
    int numOctaves,
    bool scaleUp)
  {
    const size_t required =
        RequiredSiftScratchFloats(imgW, imgH, numOctaves, scaleUp);

    if (tlsSiftScratch.numFloats >= required)
      return tlsSiftScratch.d_ptr;

    if (tlsSiftScratch.d_ptr)
      cudaFree(tlsSiftScratch.d_ptr);

    cudaMalloc(&tlsSiftScratch.d_ptr, required * sizeof(float));
    tlsSiftScratch.numFloats = required;

    return tlsSiftScratch.d_ptr;
  }


  cuda::GpuMat &getThreadConvertSpace(int width, int height) {
    thread_local cuda::GpuMat buffer;

    if (buffer.cols != width || buffer.rows != height) {
      buffer.create(height, width,CV_32FC1);
    }
    return buffer;
  }


  void ComponentMatchSearch::run() {
    auto matcher = DescriptorMatcher(parent->matcher_type);
    std::vector<std::shared_ptr<Match>> matches;

    // image->siftMutex.lock();
    // image->extract_sift(parent->siftPoints, 4, 0, 0.4f, 0.1f,
    //                     getThreadConvertSpace(parent->siftWindow, parent->siftWindow),
    //                     true,EnsureSiftScratch(parent->siftWindow, parent->siftWindow,4,false));
    // image->siftMutex.unlock();
    image->free_memory_RAW();


    for (long int prev_idx = image_index - 1; prev_idx >= 0; prev_idx--) {
      Image *previous = parent->get_image_ref(prev_idx);


      if (previous == nullptr) {continue;}
      if (!previous->is_good()) {continue;}
      if (image->label != Image::_NOLABEL && previous->label != Image::_NOLABEL && image->label != previous->label){continue;}

      Rect me(image->regInfo->absoluteCoords - Point2i(200,200),image->regInfo->absoluteCoords + Point2i(image->width+200,image->height+200));
      Rect them(previous->regInfo->absoluteCoords - Point2i(200,200),previous->regInfo->absoluteCoords + Point2i(previous->width+200,previous->height+200));
      if ((me & them).empty()){continue;}

      auto m = std::make_shared<Match>(previous, image);
      matcher.match(m);

      if (1 == MotionEstimator::findHomography(m, parent->estimator_type, 10)) {
        m->numMatches = std::accumulate(m->inliers.begin(),m->inliers.end(),0);
        //forward match to feature track generator (ftg)
        image->matches.push_back(m);
        previous->matches.push_back(m);
        matches.push_back(m);
      }
    }

    component->ftg->accessMutex.lock();
    for (auto &match: matches) {
      if (match->image_1->regInfo->component_membership != match->image_2->regInfo->component_membership) {
        auto theirComp = std::dynamic_pointer_cast<MetricComposite>
            (parent->composites[match->image_1->regInfo->component_membership]);
        if (!theirComp) {
          throw std::runtime_error("not a metric composite");
        }
        if (component->componentMagLabel == theirComp->componentMagLabel || component->componentMagLabel == Image::_NOLABEL || theirComp->componentMagLabel == Image::_NOLABEL) {
          //these two components should actually be the same component. we will suspend one and join to the other
          component->componentJoinMatches.push_back(match);
          theirComp->componentJoinMatches.push_back(match);
        }
      } else {
        component->ftg->store_match(match);
      }
    }
    component->ftg->accessMutex.unlock();
    --component->outstandingCMS_jobs;
  }

  void MatchRunnable::run() {
    launched = true;
    Image *image = parent->get_image_ref(image_idx);


    if (!image->is_good()) {
      return;
    }

    //pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);
    auto matcher = DescriptorMatcher(parent->matcher_type);
    int mostMatches = 0;
    long bestMatch = -1;

    auto tempReg = parent->get_reg_ref(image_idx);

    for (long prev_idx = image_idx - 1; prev_idx >= 0; prev_idx--) {
      Image *previous = parent->get_image_ref(prev_idx);

      if (previous == nullptr) {
        continue;
      }

      if (!previous->is_good()) { continue; }

      auto m = std::make_shared<Match>(previous,image);
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
          //parent->set_match(image_idx, prev_idx, m);

          //this should all be in the damn constructor

          tempReg->accessMutex.lock();
          tempReg->index = image_idx;
          tempReg->root = false;
          tempReg->matchedTo = prev_idx;
          tempReg->relativeCoords.x = -1 * m->t_x;
          tempReg->relativeCoords.y = -1 * m->t_y;
          tempReg->accessMutex.unlock();
          tempReg->image = image;

          parent->regCount++;
          auto rj = new RegistrationRunnable(parent, tempReg);
          parent->JobQ->add_runnable(rj);
          successful = true;
          break;
        }
      }
    }


    if (!successful) {
      auto component_index = parent->add_new_component_Q(image_idx, cv::Size(image->width, image->height));
      std::cout << "component " << component_index << " spawning from frame " << image_index << " (" <<
          image->image_file.getBaseName()<<")" << std::endl;
    }

    parent->increment_match_counter(false,image_index);
    --parent->matchableCount;
    jobComplete.set();
    successful = true;
  } //end run
}; //end namespace
