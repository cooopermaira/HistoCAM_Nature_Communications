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
    float *d_ptr = nullptr;
    size_t numFloats = 0;

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
    bool scaleUp) {
    //const int nd = NUM_SCALES + 3;
    const int nd = 8;

    int w = imgW * (scaleUp ? 2 : 1);
    int h = imgH * (scaleUp ? 2 : 1);

    auto align128 = [](int x) { return (x + 127) & ~127; };

    int p = align128(w);
    size_t size = static_cast<size_t>(h) * p;
    size_t sizeTmp = static_cast<size_t>(nd) * h * p;

    for (int i = 0; i < numOctaves; ++i) {
      w >>= 1;
      h >>= 1;
      int p2 = align128(w);
      size += static_cast<size_t>(h) * p2;
      sizeTmp += static_cast<size_t>(nd) * h * p2;
    }

    return size + sizeTmp; // total floats
  }

  inline float *EnsureSiftScratch(
    int imgW, int imgH,
    int numOctaves,
    bool scaleUp) {
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
    std::vector<std::shared_ptr<Match> > matches;

    // image->siftMutex.lock();
    // image->extract_sift(parent->siftPoints, 4, 0, 0.4f, 0.1f,
    //                     getThreadConvertSpace(parent->siftWindow, parent->siftWindow),
    //                     true,EnsureSiftScratch(parent->siftWindow, parent->siftWindow,4,false));
    // image->siftMutex.unlock();
    image->free_memory_RAW(); //incremented in MetricComposite::process_tiles(...)


    for (long int prev_idx = image_index - 1; prev_idx >= 0; prev_idx--) {
      Image *previous = parent->get_image_ref(prev_idx);


      if (previous == nullptr) {continue;}
      if (!previous->is_good()) {continue;}
      if (image->label != Image::_NOLABEL && previous->label != Image::_NOLABEL && image->label != previous->label){continue;}

      // auto meP1 = image->regInfo->absoluteCoords - Point2i(200,200);
      // auto meP2 = image->regInfo->absoluteCoords + Point2i(image->width+200,image->height+200);
      // Rect me(parent->get_AbC_relative_from_relative(image->regInfo->component_membership,meP1,0),
      //   parent->get_AbC_relative_from_relative(image->regInfo->component_membership,meP2,0));
      //
      // auto themP1 = previous->regInfo->absoluteCoords - Point2i(200,200);
      // auto themP2 = previous->regInfo->absoluteCoords + Point2i(previous->width+200,previous->height+200);
      // Rect them(parent->get_AbC_relative_from_relative(previous->regInfo->component_membership,themP1,0),
      //   parent->get_AbC_relative_from_relative(previous->regInfo->component_membership,themP2,0));
      // if ((me & them).empty()){continue;}

      auto m = std::make_shared<Match>(previous, image);
      matcher.match(m);

      if (1 == MotionEstimator::findHomography(m, parent->estimator_type, 30)) {
        m->numMatches = std::accumulate(m->inliers.begin(), m->inliers.end(), 0);
        //forward match to feature track generator (ftg)

        {
          Poco::FastMutex::ScopedLock lock(image->matchesMutex);
          image->matches.push_back(m);
        }
        {
          Poco::FastMutex::ScopedLock lock(previous->matchesMutex);
          previous->matches.push_back(m);
        }
        matches.push_back(m);
      }
    }

    //store INTRA component matches
    {
      Poco::FastMutex::ScopedLock lock(component->ftg->accessMutex);
      for (auto &match : matches) {
        component->ftg->store_match(match);

        if (match->image_1->regInfo->component_membership != component->componentIndex) {
          component->relatedComponents.insert(match->image_1->regInfo->component_membership);
        }
        if (match->image_2->regInfo->component_membership != component->componentIndex) {
          component->relatedComponents.insert(match->image_2->regInfo->component_membership);
        }
      }
    }

    //store INTER component matches
    for (auto &match : matches) {
      if (match->image_1->regInfo->component_membership != match->image_2->regInfo->component_membership) {
        auto theirComp = parent->composites[match->image_1->regInfo->component_membership];

        Poco::FastMutex::ScopedLock lock(theirComp->ftg->accessMutex);
        if (match->image_1->regInfo->component_membership != theirComp->componentIndex) {
          theirComp->relatedComponents.insert(match->image_1->regInfo->component_membership);
        }
        if (match->image_2->regInfo->component_membership != component->componentIndex) {
          theirComp->relatedComponents.insert(match->image_2->regInfo->component_membership);
        }
        theirComp->ftg->store_match(match);
      }
    }


    --component->outstandingCMS_jobs;
    if (component->alignmentHasBegun) {
      throw std::runtime_error("CMS jobs still running after CompositeManager thought they were done");
    }
  }

  void MatchRunnable::build_reg_info(Image *img) const {
    auto t = parent->get_reg_ref(image_idx);
    Poco::FastMutex::ScopedLock lock(t->accessMutex);

    t->index = img->index;
    t->root = false;
    t->image = img;
    img->regInfo = t;
  }

  void MatchRunnable::run() {
    Image *image = parent->get_image_ref(image_idx);
    int votes = 0;

    if (!image->is_good()) {
      return;
    }
    // build_reg_info(image);

    auto matcher = DescriptorMatcher(parent->matcher_type);
    int mostMatches = 0;
    long bestMatch = -1;

    for (long prev_idx = image_idx - 1; prev_idx >= 0; --prev_idx) {
      Image *previous = parent->get_image_ref(prev_idx);

      if (previous == nullptr || !previous->is_good()) {
        continue;
      }

      if (image->label != Image::_NOLABEL && image->label != previous->label) { continue; }

      auto m = std::make_shared<Match>(previous, image);
      matcher.match(m);

      int result = MotionEstimator::findHomography(m, parent->estimator_type, 10);

      if (m->good_matches.size() > mostMatches) {
        mostMatches = m->good_matches.size();
        bestMatch = prev_idx;
        image->regInfo->bestMatch = bestMatch;
        image->regInfo->numBestMatches = mostMatches;
      }

      if (result == 1) {
        if (std::abs(m->t_x) < image->width / 1.1 && std::abs(m->t_y) < image->height / 1.1) {

          votes += m->inlierCount;
          if (votes > RegInfo::minimumVote) {
            successful = true;
          }


          if (votes > RegInfo::featureQuorum) {
            image->regInfo->matchSearchComplete = true;
          }
          ++image->regInfo->outstandingPolls;

          Point2i abc;
          int compIdx;
          if (previous->regInfo->poll_abc(m, abc, compIdx)) {
            image->regInfo->vote_abc(m, abc, compIdx);
          }

          if (image->regInfo->matchSearchComplete) {
            break;
          }
        }
      }
    }
    image->regInfo->matchSearchComplete = true;
    {
      Poco::Mutex::ScopedLock lock(image->regInfo->rAccessMutex);
      if (!image->regInfo->resolved && image->regInfo->outstandingPolls == 0 && successful) {
        //you reached minimumVote but you didnt reach featureQuorum
        image->regInfo->count_votes();
      }
    }


    if (!successful) {
      auto component_index = parent->add_new_component_Q(image_idx, cv::Size(image->width, image->height));
      std::cout << "component " << component_index << " spawning from frame " << image_index << " (" <<
          image->image_file.getBaseName() << ")" << std::endl;
    }

    --parent->matchableCount;
    jobComplete.set();
    successful = true;
  }

  void MatchRunnable::run1() {
    Image *image = parent->get_image_ref(image_idx);

    if (!image->is_good()) {
      return;
    }

    auto matcher = DescriptorMatcher(parent->matcher_type);
    int mostMatches = 0;
    long bestMatch = -1;

    auto tempReg = parent->get_reg_ref(image_idx);


    for (long prev_idx = image_idx - 1; prev_idx >= 0; prev_idx--) {
      Image *previous = parent->get_image_ref(prev_idx);

      if (previous == nullptr || !previous->is_good()) {
        continue;
      }

      if (image->label != Image::_NOLABEL && image->label != previous->label) { continue; }

      auto m = std::make_shared<Match>(previous, image);
      matcher.match(m);

      int result = MotionEstimator::findHomography(m, parent->estimator_type, 10);

      if (m->good_matches.size() > mostMatches) {
        mostMatches = m->good_matches.size();
        bestMatch = prev_idx;
        tempReg->bestMatch = bestMatch;
        tempReg->numBestMatches = mostMatches;
      }

      if (result == 1) {
        if (std::abs(m->t_x) < image->width / 1.1 && std::abs(m->t_y) < image->height / 1.1) {
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

          ++parent->regCount;
          auto rj = new RegistrationRunnable(parent, tempReg);
          parent->JobQ->add_runnable(rj);
          successful = true;

          if (image_idx - prev_idx > 30) {
            std::cout << image_index << " to " << prev_idx << " suspicious" << std::endl;
          }
          break;
        }
      }
    }


    if (!successful) {
      auto component_index = parent->add_new_component_Q(image_idx, cv::Size(image->width, image->height));
      std::cout << "component " << component_index << " spawning from frame " << image_index << " (" <<
          image->image_file.getBaseName() << ")" << std::endl;
    }

    // parent->increment_match_counter(false,image_index);
    --parent->matchableCount;
    jobComplete.set();
    successful = true;
  } //end run
}; //end namespace
