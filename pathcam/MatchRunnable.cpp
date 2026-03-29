//
//  Runnables.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include "pathCam.h"

namespace pathCam {

  void ComponentMatchSearch::run() {
    auto matcher = DescriptorMatcher(parent->matcher_type);
    std::vector<std::shared_ptr<Match> > matches;

    // image->siftMutex.lock();
    // image->extract_sift(parent->siftPoints, 4, 0, 0.4f, 0.1f,
    //                     getThreadConvertSpace(parent->siftWindow, parent->siftWindow),
    //                     true,EnsureSiftScratch(parent->siftWindow, parent->siftWindow,4,false));
    // image->siftMutex.unlock();
    image->free_memory_RAW(); //incremented in MetricComposite::process_tiles(...)

    bool empty = false;
    if (candidates.empty() && image_index > 0) {
      empty = true;
      std::vector<long> indexes(image_index);
      for (long int prev_idx = image_index - 1; prev_idx >= 0; prev_idx--) {
        indexes.push_back(prev_idx);
      }
    }

    auto start = std::chrono::high_resolution_clock::now();
    int count = 0;
    for (auto candidate : candidates) {

      if (candidate == nullptr) {continue;}
      if (!candidate->is_good()) {continue;}
      if (image->label != Image::_NOLABEL && candidate->label != Image::_NOLABEL && image->label != candidate->label){continue;}

      ++image->matchCount;

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

      auto m = std::make_shared<Match>(candidate, image);
      matcher.match(m);

      if (1 == MotionEstimator::findHomography(m, parent->estimator_type, 30)) {
        m->numMatches = std::accumulate(m->inliers.begin(), m->inliers.end(), 0);
        //forward match to feature track generator (ftg)

        // {
        //   Poco::FastMutex::ScopedLock lock(image->matchesMutex);
        //   image->matches.push_back(m);
        // }
        // {
        //   Poco::FastMutex::ScopedLock lock(previous->matchesMutex);
        //   previous->matches.push_back(m);
        // }
        matches.push_back(m);
      }

      ++count;
    }
    parent->cmsCount += count;
    auto v = std::chrono::duration_cast<std::chrono::milliseconds>(
  std::chrono::high_resolution_clock::now() - start).count();
    parent->cmsTime += v;

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

}; //end namespace
