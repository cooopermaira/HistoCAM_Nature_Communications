//
// Created by max on 6/3/25.
//

#include "pathCam.h"

namespace pathCam {
  void PostProcessManager::run() {
    postProcesses.push_back(parent->sfm);

    if (parent->inferencing) {
      //postProcesses.push_back()
    }

    for (auto &pp: postProcesses) {
      pp->init();
    }

    bool terminationMet = false;
    while (!terminationMet) {
      terminationMet = true;
      for (auto pp: postProcesses) {
        pp->run();
        terminationMet = terminationMet && pp->isTerminal();
      }
    }
  }

  SiftFeatureMatcher::SiftFeatureMatcher(StreamCam *parent) : PostProcessorBase(parent),
                                                              bai(new BundleAdjustmentIntegrator(parent)),
                                                              ftg(new FeatureTrackGenerator),
                                                              queueMutex(new Poco::FastMutex),
                                                              loopMutex(new Poco::FastMutex),
                                                              loopInProcess(false),
                                                              matchWorkOutstanding(0) {
  };

  void SiftFeatureMatcher::init() {
    //std::thread([this]() { this->postMatchProcessLoop(); }).detach();
  }


  void SiftFeatureMatcher::run() {
    //set device context
    cudaSetDevice(parent->siftCudaDevice);

    auto matchPairs = parent->get_sift_match_Q_front(imagesProcessed);

    if (!matchPairs.empty()) {
      loopInProcess = true;
      for (auto mp: matchPairs) {
        // Run matching in both directions
        MatchSiftData(mp.first->siftData, mp.second->siftData);
        MatchSiftData(mp.second->siftData, mp.first->siftData);

        // queueMutex->lock();
        // postMatchQueue.push(mp);
        // queueMutex->unlock();
        //
        // continue;
        pMatch matchesInfo;
        matchesInfo.src_img_idx = mp.first->index;
        matchesInfo.dst_img_idx = mp.second->index;
        std::vector<DMatch> mutualMatches;
        std::vector<Point2f> pts1, pts2;


        // Track mutual matches
        for (int i = 0; i < mp.first->siftData.numPts; ++i) {
          int match_idx = mp.first->siftData.h_data[i].match;
          if (match_idx < 0 || match_idx >= mp.second->siftData.numPts) { continue; }

          // Confirm mutual match
          if (mp.second->siftData.h_data[match_idx].match == i) {
            mutualMatches.emplace_back(i, match_idx, mp.first->siftData.h_data[i].match_error);

            pts1.emplace_back(mp.first->siftData.h_data[i].xpos, mp.first->siftData.h_data[i].ypos);
            pts2.emplace_back(mp.second->siftData.h_data[match_idx].xpos, mp.second->siftData.h_data[match_idx].ypos);
          }
        }

        std::vector<uchar> inlierMask;
        int numInliers;
        if (mutualMatches.size() > 8) {
          Mat H = findHomography(pts1, pts2, RANSAC, 3.0, inlierMask);
          numInliers = std::count(inlierMask.begin(), inlierMask.end(), 1);
          if (mp.second->regInfo->root) {
            int k = 0;
          }
        }
        if (numInliers > 50) {
          for (size_t i = 0; i < mutualMatches.size(); ++i) {
            if (inlierMask[i]) {
              matchesInfo.matches.push_back(mutualMatches[i]);
              ftg->process_match(mp.first->index, mp.second->index, mutualMatches[i]);
            }
          }
        }

        allMatches.push_back(matchesInfo);
        --matchWorkOutstanding;
      }
    }
    loopInProcess = false;
  }

  void SiftFeatureMatcher::postMatchProcessLoop() {
    sleep(1);
    //bool term = isTerminal();
    while (parent->compositing || matchWorkOutstanding > 0) {
      queueMutex->lock();
      if (postMatchQueue.empty()) {
        queueMutex->unlock();
        continue;
      }
      loopMutex->lock();

      auto mp = postMatchQueue.front();
      postMatchQueue.pop();
      queueMutex->unlock();

      pMatch matchesInfo;
      matchesInfo.src_img_idx = mp.first->index;
      matchesInfo.dst_img_idx = mp.second->index;
      std::vector<DMatch> mutualMatches;
      std::vector<Point2f> pts1, pts2;


      // Track mutual matches
      for (int i = 0; i < mp.first->siftData.numPts; ++i) {
        int match_idx = mp.first->siftData.h_data[i].match;
        if (match_idx < 0 || match_idx >= mp.second->siftData.numPts) { continue; }

        // Confirm mutual match
        if (mp.second->siftData.h_data[match_idx].match == i) {
          mutualMatches.emplace_back(i, match_idx, mp.first->siftData.h_data[i].match_error);

          pts1.emplace_back(mp.first->siftData.h_data[i].xpos, mp.first->siftData.h_data[i].ypos);
          pts2.emplace_back(mp.second->siftData.h_data[match_idx].xpos, mp.second->siftData.h_data[match_idx].ypos);
        }
      }

      std::vector<uchar> inlierMask;
      if (mutualMatches.size() > 8) {
        findHomography(pts1, pts2, RANSAC, 3.0, inlierMask);
      } else {
        inlierMask.resize(mutualMatches.size(), 1);
      }
      for (size_t i = 0; i < mutualMatches.size(); ++i) {
        if (inlierMask[i]) {
          matchesInfo.matches.push_back(mutualMatches[i]);
          ftg->process_match(mp.first->index, mp.second->index, mutualMatches[i]);
        }
      }

      allMatches.push_back(matchesInfo);
      loopMutex->unlock();
    }
    int k = 0;
  }


  bool SiftFeatureMatcher::isTerminal() {
    return !parent->compositing && parent->siftMatchQueue.empty() && parent->siftMatchQueue.empty();
  }

  bool SiftFeatureMatcher::tracksReady() {
    // loopMutex->lock();
    // bool answer = parent->siftMatchQueue.empty() && !loopInProcess && parent->siftDataQueue.empty();
    // loopMutex->unlock();
    bool answer = matchWorkOutstanding == 0;
    if (answer) {
      int k = 0;
    }
    return answer;
  }
}
