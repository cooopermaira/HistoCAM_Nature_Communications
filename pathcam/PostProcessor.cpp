//
// Created by max on 6/3/25.
//

#include "pathCam.h"
namespace pathCam {

  void PostProcessManager::run() {
    postProcesses.push_back(new SiftFeatureMatcher(parent));

    if (parent->inferencing) {
      //postProcesses.push_back()
    }

    bool terminationMet = false;
    while (!terminationMet) {
      terminationMet = true;
      for (auto pp : postProcesses) {
        pp->run();
        terminationMet = terminationMet && pp->isTerminal();
      }
    }
  }

  SiftFeatureMatcher::SiftFeatureMatcher(StreamCam* parent) : PostProcessorBase(parent) {
  };

  void SiftFeatureMatcher::run() {
    //set device context
    cudaSetDevice(parent->siftCudaDevice);

    auto matchPairs = parent->get_sift_match_Q_front();
    if (!matchPairs.empty()) {
      for (auto mp : matchPairs) {
        cv::detail::MatchesInfo matches_info;
        matches_info.src_img_idx = mp.first->index;
        matches_info.dst_img_idx = mp.second->index;

        // Run matching in both directions
        MatchSiftData(mp.first->siftData, mp.second->siftData);
        MatchSiftData(mp.second->siftData, mp.first->siftData);

        // // Ensure host-side SiftPoints are synchronized from GPU. need xpos, ypos later for bundle adjustment
        // cudaMemcpy(_sift1.h_data, _sift1.d_data, sizeof(SiftPoint) * _sift1.numPts, cudaMemcpyDeviceToHost);
        // cudaMemcpy(_sift2.h_data, _sift2.d_data, sizeof(SiftPoint) * _sift2.numPts, cudaMemcpyDeviceToHost);


        // Track mutual matches
        for (int i = 0; i < mp.first->siftData.numPts; ++i) {
          int match_idx = mp.first->siftData.h_data[i].match;
          if (match_idx < 0 || match_idx >= mp.second->siftData.numPts) {continue;}

          // Confirm mutual match
          if (mp.second->siftData.h_data[match_idx].match == i) {
            DMatch m;
            m.queryIdx = i;
            m.trainIdx = match_idx;
            m.distance = mp.first->siftData.h_data[i].match_error;
            matches_info.matches.push_back(m);
          }
        }
      }
    }
  }

  bool SiftFeatureMatcher::isTerminal() {
    return !parent->compositing && parent->siftMatchQueue.empty();
  }
}