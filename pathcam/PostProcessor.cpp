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

    auto matchPairs = parent->get_sift_match_Q_front(imagesProcessed);
    if (!matchPairs.empty()) {
      for (auto mp : matchPairs) {
        pMatch matchesInfo;
        matchesInfo.src_img_idx = mp.first->index;
        matchesInfo.dst_img_idx = mp.second->index;

        // Run matching in both directions
        MatchSiftData(mp.first->siftData, mp.second->siftData);
        MatchSiftData(mp.second->siftData, mp.first->siftData);

        std::vector<DMatch> mutualMatches;
        std::vector<Point2f> pts1, pts2;


        // Track mutual matches
        for (int i = 0; i < mp.first->siftData.numPts; ++i) {
          int match_idx = mp.first->siftData.h_data[i].match;
          if (match_idx < 0 || match_idx >= mp.second->siftData.numPts) {continue;}

          // Confirm mutual match
          if (mp.second->siftData.h_data[match_idx].match == i) {
            // DMatch m;
            // m.queryIdx = i;
            // m.trainIdx = match_idx;
            // m.distance = mp.first->siftData.h_data[i].match_error;
            // matches_info.matches.push_back(m);
            mutualMatches.emplace_back(i,match_idx,mp.first->siftData.h_data[i].match_error);
            pts1.emplace_back(mp.first->siftData.h_data[i].xpos,mp.first->siftData.h_data[i].ypos);
            pts2.emplace_back(mp.second->siftData.h_data[match_idx].xpos,mp.second->siftData.h_data[match_idx].ypos);
          }
        }

        std::vector<uchar> inlierMask;
        if (mutualMatches.size() > 8) {
          findHomography(pts1,pts2,RANSAC,3.0,inlierMask);
        }else {
          inlierMask.resize(mutualMatches.size(),1);
        }
        for (size_t i = 0; i < mutualMatches.size();++i) {
          if (inlierMask[i]) {
            matchesInfo.matches.push_back(mutualMatches[i]);
          }
        }
        allMatches.push_back(matchesInfo);
      }
    }
  }

  bool SiftFeatureMatcher::isTerminal() {
    if (!parent->compositing && parent->siftMatchQueue.empty()) {
      FeatureTrackGenerator ftg;
      BundleAdjustmentIntegrator bai(parent);
      auto start = std::chrono::high_resolution_clock::now();

      auto tracks = ftg.generateTracks(imagesProcessed, allMatches);
      bai.setupBundleAdjustment(tracks,imagesProcessed);

      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
      std::cout << duration << "ms" << std::endl;
      for (auto cam : bai.poseVertices) {
        auto pv = bai.optimizer->poseVertex(cam.first);
        auto img = parent->get_image_ref(cam.first);

        //auto t = cam.second->
        std::cout << img->absoluteCoords.x<<" "<<pv->t[0]<<" "<<img->absoluteCoords.y<<" "<<pv->t[1]<<std::endl;
      }
      for (const auto& stat : bai.optimizer->batchStatistics()){
        std::printf("iter: %2d, chi2: %.6f\n", stat.iteration + 1, stat.chi2);
      }
      for (const auto& [id, vertex] : bai.poseVertices) {
        Eigen::Vector3d t = vertex->t;
        std::cout << "Pose " << id << " translation: " << t.transpose() << std::endl;
      }
      int k = 0;
    }
    return !parent->compositing && parent->siftMatchQueue.empty();
  }
}