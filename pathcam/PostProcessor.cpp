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
    bool rootFound = false;
    RegInfo* rootRegRef;

    auto matchPairs = parent->get_sift_match_Q_front(imagesProcessed);

    if (!matchPairs.empty()) {
      loopInProcess = true;
      rootFound = matchPairs.back().second->regInfo->root;
      if (rootFound) {
        rootRegRef = matchPairs.back().second->regInfo;
      }


      size_t mostMutualMatches = 0,mostInliers = 0;
      for (auto mp: matchPairs) {
        if (mp.first->regInfo->component_membership == mp.second->regInfo->component_membership) {
          //match with orb features
          Match m = Match(mp.first, mp.second);
          matcher->match(&m);
        }
        ++numMatchesProcessed;
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
        Mat H;

        if (mutualMatches.size() > mostMutualMatches) {
          mostMutualMatches = mutualMatches.size();
        }
        if (mutualMatches.size() > 200 || (mp.first->regInfo->component_membership != mp.second->regInfo->component_membership && mutualMatches.size() > 10)) {
          H = findHomography(pts2, pts1, RANSAC, 3.0, inlierMask);
          numInliers = std::count(inlierMask.begin(), inlierMask.end(), 1);

          if (numInliers > mostInliers) {
            mostInliers = numInliers;
          }

          if (numInliers > 150|| (mp.first->regInfo->component_membership != mp.second->regInfo->component_membership && numInliers > 10)) {
            for (size_t i = 0; i < mutualMatches.size(); ++i) {
              if (inlierMask[i]) {
                matchesInfo.matches.push_back(mutualMatches[i]);
                ftg->process_match(mp.first->index, mp.second->index, mutualMatches[i]);

                if (mp.first->regInfo->component_membership == mp.second->regInfo->component_membership) {
                  auto m = new Match(mp.first,mp.second);
                  parent->set_match(mp.first->index,mp.second->index,m,false);

                  m->H = H;
                  m->t_x = H.at<double>(0, 2);
                  m->t_y = H.at<double>(1, 2);
                }
              }
            }
            //logic for handling root when called as part of adding new component
            auto myRi = mp.second->regInfo;
            if (myRi->root && !myRi->rootOfRoot) {

              double relativeScale = (H.at<double>(0, 0) + H.at<double>(1, 1)) / 2;

              unsigned int queryComponentSpace = mp.first->regInfo->component_membership;
              Point2f theirAbC(mp.first->regInfo->absoluteCoords.x, mp.first->regInfo->absoluteCoords.y);
              Point2f pairwiseDistance = Point2f(H.at<double>(0, 2), H.at<double>(1, 2));
              Point2f queryAbC = pairwiseDistance + theirAbC;
              auto resultantPoint = parent->get_AbC_relative_from_relative(queryComponentSpace, queryAbC, 0);

              double scale = relativeScale * parent->composites[mp.first->regInfo->component_membership]->imagePyramid->scale;
              myRi->rootHomographies.emplace_back(resultantPoint, scale);
              int k = 0;
              // if (myRi->rootHomographies.size() > 4) {
              //   break;
              // }
            }
          }
        }

        allMatches.push_back(matchesInfo);
        --matchWorkOutstanding;
      }
      // if (rootFound) {
      //   assert(rootRegRef->rootHomographies.size() > 0);
      //   double scale;
      //   Point2f rootGuess;
      //   rootRegRef->average_from_homographies(rootGuess,scale);
      //   auto comp = parent->composites[rootRegRef->component_membership];
      //   cudaSetDevice(parent->compositorCudaDevice);
      //   comp->set_scale(scale);
      //   comp->set_offset(rootGuess/scale);
      //   comp->wakeEvent.set();
      // }
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
    bool terminate = !parent->compositing && parent->siftMatchQueue.empty();
    if (terminate){std::cout<<"matches processed: "+std::to_string(numMatchesProcessed)<<std::endl;}
    return terminate;
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
