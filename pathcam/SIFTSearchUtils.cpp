//
// Created by cooper maira on 6/4/25.
//

#include "pathCam.h"
#include <unordered_map>

namespace pathCam {
  bool are_same_feature(const Image *img, int f1, int f2) {
    const auto &kp1 = img->keypoints[f1];
    const auto &kp2 = img->keypoints[f2];

    float dx = kp1.pt.x - kp2.pt.x;
    float dy = kp1.pt.y - kp2.pt.y;

    if (dx * dx + dy * dy > 3 * 3) return false;

    float angleDiff = std::abs(kp1.angle - kp2.angle);
    angleDiff = std::min(angleDiff, 360.f - angleDiff);
    if (angleDiff > 10.f) return false;

    if (kp1.octave != kp2.octave) return false;

    int hamming = cv::norm(img->descriptors.row(f1),
                           img->descriptors.row(f2),
                           cv::NORM_HAMMING);

    return hamming < 20;
  }


  int FeatureTrackGenerator::get_or_create_feature_index(const ImageFeaturePair &_pair) {
    auto it = feature_to_index.find(_pair);
    if (it != feature_to_index.end()) {
      return it->second;
    }

    // Create new index
    int new_index = index_to_feature.size();
    feature_to_index[_pair] = new_index;
    index_to_feature.push_back(_pair);
    baFeatures.push_back(new BAFeature);


    // add spot for tracking its members
    component_features.resize(new_index + 1);
    component_features[new_index][_pair.image_id] = _pair.feature_id;
    adjacency.resize(new_index + 1);

    // Expand Union-Find if necessary
    if (!uf_ptr) {
      uf_ptr = std::make_unique<UnionFind>(1);
    } else if (new_index >= uf_ptr->size()) {
      // Expand Union-Find structure
      uf_ptr->expand(new_index + 1);
    }

    return new_index;
  }

  std::pair<Point2i, bool> FeatureTrackGenerator::estimate_image_coords_from_feature_tracks(Image *img) {
    Poco::RWLock::ScopedReadLock lock(rwLock);

    float count = 0;
    float xTotal = 0,yTotal = 0;

    for (auto &obs : img->observations) {
      bool one = obs->feature->find()->live;
      bool two = obs->feature->find()->lastIteration > 0;
      if (obs->feature->find()->live && obs->feature->find()->lastIteration > 0) {
        ++count;
        xTotal += obs->feature->find()->xy.x - obs->obs_x;
        yTotal += obs->feature->find()->xy.y - obs->obs_y;
      }
    }

    if (count > 0) {
      xTotal /= count;
      yTotal /= count;
      return {Point2i(xTotal,yTotal),true};
    }
    return {{},false}; //uh oh, what happens if none of the features have ever been updated? we return false
  }



  void FeatureTrackGenerator::consume(FeatureTrackGenerator *consumedFTG, Point2f translation) {
    baFeatures.reserve( baFeatures.size() + consumedFTG->baFeatures.size());
    baImages.reserve(baImages.size() + consumedFTG->baImages.size());

    for (auto baImg : consumedFTG->baImages) {
      baImg->xy += translation;
      baImages.push_back(baImg);
    }
    for (auto baFt : consumedFTG->baFeatures) {
      baFt->xy += translation;
      baFeatures.push_back(baFt);
    }

  }


  void FeatureTrackGenerator::add_image(Image *img) {
    if (img->addedToFTG){return;}

    img->observations.resize(img->keypoints.size(), nullptr);
    auto baImg = new BAImage(img->regInfo->absoluteCoords, img->regInfo->root,
                             img->regInfo);
    baImages.push_back(baImg);

    baFeatures.reserve(baFeatures.size() + img->keypoints.size());
    const float regScale = img->get_reg_scale();

    Poco::RWLock::ScopedWriteLock lock(featureGrid.rwLock); //for .insert()

    for (int i = 0; i < img->observations.size(); ++i) {
      const auto ftPixelCoords = img->keypoints[i].pt / regScale;

      auto feat = new BAFeature;
      baFeatures.push_back(feat);

      img->observations[i] = new Observation(baImg, feat, ftPixelCoords.x, ftPixelCoords.y);

      feat->xy = Point2f(img->regInfo->absoluteCoords) + ftPixelCoords;
      feat->imageFeatures.emplace(img, i);

      featureGrid.insert(feat);
    }

    img->addedToFTG = true;
  }

  int FeatureTrackGenerator::process_match_queue() {
    Poco::FastMutex::ScopedLock lock(accessMutex);
    int count = queuedMatches.size();
    while (!queuedMatches.empty()) {
      auto m = queuedMatches.front();
      queuedMatches.pop();
      process_match2(m);
    }
    return count;
  }

  void FeatureTrackGenerator::process_match2(const std::shared_ptr<Match> &match_) {
    auto [it,inserted] = matches.insert(match_);
    if (!inserted){return;}

    auto img1 = match_->image_1;
    auto img2 = match_->image_2;

    Poco::RWLock::ScopedWriteLock lock(featureGrid.rwLock); //for .remove()
    for (int i = 0; i < match_->good_matches.size(); ++i) {
      if (match_->inliers[i]) {
        auto qInd = match_->good_matches[i].queryIdx;
        auto tInd = match_->good_matches[i].trainIdx;

        //path compression happens here, so baFeat1 and baFeat2 are their own parents
        auto baFeat1 = img1->observations[qInd]->feature->find();
        auto baFeat2 = img2->observations[tInd]->feature->find();

        if (baFeat1 != baFeat2) {
          bool conflict = false;
          auto *small = (baFeat1->imageFeatures.size() < baFeat2->imageFeatures.size()) ? baFeat1 : baFeat2;
          auto *large = (small == baFeat1) ? baFeat2 : baFeat1;

          for (const auto &[img, feat_id]: small->imageFeatures) {
            auto it = large->imageFeatures.find(img);
            if (it != large->imageFeatures.end() && it->second != feat_id) {
              conflict = true;
              break;
            }
          }

          /*
           make the merge, deactivate if there was a conflict. clear
           imageFeature list from loser, remove loser from featureGrid */
          BAFeature *winner, *loser;
          // long imgIdx;
          // int ftIdx;

          if (baFeat1->lastIteration < baFeat2->lastIteration) {
          // if (baFeat1->stableID.size() < baFeat2->stableID.size()){
            winner = baFeat2;
            loser = baFeat1;
            // imgIdx = img2->index;
            // ftIdx = tInd;
          } else {
            winner = baFeat1;
            loser = baFeat2;
            // imgIdx = img1->index;
            // ftIdx = qInd;
          }

          loser->parent = winner;
          winner->imageFeatures.insert(loser->imageFeatures.begin(),loser->imageFeatures.end());
          loser->imageFeatures.clear();
          featureGrid.remove(loser);

          winner->live = true;

          // if (winner->stableID.empty()) {
          //   if(!loser->stableID.empty()) {
          //     int k = 0;
          //   }
          //   winner->stableID.push_back(make_stable_id(imgIdx,ftIdx));
          // }else if (!loser->stableID.empty()) {
          //   winner->stableID.insert(winner->stableID.end(),loser->stableID.begin(),loser->stableID.end());
          // }


          if (conflict) {
            winner->active = false;
            featureGrid.remove(winner);
          }
        }
      }
    }
  }

  void FeatureTrackGenerator::process_match(const std::shared_ptr<Match> &match_) {
    auto srcImgIdx = match_->image_1->index;
    auto dstImgIdx = match_->image_2->index;

    for (int i = 0; i < match_->good_matches.size(); ++i) {
      if (match_->inliers[i]) {
        auto qInd = match_->good_matches[i].queryIdx;
        auto tInd = match_->good_matches[i].trainIdx;
        ImageFeaturePair feat1{srcImgIdx, qInd};
        ImageFeaturePair feat2{dstImgIdx, tInd};

        // Get or create indices for both features
        int idx1 = get_or_create_feature_index(feat1);
        int idx2 = get_or_create_feature_index(feat2);

        int root1 = uf_ptr->find(idx1);
        int root2 = uf_ptr->find(idx2);

        if (root1 != root2) {
          const auto &set1 = component_features[root1];
          const auto &set2 = component_features[root2];

          // Check if merging would close a loop
          bool conflict = false;
          for (const auto &[img_id, feat_idx]: component_features[root1]) {
            auto it = component_features[root2].find(img_id);
            if (it != component_features[root2].end()) {
              //this match closes a loop, check if there is an inconsistency for this feature
              if (it->second != feat_idx) {
                // auto img = imageRefs[img_id];
                // if (are_same_feature(img,it->second,feat_idx)) {
                //
                // }
                conflict = true; // same image, different feature -> conflict
                break;
              }
            }
          }

          // uf_ptr->unite(root1,root2);
          // auto newRoot = uf_ptr->find(root1);


          if (conflict) {
            ++invalidCount;
          } else {
            // safe to merge
            uf_ptr->unite(root1, root2);

            int new_root = uf_ptr->find(root1);

            // merge metadata
            if (new_root == root1) {
              component_features[root1].insert(set2.begin(), set2.end());
              adjacency[root1].insert(adjacency[root1].end(), adjacency[root2].begin(), adjacency[root2].end());
              adjacency[root1].push_back(match_);
            } else {
              component_features[root2].insert(set1.begin(), set1.end());
              adjacency[root2].insert(adjacency[root2].end(), adjacency[root1].begin(), adjacency[root1].end());
              adjacency[root2].push_back(match_);
            }

            //feature track will be constructed here in mirror of the union find
          }
        }
        // Unite them in the Union-Find structure
        // uf_ptr->unite(idx1, idx2);
      }
    }
  }



  void FeatureTrackGenerator::launch_inprocess_sparse_CG_iterator(const std::vector<Observation *> &observations,
                                                                  int maxIters,
                                                                  float tol) {
    if (observations.empty()){return;}

    std::vector<BAImage *> activeImages; // free images only
    std::vector<BAImage *> touchedImages; // all images, for reset/check
    std::vector<BAFeature *> activeFeatures;
    std::vector<float> ftXY,imgXY;

    // ---- build local system ----

    for (auto *ob: observations) {
      BAImage *img = ob->image;

      if (img->touchIdx < 0) {
        // add this field, or use another temp marker
        img->touchIdx = touchedImages.size();
        touchedImages.push_back(img);
      }

      if (!img->fixed && img->systemIdx < 0) {
        img->systemIdx = activeImages.size();
        activeImages.push_back(img);
      }

      BAFeature *root = ob->feature->find();
      ob->feature = root; // IMPORTANT: always canonicalize

      if (root->systemIdx < 0) {
        root->systemIdx = activeFeatures.size();
        activeFeatures.push_back(root);
      }
    }

    {
      Poco::RWLock::ScopedWriteLock lock(rwLock);

      ftXY.resize(2 * activeFeatures.size());
      for (int i = 0; i <activeFeatures.size(); ++i) {
        ftXY[2 * i] = activeFeatures[i]->xy.x;
        ftXY[2 * i + 1] = activeFeatures[i]->xy.y;
      }

      imgXY.resize(activeImages.size() * 2);
      for (int i = 0; i < activeImages.size(); ++i) {
        imgXY[2 * i] = activeImages[i]->xy.x;
        imgXY[2 * i + 1] = activeImages[i]->xy.y;
      }
    } //end scoped write lock

    int fixedCount = 0;
    for (auto *img: touchedImages) {
      if (img->fixed) ++fixedCount;
    }
    if (fixedCount < 1) {
      std::cout<<"SPARSE CONJUGATE GRADIENT LAUNCHED WITH NO ANCHOR"<<std::endl;
      return;
    }

    const int nI = (int) activeImages.size(); // free images only
    const int nF = (int) activeFeatures.size();

    // ---- allocate ----

    auto ensureSize = [](std::vector<float> &v, int n, int pad) {
      if ((int) v.size() < n) v.resize(n + pad);
    };

    ensureSize(solverState.img_r_x, nI, 50);
    ensureSize(solverState.img_r_y, nI, 50);
    ensureSize(solverState.img_p_x, nI, 50);
    ensureSize(solverState.img_p_y, nI, 50);
    ensureSize(solverState.img_Ap_x, nI, 50);
    ensureSize(solverState.img_Ap_y, nI, 50);
    ensureSize(solverState.img_z_x, nI, 50);
    ensureSize(solverState.img_z_y, nI, 50);
    ensureSize(solverState.img_inv_diag, nI, 50);

    ensureSize(solverState.feat_r_x, nF, 5000);
    ensureSize(solverState.feat_r_y, nF, 5000);
    ensureSize(solverState.feat_p_x, nF, 5000);
    ensureSize(solverState.feat_p_y, nF, 5000);
    ensureSize(solverState.feat_Ap_x, nF, 5000);
    ensureSize(solverState.feat_Ap_y, nF, 5000);
    ensureSize(solverState.feat_z_x, nF, 5000);
    ensureSize(solverState.feat_z_y, nF, 5000);
    ensureSize(solverState.feat_inv_diag, nF, 5000);

    auto zeroN = [](std::vector<float> &v, int n) {
      std::fill(v.begin(), v.begin() + n, 0.0f);
    };

    zeroN(solverState.img_r_x, nI);
    zeroN(solverState.img_r_y, nI);
    zeroN(solverState.img_p_x, nI);
    zeroN(solverState.img_p_y, nI);
    zeroN(solverState.img_inv_diag, nI);

    zeroN(solverState.feat_r_x, nF);
    zeroN(solverState.feat_r_y, nF);
    zeroN(solverState.feat_p_x, nF);
    zeroN(solverState.feat_p_y, nF);
    zeroN(solverState.feat_inv_diag, nF);

    // ---- build RHS and diagonal preconditioner ----

    for (const auto *o: observations) {
      const int ii = o->image->systemIdx; // -1 if fixed
      const int fi = o->feature->systemIdx;

      BAImage *img = o->image;
      BAFeature *feat = o->feature;

      const float rx = feat->xy.x - img->xy.x - o->obs_x;
      const float ry = feat->xy.y - img->xy.y - o->obs_y;

      solverState.feat_r_x[fi] -= rx;
      solverState.feat_r_y[fi] -= ry;
      solverState.feat_inv_diag[fi] += 1;

      if (ii >= 0) {
        solverState.img_r_x[ii] += rx;
        solverState.img_r_y[ii] += ry;
        solverState.img_inv_diag[ii] += 1;
      }
    }

    constexpr float eps = 1e-8f;

    for (int i = 0; i < nI; ++i) {
      solverState.img_inv_diag[i] = 1.0f / (solverState.img_inv_diag[i] + eps);
    }

    for (int i = 0; i < nF; ++i) {
      solverState.feat_inv_diag[i] = 1.0f / (solverState.feat_inv_diag[i] + eps);
    }

    // ---- PCG init: z = M^-1 r, p = z ----

    float prev_rTz = 0.0f;

    for (int i = 0; i < nI; ++i) {
      solverState.img_z_x[i] = solverState.img_inv_diag[i] * solverState.img_r_x[i];
      solverState.img_z_y[i] = solverState.img_inv_diag[i] * solverState.img_r_y[i];

      solverState.img_p_x[i] = solverState.img_z_x[i];
      solverState.img_p_y[i] = solverState.img_z_y[i];

      prev_rTz += solverState.img_r_x[i] * solverState.img_z_x[i]
          + solverState.img_r_y[i] * solverState.img_z_y[i];
    }

    for (int i = 0; i < nF; ++i) {
      solverState.feat_z_x[i] = solverState.feat_inv_diag[i] * solverState.feat_r_x[i];
      solverState.feat_z_y[i] = solverState.feat_inv_diag[i] * solverState.feat_r_y[i];

      solverState.feat_p_x[i] = solverState.feat_z_x[i];
      solverState.feat_p_y[i] = solverState.feat_z_y[i];

      prev_rTz += solverState.feat_r_x[i] * solverState.feat_z_x[i]
          + solverState.feat_r_y[i] * solverState.feat_z_y[i];
    }

    int c = 0;

    // ---- convergence tracking ----

    const float initial_rTz = prev_rTz;

    float last_rel_improvement = std::numeric_limits<float>::infinity();
    int stagnantIters = 0;

    int runIters = maxIters == 0 ? 30 : maxIters;

    for (int iteration = 0; iteration < runIters; ++iteration) {
      ++c;

      // --------------------------------------------
      // early-out: invalid / converged residual
      // --------------------------------------------

      if (!std::isfinite(prev_rTz) || prev_rTz <= 1e-20f) {
        break;
      }

      zeroN(solverState.img_Ap_x, nI);
      zeroN(solverState.img_Ap_y, nI);
      zeroN(solverState.feat_Ap_x, nF);
      zeroN(solverState.feat_Ap_y, nF);

      // ---- Ap = A p ----

      for (const auto *o: observations) {
        const int ii = o->image->systemIdx;
        const int fi = o->feature->systemIdx;

        const float img_px = (ii >= 0) ? solverState.img_p_x[ii] : 0.0f;
        const float img_py = (ii >= 0) ? solverState.img_p_y[ii] : 0.0f;

        const float vx = solverState.feat_p_x[fi] - img_px;
        const float vy = solverState.feat_p_y[fi] - img_py;

        solverState.feat_Ap_x[fi] += vx;
        solverState.feat_Ap_y[fi] += vy;

        if (ii >= 0) {
          solverState.img_Ap_x[ii] -= vx;
          solverState.img_Ap_y[ii] -= vy;
        }
      }

      float pAp = 0.0f;

      for (int i = 0; i < nI; ++i) {
        pAp += solverState.img_p_x[i] * solverState.img_Ap_x[i]
            + solverState.img_p_y[i] * solverState.img_Ap_y[i];
      }

      for (int i = 0; i < nF; ++i) {
        pAp += solverState.feat_p_x[i] * solverState.feat_Ap_x[i]
            + solverState.feat_p_y[i] * solverState.feat_Ap_y[i];
      }

      // --------------------------------------------
      // early-out: degenerate denominator
      // --------------------------------------------

      if (!std::isfinite(pAp) || std::abs(pAp) < 1e-20f) {
        break;
      }

      const float alpha = prev_rTz / pAp;

      if (!std::isfinite(alpha)) {
        break;
      }

      // ---- update solution and residual ----

      for (int i = 0; i < nI; ++i) {
        imgXY[2 * i] += alpha * solverState.img_p_x[i];
        imgXY[2 * i + 1] += alpha * solverState.img_p_y[i];

        solverState.img_r_x[i] -= alpha * solverState.img_Ap_x[i];
        solverState.img_r_y[i] -= alpha * solverState.img_Ap_y[i];
      }

      for (int i = 0; i < nF; ++i) {
        ftXY[2 * i] += alpha * solverState.feat_p_x[i];
        ftXY[2 * i + 1] += alpha * solverState.feat_p_y[i];

        solverState.feat_r_x[i] -= alpha * solverState.feat_Ap_x[i];
        solverState.feat_r_y[i] -= alpha * solverState.feat_Ap_y[i];
      }

      // ---- z = M^-1 r, compute rTz and residual norm ----

      float new_rTz = 0.0f;
      float residualNorm2 = 0.0f;

      for (int i = 0; i < nI; ++i) {
        solverState.img_z_x[i] = solverState.img_inv_diag[i] * solverState.img_r_x[i];
        solverState.img_z_y[i] = solverState.img_inv_diag[i] * solverState.img_r_y[i];

        new_rTz += solverState.img_r_x[i] * solverState.img_z_x[i]
            + solverState.img_r_y[i] * solverState.img_z_y[i];

        residualNorm2 += solverState.img_r_x[i] * solverState.img_r_x[i]
            + solverState.img_r_y[i] * solverState.img_r_y[i];
      }

      for (int i = 0; i < nF; ++i) {
        solverState.feat_z_x[i] = solverState.feat_inv_diag[i] * solverState.feat_r_x[i];
        solverState.feat_z_y[i] = solverState.feat_inv_diag[i] * solverState.feat_r_y[i];

        new_rTz += solverState.feat_r_x[i] * solverState.feat_z_x[i]
            + solverState.feat_r_y[i] * solverState.feat_z_y[i];

        residualNorm2 += solverState.feat_r_x[i] * solverState.feat_r_x[i]
            + solverState.feat_r_y[i] * solverState.feat_r_y[i];
      }

      // --------------------------------------------
      // convergence checks
      // --------------------------------------------

      if (!std::isfinite(new_rTz) || !std::isfinite(residualNorm2)) {
        break;
      }

      const float residualNorm = std::sqrt(residualNorm2);

      // absolute residual
      if (residualNorm < tol && maxIters == 0) {
        break;
      }

      // relative preconditioned residual reduction
      const float relResidual = new_rTz / (initial_rTz + eps);

      if (relResidual < 1e-4f && maxIters == 0) {
        break;
      }

      // stagnation detection
      const float relImprovement = std::abs(prev_rTz - new_rTz) / (prev_rTz + eps);

      if (relImprovement < 1e-3f) {
        ++stagnantIters;
      } else {
        stagnantIters = 0;
      }

      // require several stagnant iterations in a row
      // to avoid premature exits
      if (stagnantIters >= 3 && maxIters == 0) {
        break;
      }

      last_rel_improvement = relImprovement;

      // ---- p = z + beta p ----

      const float beta = new_rTz / (prev_rTz + eps);

      if (!std::isfinite(beta)) {
        break;
      }

      for (int i = 0; i < nI; ++i) {
        solverState.img_p_x[i] =
            solverState.img_z_x[i] + beta * solverState.img_p_x[i];

        solverState.img_p_y[i] =
            solverState.img_z_y[i] + beta * solverState.img_p_y[i];
      }

      for (int i = 0; i < nF; ++i) {
        solverState.feat_p_x[i] =
            solverState.feat_z_x[i] + beta * solverState.feat_p_x[i];

        solverState.feat_p_y[i] =
            solverState.feat_z_y[i] + beta * solverState.feat_p_y[i];
      }

      prev_rTz = new_rTz;
    }
    // std::cout<<"iters "<<c<<std::endl;

    Poco::RWLock::ScopedWriteLock lock(rwLock);
    for (int i = 0; i < activeFeatures.size(); ++i) {
      auto ft = activeFeatures[i];
      ft->xy.x = ftXY[2 * i];
      ft->xy.y = ftXY[2 * i + 1];
      ft->systemIdx = -1;
      ft->lastIteration += c;
      featureGrid.update(ft);
    }

    for (int i = 0; i < activeImages.size(); ++i) {
      auto img = activeImages[i];
      img->xy.x = imgXY[2 * i];
      img->xy.y = imgXY[2 * i + 1];
      img->systemIdx = -1;
    }

    for (auto *img: touchedImages) {
      img->touchIdx = -1;
    }
  }


  // Generate tracks from current state
  std::vector<FeatureTrack> FeatureTrackGenerator::generateCurrentTracks(const std::vector<Image *> &images) {
    if (!uf_ptr || index_to_feature.empty()) {
      return {};
    }

    auto ans = createTracksFromConnections(images, *uf_ptr);
    return ans;
  }

  std::vector<FeatureTrack> FeatureTrackGenerator::createTracksFromConnections(
    const std::vector<Image *> &images,
    const UnionFind &uf) {
    invalidCount = 0;

    // Group features by their root in Union-Find
    std::unordered_map<int, std::vector<int> > root_to_features;

    for (int i = 0; i < index_to_feature.size(); i++) {
      int root = uf.find(i);
      root_to_features[root].push_back(i);
    }

    // Create image lookup for coordinates
    std::unordered_map<long, const Image *> image_lookup;
    for (auto img: images) {
      image_lookup[img->index] = img;
    }

    // Create tracks
    std::vector<FeatureTrack> tracks;
    int track_id = 0;

    for (const auto &[root, feature_indices]: root_to_features) {
      // Skip single-image tracks (not useful for bundle adjustment)
      if (feature_indices.size() < 2) continue;

      FeatureTrack track(track_id++);

      // Check for multiple observations in same image (should not happen with good matching)
      std::set<long> images_in_track;
      bool valid_track = true;

      for (int global_idx: feature_indices) {
        const auto &pair = index_to_feature[global_idx];

        //check if the image is already in the feature track
        if (images_in_track.count(pair.image_id)) {
          // Multiple features from same image in track - invalid
          valid_track = false;
          ++invalidCount;
          break;
        }
        images_in_track.insert(pair.image_id);

        // Get feature coordinates
        auto img_it = image_lookup.find(pair.image_id);
        bool cond1 = img_it != image_lookup.end();
        if (cond1 && pair.feature_id < img_it->second->keypointsImageSpace.size()) {
          const auto &pos = img_it->second->keypointsImageSpace[pair.feature_id];


          auto fo = FeatureObservation(int(pair.image_id), pair.feature_id,
                                       pos.pt.x, pos.pt.y, img_it->second);
          track.addObservation(fo);
          track.world_x += pos.pt.x + img_it->second->regInfo->absoluteCoords.x;
          track.world_y += pos.pt.y + img_it->second->regInfo->absoluteCoords.y;
        } else {
          int k = 0;
        }
      }

      if (valid_track && track.observations.size() >= 2) {
        //average out world location;
        track.world_x /= track.observations.size();
        track.world_y /= track.observations.size();

        tracks.push_back(track);
      }
    }

    return tracks;
  }

  bool coopers_conjugate_gradient(const Mat &A, Mat &x, const Mat &b, int maxIters = 1000, double tol = 1e-8) {
    Mat ATranspose = A.t();
    Mat ATA = ATranspose * A;
    Mat ATb = ATranspose * b;
    Mat r = ATb - (ATA * x);
    Mat p = r.clone();
    double n0 = cv::norm(A * x - b);

    double n1;
    Mat ATAp;

    int iters = 0;
    for (int i = 0; i < maxIters; ++i) {
      ++iters;

      double rdr = r.dot(r);
      ATAp = ATA * p;
      double stepSize = rdr / p.dot(ATAp);

      x += stepSize * p;
      n1 = cv::norm(A * x - b);

      if (abs(n1 - n0) < tol) {
        break;
      }
      n0 = n1;

      r -= stepSize * ATAp;
      double adjustment = r.dot(r) / rdr;
      p = r + adjustment * p;
    }
    std::cout << "conjugate gradient ran " << iters << " iterations" << std::endl;
    return iters < maxIters;
  }

  void BundleAdjustmentIntegrator::run_coopers_planar_bundle_adjustment(const std::vector<FeatureTrack> &_tracks,
                                                                        const std::vector<Image *> &_images) {
    int totalConstraints = 0;
    for (auto &t: _tracks) {
      totalConstraints += t.observations.size();
    }

    Mat A(totalConstraints, _images.size() - 1 + _tracks.size(),CV_64FC1, Scalar(0));
    Mat xx(_images.size() - 1 + _tracks.size(), 1,CV_64FC1, Scalar(0));
    Mat xy(_images.size() - 1 + _tracks.size(), 1,CV_64FC1, Scalar(0));
    Mat bx(totalConstraints, 1,CV_64FC1, Scalar(0));
    Mat by(totalConstraints, 1,CV_64FC1, Scalar(0));

    std::map<long, int> imageToSystem;
    std::map<int, Image *> systemToImage;

    {
      int rootCount = 0;
      int i = 0;
      for (auto &img: _images) {
        if (img->regInfo->root) {
          ++rootCount;
          continue;
        }
        imageToSystem[img->index] = i;
        systemToImage[i] = img;

        xx.at<double>(i, 0) = img->regInfo->absoluteCoords.x;
        xy.at<double>(i, 0) = img->regInfo->absoluteCoords.y;

        ++i;
      }
      assert(rootCount == 1);
    }

    int rowConstraintIdx = 0;
    for (int i = 0; i < _tracks.size(); ++i) {
      int landmarkCol = i + _images.size() - 1;

      xx.at<double>(landmarkCol) = _tracks[i].world_x;
      xy.at<double>(landmarkCol) = _tracks[i].world_y;

      for (auto &obs: _tracks[i].observations) {
        A.at<double>(rowConstraintIdx, landmarkCol) = 1;
        if (!obs.imgRef->regInfo->root) {
          A.at<double>(rowConstraintIdx, imageToSystem[obs.image_id]) = -1;
        }
        bx.at<double>(rowConstraintIdx) = obs.x;
        by.at<double>(rowConstraintIdx) = obs.y;
        ++rowConstraintIdx;
      }
    }

    coopers_conjugate_gradient(A, xx, bx);
    coopers_conjugate_gradient(A, xy, by);

    for (auto img: _images) {
      if (img->regInfo->root) { continue; }
      img->regInfo->absoluteCoords.x = xx.at<double>(imageToSystem[img->index]);
      img->regInfo->absoluteCoords.y = xy.at<double>(imageToSystem[img->index]);
    }
  }

  // CPU edge-list conjugate gradient solver for planar (no-rotation) "bundle adjustment"
  // Model per observation k (camera i observes landmark j):
  //   (L_j - C_i) = u_ij
  // Solve least squares:  min_x ||A x - b||^2  where x = [C (Nc), L (Nl)]  (scalar per axis)
  // We solve x and y separately with identical structure.
  //
  // Key points:
  // - No OpenCV, no dense matrices
  // - No explicit A, no At, no AtA
  // - CG is run on normal equations implicitly using edge list
  // - Work per CG iteration is O(M) where M=#observations
  // - Gauge: root camera is removed from variables (no entry) => treated as fixed at 0
  //
  // Typical sizes: Nc~299, Nl~20k, M~100k+
  // This should be orders of magnitude faster than dense cv::Mat.


  // ------------------------- Edge list -------------------------

  struct EdgeObs {
    int cam; // 0..Nc-1 (for non-root cameras only)
    int lm; // 0..Nl-1
    double u; // observed coordinate (x or y) in camera coords
    bool hasCam; // false means observation came from root camera (cam term omitted)
  };

  // ------------------------- Small vector ops -------------------------

  static inline double dot(const std::vector<double> &a, const std::vector<double> &b) {
    assert(a.size() == b.size());
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
  }

  static inline double norm2(const std::vector<double> &a) {
    return dot(a, a);
  }

  static inline void axpy(std::vector<double> &y, double alpha, const std::vector<double> &x) {
    assert(y.size() == x.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] += alpha * x[i];
  }

  static inline void xpay(std::vector<double> &y, const std::vector<double> &x, double beta) {
    // y = x + beta*y
    assert(y.size() == x.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] = x[i] + beta * y[i];
  }

  // ------------------------- Core operator: AtA * p -------------------------
  //
  // For each observation (i,j):
  //   row residual on vector v is: q = (v_L[j] - (hasCam ? v_C[i] : 0))
  // Then (AtA v) accumulates:
  //   out_L[j] += q
  //   if hasCam: out_C[i] += -q
  //
  static inline void apply_AtA(
    const std::vector<EdgeObs> &edges,
    int numCams, int numLms,
    const std::vector<double> &v, // size = numCams + numLms
    std::vector<double> &out) // size = numCams + numLms
  {
    const int N = numCams + numLms;
    assert((int)v.size() == N);
    out.assign(N, 0.0);

    // index helpers: cams [0..numCams-1], lms [numCams..numCams+numLms-1]
    for (const auto &e: edges) {
      const int camIdx = e.cam;
      const int lmIdx = numCams + e.lm;

      double q = v[lmIdx];
      if (e.hasCam) q -= v[camIdx];

      out[lmIdx] += q;
      if (e.hasCam) out[camIdx] -= q;
    }
  }

  // ------------------------- Core operator: At*(A x - b) -------------------------
  //
  // e = (x_L[j] - (hasCam ? x_C[i] : 0)) - u
  // Then gradient g = At e is:
  //   g_L[j] += e
  //   if hasCam: g_C[i] += -e
  //
  static inline void apply_At_residual(
    const std::vector<EdgeObs> &edges,
    int numCams, int numLms,
    const std::vector<double> &x, // size N
    std::vector<double> &g) // size N
  {
    const int N = numCams + numLms;
    assert((int)x.size() == N);
    g.assign(N, 0.0);

    for (const auto &e: edges) {
      const int camIdx = e.cam;
      const int lmIdx = numCams + e.lm;

      double pred = x[lmIdx];
      if (e.hasCam) pred -= x[camIdx];

      const double err = pred - e.u;

      g[lmIdx] += err;
      if (e.hasCam) g[camIdx] -= err;
    }
  }

  // Optional: compute RMS of Ax-b without forming Ax (for reporting only)
  static inline double rms_residual(
    const std::vector<EdgeObs> &edges,
    int numCams, int numLms,
    const std::vector<double> &x) {
    double sse = 0.0;
    for (const auto &e: edges) {
      const int camIdx = e.cam;
      const int lmIdx = numCams + e.lm;
      double pred = x[lmIdx];
      if (e.hasCam) pred -= x[camIdx];
      double r = pred - e.u;
      sse += r * r;
    }
    return edges.empty() ? 0.0 : std::sqrt(sse / (double) edges.size());
  }

  // ------------------------- Conjugate Gradient on normal equations -------------------------
  //
  // We solve: (AtA) x = Atb
  // but we do it without forming AtA or Atb:
  //   r = Atb - AtA x  = -At(Ax - b)
  // We compute g = At(Ax - b), then set r = -g.
  //
  // Stop criterion: sqrt(r·r) <= tolAbs OR relative reduction <= tolRel
  //
  bool coopers_cg_edge_list(
    const std::vector<EdgeObs> &edges,
    int &ranIters,
    int numCams, int numLms,
    std::vector<double> &x, // size N, in/out
    int maxIters = 200,
    double tolRel = 1e-8,
    double tolAbs = 1e-12,
    bool verbose = true) {
    const int N = numCams + numLms;
    assert((int)x.size() == N);

    std::vector<double> g, r, p, Ap;

    // r = -At(Ax - b)
    apply_At_residual(edges, numCams, numLms, x, g);
    r.resize(N);
    for (int i = 0; i < N; ++i) r[i] = -g[i];

    p = r;
    double rsold = norm2(r);
    const double rs0 = rsold;

    if (std::sqrt(rsold) <= tolAbs) {
      if (verbose) std::cout << "CG: already converged (abs)\n";
      return true;
    }

    int iters = 0;
    for (; iters < maxIters; ++iters) {
      apply_AtA(edges, numCams, numLms, p, Ap);

      const double denom = dot(p, Ap);
      if (std::abs(denom) < 1e-30) {
        if (verbose) std::cout << "CG: breakdown (singular / bad gauge)\n";
        return false;
      }

      const double alpha = rsold / denom;

      // x += alpha p
      axpy(x, alpha, p);

      // r -= alpha Ap
      axpy(r, -alpha, Ap);

      const double rsnew = norm2(r);

      // stopping
      const double rel = (rs0 > 0.0) ? std::sqrt(rsnew / rs0) : std::sqrt(rsnew);
      if (std::sqrt(rsnew) <= tolAbs || rel <= tolRel) {
        ++iters; // count this iteration
        break;
      }

      const double beta = rsnew / rsold;

      // p = r + beta p
      for (int i = 0; i < N; ++i) p[i] = r[i] + beta * p[i];

      rsold = rsnew;
    }

    if (verbose) {
      std::cout << "CG ran " << iters << " iterations"
          << ", rel_res=" << ((rs0 > 0.0) ? std::sqrt(rsold / rs0) : std::sqrt(rsold))
          << ", rms=" << rms_residual(edges, numCams, numLms, x)
          << "\n";
    }

    ranIters = iters;
    return iters < maxIters;
  }

  // ------------------------- Building the edge list -------------------------
  //
  // You’ll adapt these to your types; here’s the shape:
  //
  // - exactly one root image is fixed => not included in variables
  // - cameras are remapped to 0..Nc-1 excluding root
  // - landmarks are 0..Nl-1 in track vector order (your track_id is already dense)
  //
  // The resulting unknown vector per axis is:
  //   x[0..Nc-1]          = camera coordinate (non-root cams)
  //   x[Nc..Nc+Nl-1]      = landmark coordinate
  //
  // You run twice: one edge list with u=obs.x, and another with u=obs.y
  //

  // Forward declare your types (replace with your actual includes)

  struct PlanarBAResult {
    int numCams = 0;
    int numLms = 0;
    std::vector<double> x_x; // solution vector for x axis, size Nc+Nl
    std::vector<double> x_y; // solution vector for y axis, size Nc+Nl
  };

  std::pair<int, int> BundleAdjustmentIntegrator::run_coopers_planar_ba_edge_list(
    const std::vector<FeatureTrack> &tracks,
    std::vector<Image *> &images,
    int maxIters,
    double tolRel) {
    // Map images (excluding root) -> camera variable index
    int rootCount = 0;
    std::unordered_map<long, int> imageToCam;
    imageToCam.reserve(images.size() * 2);

    int camIdx = 0;
    for (auto *img: images) {
      if (img->regInfo->root) {
        ++rootCount;
        continue;
      }
      imageToCam[img->index] = camIdx++;
    }


    assert(rootCount == 1);

    const int Nc = camIdx;
    const int Nl = (int) tracks.size();
    const int N = Nc + Nl;

    // Build edges for x and y
    size_t totalObs = 0;
    for (auto &t: tracks) totalObs += t.observations.size();

    std::vector<EdgeObs> edgesX;
    std::vector<EdgeObs> edgesY;
    edgesX.reserve(totalObs);
    edgesY.reserve(totalObs);

    for (int lm = 0; lm < Nl; ++lm) {
      for (const auto &obs: tracks[lm].observations) {
        EdgeObs ex;
        ex.lm = lm;
        ex.u = obs.x;

        if (obs.imgRef->regInfo->root) {
          ex.hasCam = false;
          ex.cam = 0; // unused
        } else {
          ex.hasCam = true;
          auto it = imageToCam.find(obs.image_id);
          assert(it != imageToCam.end());
          ex.cam = it->second;
        }
        edgesX.push_back(ex);

        EdgeObs ey = ex;
        ey.u = obs.y;
        edgesY.push_back(ey);
      }
    }

    // Initial guess from existing regInfo + track world coords
    std::vector<double> x0x(N, 0.0), x0y(N, 0.0);

    // cameras
    for (auto *img: images) {
      if (img->regInfo->root) continue;
      int ci = imageToCam.at(img->index);
      x0x[ci] = img->regInfo->absoluteCoords.x;
      x0y[ci] = img->regInfo->absoluteCoords.y;
    }
    // landmarks
    for (int lm = 0; lm < Nl; ++lm) {
      x0x[Nc + lm] = tracks[lm].world_x;
      x0y[Nc + lm] = tracks[lm].world_y;
    }

    // Solve
    int xIters = 0, yIters = 0;
    bool okx = coopers_cg_edge_list(edgesX, xIters, Nc, Nl, x0x, maxIters, tolRel, 1e-12, false);
    bool oky = coopers_cg_edge_list(edgesY, yIters, Nc, Nl, x0y, maxIters, tolRel, 1e-12, false);
    (void) okx;
    (void) oky;

    for (auto *img: images) {
      if (img->regInfo->root) continue;
      int ci = imageToCam.at(img->index);
      img->regInfo->absoluteCoords.x = x0x[ci];
      img->regInfo->absoluteCoords.y = x0y[ci];
    }
    return {xIters, yIters};
  }


  int UnionFind::find(int x) const {
    if (parent[x] != x) {
      parent[x] = find(parent[x]); // path compression
    }
    return parent[x];
  }

  void UnionFind::expand(int new_size) {
    if (new_size <= parent.size()) return;

    int old_size = parent.size();
    parent.resize(new_size);
    rank.resize(new_size, 0);

    // Initialize new elements
    for (int i = old_size; i < new_size; i++) {
      parent[i] = i;
    }
  }

  void UnionFind::unite(int x, int y) {
    int px = find(x);
    int py = find(y);

    if (px == py) return;

    if (rank[px] < rank[py]) {
      parent[px] = py;
    } else if (rank[px] > rank[py]) {
      parent[py] = px;
    } else {
      parent[py] = px;
      rank[px]++;
    }
  }

  // After you get PlanarBAResult, write back camera coords:
  //   img->regInfo->absoluteCoords.x = res.x_x[camIdx]
  // landmarks are in res.x_x[Nc + lm]


  /*
    void BundleAdjustmentIntegrator::run_bundle_adjustment(const std::vector<FeatureTrack> &_tracks,
                                                           const std::vector<Image *> &_images) {
      optimizer->clear();
      //camera poses represent absolute coordinates of images
      int stayFixedCount = 0;
      for (const auto &img: _images) {
        cuba::CameraParams camParams;

        //10000 is for numerical stability.
        camParams.fx = 10000;
        camParams.fy = 10000;

        //this is essentially "where the camera sits relative to the image it took" ie the middle (generally)
        //but for simplicity we say the camera was at the image origin (upper left corner)
        camParams.cx = 0; //parent->image_width / 2;
        camParams.cy = 0; //parent->image_height / 2;

        //bf only used for stereo photos - not relevant. this is actually set in the constructor as well.
        camParams.bf = 0;

        //images have no rotation
        auto AbC = img->regInfo->absoluteCoords;

        auto camRotation = Eigen::Quaterniond::Identity();
        cuba::Array<double, 3> translation(-AbC.x,-AbC.y,10000);// * scale);

        bool fixed = img->regInfo->rootOfRoot || img->regInfo->stayFixedDuringBundleAdjustment;
        if (fixed){++stayFixedCount;}

        auto poseVertex = new cuba::PoseVertex(img->index, camRotation, translation, camParams, fixed);

        //add it to the optimizer
        optimizer->addPoseVertex(poseVertex);

        //keep possession of it
        poseVertices[img->index] = poseVertex;
      }
      assert(stayFixedCount == 1);


      //landmark vertexes are feature points placed in world/composite pixel coordinates
      for (const auto &track: _tracks) {
        cuba::Array<double, 3> featurePositionInComposite = {track.world_x, track.world_y, 0};

        auto landmarkVertex = new cuba::LandmarkVertex(track.track_id, featurePositionInComposite, false);

        optimizer->addLandmarkVertex(landmarkVertex);

        landmarkVertices[track.track_id] = landmarkVertex;

        //add edges (observations) for this track
        for (const auto &obs: track.observations) {
          //get the pose (image) associated with the obervation
          auto poseVertex = optimizer->poseVertex(obs.image_id);

          cuba::Array<double, 2> landmarkPositionInFrame = {obs.x, obs.y};

          auto edge = new cuba::MonoEdge(landmarkPositionInFrame, 1.0, poseVertex, landmarkVertex);

          optimizer->addMonocularEdge(edge);

          monoEdges.push_back(edge);
        }
      }
      constexpr auto robustKernelType = cuba::RobustKernelType::HUBER;
      const double deltaMono = sqrt(5.9);

      //optimizer->setRobustKernels(robustKernelType, deltaMono, cuba::EdgeType::MONOCULAR);

      optimizer->initialize();
      optimizer->setPoseUpdateAllowance(true, true);

      optimizer->optimize(100);
    }
    */
}
