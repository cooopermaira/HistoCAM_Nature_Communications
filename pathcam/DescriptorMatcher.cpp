//
//  DescriptorMatcher.cpp
//  pathCam
//
//  Created by Brian Summa on 10/6/22.
//

#include "pathCam.h"

namespace pathCam {
  DescriptorMatcher::DescriptorMatcher(cv::DescriptorMatcher::MatcherType matcher_type,
                                       float ratio_thresh) : matcher_type(matcher_type), ratio_thresh(ratio_thresh) {
    matcher = cv::DescriptorMatcher::create(matcher_type);
  }



void DescriptorMatcher::match(const std::shared_ptr<Match> &match) const {
  std::vector<std::vector<DMatch>> knn_matches_12;

  // A -> B
  matcher->knnMatch(match->image_1->descriptors,
                    match->image_2->descriptors,
                    knn_matches_12, 2);

  // Apply ratio test immediately
  std::vector<DMatch> good_12;
  good_12.reserve(knn_matches_12.size());

  // Track which descriptors in image_2 we actually need to check
  std::unordered_set<int> needed_train_indices;

  for (size_t i = 0; i < knn_matches_12.size(); i++) {
    if (knn_matches_12[i].size() < 2) continue;

    const auto& m0 = knn_matches_12[i][0];
    const auto& m1 = knn_matches_12[i][1];

    if (m0.distance < ratio_thresh * m1.distance) {
      good_12.push_back(m0);
      needed_train_indices.insert(m0.trainIdx);
    }
  }

  // Early out
  if (good_12.empty()) {
    match->good_matches.clear();
    return;
  }

  // Build a compact descriptor matrix for only needed descriptors in image_2
  std::vector<int> index_map; // maps compact index -> original index
  index_map.reserve(needed_train_indices.size());

  cv::Mat filtered_desc2;
  for (int idx : needed_train_indices) {
    filtered_desc2.push_back(match->image_2->descriptors.row(idx));
    index_map.push_back(idx);
  }

  // B' -> A (only filtered descriptors)
  std::vector<std::vector<DMatch>> knn_matches_21;
  matcher->knnMatch(filtered_desc2,
                    match->image_1->descriptors,
                    knn_matches_21, 2);

  // Build reverse lookup
  std::unordered_map<int, int> reverse_best;
  reverse_best.reserve(knn_matches_21.size());

  for (size_t i = 0; i < knn_matches_21.size(); i++) {
    if (knn_matches_21[i].size() < 2) continue;

    const auto& m0 = knn_matches_21[i][0];
    const auto& m1 = knn_matches_21[i][1];

    if (m0.distance < ratio_thresh * m1.distance) {
      int original_query_idx = index_map[i]; // map back to original img2 index
      reverse_best[original_query_idx] = m0.trainIdx;
    }
  }

  // Mutual check
  match->good_matches.clear();
  match->good_matches.reserve(good_12.size());

  for (const auto& m : good_12) {
    auto it = reverse_best.find(m.trainIdx);
    if (it != reverse_best.end() && it->second == m.queryIdx) {
      match->good_matches.push_back(m);
    }
  }
}
}
