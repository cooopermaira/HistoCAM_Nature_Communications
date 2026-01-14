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
    //From OpenCV tutorial
    std::vector<std::vector<DMatch> > knn_matches;

    matcher->knnMatch(match->image_1->descriptors, match->image_2->descriptors, knn_matches, 2);


    //-- Filter matches using the Lowe's ratio test
    for (size_t i = 0; i < knn_matches.size(); i++) {
      if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
        match->good_matches.push_back(knn_matches[i][0]);
      }
    }
  }
}
