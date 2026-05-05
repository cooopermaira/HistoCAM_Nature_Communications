//
//  DescriptorMatcher.hpp
//  pathCam
//
//  Created by Brian Summa on 10/6/22.
//

#ifndef DescriptorMatcher_h
#define DescriptorMatcher_h

#include <stdio.h>

namespace pathCam{

class DescriptorMatcher{
public:

  cv::DescriptorMatcher::MatcherType matcher_type;
  Ptr<cv::DescriptorMatcher> matcher,matcher2;
  float ratio_thresh = 0.75f;


  DescriptorMatcher(cv::DescriptorMatcher::MatcherType matcher_type, float ratio_thresh = 0.75f);
  
  ~DescriptorMatcher()= default;


  void match(const std::shared_ptr<Match> &match) const;

};


}

#endif /* DescriptorMatcher_hpp */
