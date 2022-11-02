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
  
  cv::DescriptorMatcher::MatcherType matcher_type;
  Ptr<cv::DescriptorMatcher> matcher;
  
  float ratio_thresh = 0.75f;

public:
    
  DescriptorMatcher(cv::DescriptorMatcher::MatcherType matcher_type, float ratio_thresh = 0.75f): matcher_type(matcher_type), ratio_thresh(ratio_thresh){
    matcher = cv::DescriptorMatcher::create(matcher_type);
  };
  
  ~DescriptorMatcher(){};

  
  
  void match(Match *match);
  
  
  
  
  
  
};


}

#endif /* DescriptorMatcher_hpp */
