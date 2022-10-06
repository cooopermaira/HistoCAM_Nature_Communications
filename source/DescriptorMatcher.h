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

public:
    
  DescriptorMatcher(cv::DescriptorMatcher::MatcherType matcher_type): matcher_type(matcher_type){
    matcher = cv::DescriptorMatcher::create(matcher_type);
  };
  
  ~DescriptorMatcher(){};

  
  
  void match(Match *match);
  
  
  
  
  
  
};


}

#endif /* DescriptorMatcher_hpp */
