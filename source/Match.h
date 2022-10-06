//
//  Match.hpp
//  pathCam
//
//  Created by Brian Summa on 10/6/22.
//

#ifndef Match_h
#define Match_h

#include "common.h"

namespace pathCam{

class Match{
public:
  Image * image_1;
  Image * image_2;
  
  std::vector<DMatch> good_matches;

  cv::Mat H;
  double t_x, t_y;
  
  Match(Image * image_1, Image * image_2): image_1(image_1), image_2(image_2){};
  ~Match(){};
  
  
  
};



}



#endif /* Match_hpp */
