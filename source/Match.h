//
//  Match.hpp
//  pathCam
//
//  Created by Brian Summa on 10/6/22.
//

#ifndef Match_h
#define Match_h

#include "pathCam.h"

namespace pathCam{

class Match{
public:
  Image * image_1;
  Image * image_2;
  
  //not used right now
  double quality;
  
  std::vector<DMatch> good_matches;

  cv::Mat H;
  double t_x, t_y;
  
  Match(Image * image_1, Image * image_2): image_1(image_1),
                                           image_2(image_2),
                                           quality(0.0),
                                           t_x(0.0), t_y(0.0) {};
  ~Match(){};
  
  
  
};



}



#endif /* Match_hpp */
