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
  
  std::vector<DMatch> good_matches;

  //image_2 from image_1
  cv::Mat H;
  double t_x, t_y;
  
  Match(Image * image_1, Image * image_2): image_1(image_1),
                                           image_2(image_2),
                                           t_x(0.0), t_y(0.0) {};
  ~Match(){};
  
};


class MatchMatrix{
public:
  std::vector < std:: vector < Match * > > match;

  MatchMatrix(){};
  
  void resize(unsigned int size=0){
    match.resize(size);
    for(unsigned int i=0; i < size; i++){
      match[i].resize(size, NULL);
    }
  };
  
  ~MatchMatrix(){
    for(unsigned int i=0; i < match.size(); i++){
      for(unsigned int j=0; j < match[i].size(); j++){
        if(match[i][j] != NULL){ delete match[i][j];}
      }
      match[i].clear();
    }
    match.clear();
  };
  
};

}



#endif /* Match_hpp */
