//
//  util.h
//  pathCam
//
//  Created by Brian on 3/16/23.
//

#ifndef util_h
#define util_h

namespace pathCam{

class Vec2{
public:
  double x, y;
  Vec2(double x, double y): x(x), y(y){};
};

class Bbox{
public:
  double min_x, min_y, max_x, max_y;
  Bbox(double min_x=std::numeric_limits<double>::infinity(),
       double min_y=std::numeric_limits<double>::infinity(),
       double max_x=-std::numeric_limits<double>::infinity(),
       double max_y=-std::numeric_limits<double>::infinity()):
  min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y) {};
};

}

#endif /* util_h */
