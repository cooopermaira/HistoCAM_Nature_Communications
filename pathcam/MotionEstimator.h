//
//  MotionEstimator.h
//  pathCam
//
//  Created by Brian Summa on 11/1/22.
//

#ifndef MotionEstimator_h
#define MotionEstimator_h

#include "pathCam.h"

namespace pathCam{
class StreamCam;

class MotionEstimator{
public:
  
  MotionEstimator(){};
  
  int findHomography(pathCam::Match *m, int estimator_type, int requiredGoodMatches, int flag,
                     double ransacReprojThreshold = 3, int maxIters = 2000,
                     double confidence = 0.995);
  
  void phaseCorrelate(pathCam::Match *m, Image *image_1, Image *image_2);
  
  
};


class RegInfo{
public:
  StreamCam* parent;
  unsigned long int index,matchedTo;
  bool successful,root,resolved;
  unsigned int component_membership;
  Vec2 absoluteCoords;
  Vec2 relativeCoords = Vec2(0.0, 0.0);
  Poco::FastMutex *accessMutex;
  std::vector<RegInfo*> callersWaiting;
  std::vector<std::pair<unsigned int, Match*>> componentCallersWaiting;
  
  RegInfo(StreamCam* parent, bool successful=false, Vec2 absoluteCoords=Vec2(0.0, 0.0),bool root = false,unsigned int component_membership = 0):
  successful(successful), resolved(false), absoluteCoords(absoluteCoords),component_membership(component_membership),root(root),accessMutex(new Poco::FastMutex), parent(parent) {
  };

  bool get_abc(RegInfo* caller, Vec2& _absoluteCoords, unsigned int& _componentMembership);

  void set_abc(Vec2 _absoluteCoords, unsigned int _componentMembership);

  void set_waiting_component(unsigned int componentIndex, Match* m);
  
  // to allow for sorting of reginfo objects by component membership
  bool operator < (const RegInfo& other) const {
    return component_membership < other.component_membership;
  }
  
  std::string toString(){
    std::stringstream ss;
    ss << ((successful) ? "good" : "bad") << "\t";
    ss << absoluteCoords.toString();
    return ss.str();
  }
  
};

}

#endif /* MotionEstimator_h */
