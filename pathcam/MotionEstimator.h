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

  static int findHomography(std::shared_ptr<Match> m, int estimator_type, int requiredGoodMatches,
                            double ransacReprojThreshold = 3, int maxIters = 2000,
                            double confidence = 0.995);
  
  void phaseCorrelate(pathCam::Match *m, Image *image_1, Image *image_2);
  
  
};


class RegInfo{
public:
  inline static int featureQuorum = 100;
  inline static int minimumVote = 50;
  inline static Poco::RWLock registrationProcessHalt;
  inline static std::atomic<bool> flag = false;

  struct vote {
    int componentIndex;
    Point2i abc;
    std::shared_ptr<Match> m;
  };

  StreamCam* parent;
  Image* image;
  long index, matchedTo, matchedBy;
  bool matchSearchComplete = false;
  bool successful,root,resolved,rootOfRoot;
  bool stayFixedDuringBundleAdjustment = false;
  bool wasAligned = false;
  bool tryComposite = false;
  bool inCompositeQ = false;

  // bool registered = false;
  bool queued = false;
  bool staged = false;

  int component_membership;
  Point2i absoluteCoords;
  Point2i relativeCoords = Point2i(0.0, 0.0);
  Poco::FastMutex accessMutex;
  Poco::Mutex rAccessMutex;
  Poco::Event waitOnResolve;
  std::vector<RegInfo*> callersWaiting, children;
  std::vector<std::pair<unsigned int, Match*>> componentCallersWaiting;
  std::vector<std::pair<Point2f,double>> rootHomographies;
  std::vector<vote> votes;
  vote winningVote;
  std::vector<std::shared_ptr<Match>> pollers;

  int bestMatch;
  int numBestMatches = 0;
  std::atomic<int> outstandingPolls = 0;
  
  RegInfo(StreamCam* parent, bool successful=false, Point2f absoluteCoords=Point2f(0.0, 0.0),bool root = false, int component_membership = 0):
  successful(successful), resolved(false), absoluteCoords(absoluteCoords),component_membership(component_membership),root(root), parent(parent),matchedBy(-1),matchedTo(-1),index(-1),
  waitOnResolve(true),rootOfRoot(false) {
  };

  void attempt_absolute_reg(bool queue_for_compositing);

  bool get_abc(RegInfo* caller, Point2i &_absoluteCoords, int &_componentMembership);

  bool poll_abc(std::shared_ptr<Match> m, Point2i &_absoluteCoords, int &_componentMembership);

  void set_abc(Point2f _absoluteCoords, int _componentMembership, bool queue_for_compositing);

  void vote_abc(std::shared_ptr<Match> m, Point2i _absoluteCoords, int _componentMembership);

  void count_votes();

  void cast_requested_votes();

  void set_waiting_component(unsigned int componentIndex, Match* m);

  void set_AbC_local_from_relative(unsigned int _relativeComponentSpace,Point2f _AbCInRelativeSpace);

  Point2f get_AbC_relative_from_local(unsigned int _relativeComponentSpace);

  void average_from_homographies(Point2f &_rootGuess, double &_scale);
  
  // to allow for sorting of reginfo objects by component membership
  bool operator < (const RegInfo& other) const {
    return component_membership < other.component_membership;
  }
  
  // std::string toString(){
  //   std::stringstream ss;
  //   ss << ((successful) ? "good" : "bad") << "\t";
  //   ss << absoluteCoords.toString();
  //   return ss.str();
  // }
  
};

}

#endif /* MotionEstimator_h */
