//
// Created by cooper maira on 6/3/25.
//

#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include "pathCam.h"

namespace pathCam {

  class PostProcessorBase {
  public:
    PostProcessorBase(StreamCam* _parent):parent(_parent){};

     ~PostProcessorBase(){};

    virtual void run() = 0;

    virtual bool isTerminal() = 0;

    virtual void init() = 0;

    StreamCam *parent;
  };

struct pMatch;

  class FeatureTrackGenerator;
  class BundleAdjustmentIntegrator;
class SiftFeatureMatcher : public PostProcessorBase {
  public:
  SiftFeatureMatcher(StreamCam *_parent);
  //~SiftFeatureMatcher(){};

  void run() override;

  bool isTerminal() override;

  void init() override;

  void postMatchProcessLoop();

  bool tracksReady();

  Poco::FastMutex* queueMutex;
  Poco::FastMutex* loopMutex;

  std::queue<std::pair<Image*,Image*>> postMatchQueue;

  std::vector<Image*> imagesProcessed;

  std::vector<pMatch> allMatches;

  FeatureTrackGenerator *ftg;
  BundleAdjustmentIntegrator *bai;

  std::atomic<bool> loopInProcess;
  std::atomic<int> matchWorkOutstanding;

  int numMatchesProcessed = 0;

  pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(
    parent->matcher_type);
  pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();
};

  // class Inferencer : public PostProcessorBase {
  //   public:
  //   Inferencer(StreamCam *_parent);
  //   ~Inferencer(){};
  //
  //   void run() override;
  //
  //   bool isTerminal() override;
  // };



}
#endif //POSTPROCESSOR_H
