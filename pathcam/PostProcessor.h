//
// Created by cooper maira on 6/3/25.
//

#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include "pathCam.h"

namespace pathCam {

  class PostProcessorBase {
  public:
    PostProcessorBase(pathCam::StreamCam* _parent):parent(_parent){};

     ~PostProcessorBase(){};

    virtual void run() = 0;

    virtual bool isTerminal() = 0;

    StreamCam *parent;
  };

struct pMatch;

class SiftFeatureMatcher : public PostProcessorBase {
  public:
  SiftFeatureMatcher(StreamCam *_parent);
  ~SiftFeatureMatcher(){};

  void run() override;

  bool isTerminal() override;

  std::vector<Image*> imagesProcessed;

  std::vector<pMatch> allMatches;
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
