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


class SiftFeatureMatcher : public PostProcessorBase {
  public:
  SiftFeatureMatcher(StreamCam *_parent);
  ~SiftFeatureMatcher(){};

  virtual void run();

  virtual bool isTerminal();

  int device;
};



}
#endif //POSTPROCESSOR_H
