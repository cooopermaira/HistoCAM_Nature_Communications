//
//  CaptureComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"


class sCamPocoRunnable : public Poco::Runnable {
public:
  sCamPocoRunnable(CaptureComponent *cptcmp) : cptcmp(cptcmp) {};
  CaptureComponent *cptcmp;

  virtual void run() {
    cptcmp->sCam->run();
  }
};

class bcamPocoRunnable : public Poco::Runnable {
public:
  bcamPocoRunnable(CaptureComponent *cptcmp) : cptcmp(cptcmp) {};
  CaptureComponent *cptcmp;

  virtual void run() {
#ifdef WITH_SPINNAKER
    cptcmp->bcam->run();
#endif
  }
};

CaptureComponent::CaptureComponent(std::shared_ptr<fRectangle> view,
                                   StringArray &iconNames,
                                   OwnedArray<Drawable> &iconsFromZipFile, Poco::Util::LayeredConfiguration::Ptr config,
                                   MainComponent *parent) : config(config), parent(parent),
                                                            ImageViewComponent(view, iconNames, iconsFromZipFile),
                                                            recording(false), simulating(false) {

#ifdef WITH_SPINNAKER
  bcam.reset(new pathCam::SpinPath(config));
  bcam->add_observer(parent);
  
#endif

  sCam.reset(new pathCam::StreamCam(config));
  sCam->add_observer(parent);
  

  captureOverlay.reset(new CaptureOverlay(this, iconNames, iconsFromZipFile));
  addAndMakeVisible(captureOverlay.get());
}


void CaptureComponent::startRecording() {
  recording = true;
#ifdef WITH_SPINNAKER
  parent->MRimage = bcam->get_image_reference();
#endif

  //bcam->run();
  parent->imageview->setImage(parent->MRimage);
  parent->capture->setImage(parent->MRimage);
  parent->annotate->setImage(parent->MRimage);


  auto r = new bcamPocoRunnable(this);
  compositeThread.start(r);

  repaint();
}

void CaptureComponent::startSimulating() {
  simulating = true;
  parent->MRimage = sCam->get_image_reference();
  parent->imageview->setImage(parent->MRimage);
  parent->capture->setImage(parent->MRimage);
  parent->annotate->setImage(parent->MRimage);


  auto r = new sCamPocoRunnable(this);
  compositeThread.start(r);
  repaint();
}


void CaptureComponent::stop() {
  if (recording) { stopRecording(); }
  if (simulating) { stopSimulating(); }
}

void CaptureComponent::stopRecording() {
#ifdef WITH_SPINNAKER
  bcam->stopCamera();
#endif

  recording = false;
  repaint();
}

void CaptureComponent::stopSimulating() {
  simulating = false;
  //TODO
  repaint();
}

