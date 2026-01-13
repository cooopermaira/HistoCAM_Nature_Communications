//
//  CaptureComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include <memory>

#include "JuceHeader.h"

class drawThreadRunnable : public Poco::Runnable {
public:
  drawThreadRunnable(MainComponent *parent, Poco::Thread &sCamThread) : parent(parent), sCamThread(sCamThread) {
  };

  virtual void run() {
    while (sCamThread.isRunning()) {
      parent->update();
      Poco::Thread::sleep(100);
    }
    int k = 0;
  }

private:
  MainComponent *parent;
  Poco::Thread &sCamThread;
};

class sCamPocoRunnable : public Poco::Runnable {
public:
  sCamPocoRunnable(CaptureComponent *cptcmp) : cptcmp(cptcmp) {
  };
  CaptureComponent *cptcmp;

  virtual void run() {
    cptcmp->sCam->run();
    //cptcmp->stopSimulating();
  }
};

class bcamPocoRunnable : public Poco::Runnable {
public:
  bcamPocoRunnable(CaptureComponent *cptcmp) : cptcmp(cptcmp) {
  };
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
                                                            ImageViewComponent(
                                                              view, iconNames, iconsFromZipFile, parent),
                                                            recording(false), simulating(false) {
  captureOverlay.reset(new CaptureOverlay(this, iconNames, iconsFromZipFile));
  addAndMakeVisible(captureOverlay.get());
  aiOverlay.reset(new AIOverlay(this, iconNames, iconsFromZipFile));
  addAndMakeVisible(aiOverlay.get());
  // reportOverlay.reset(new ReportOverlay(this, iconNames, iconsFromZipFile));
  // addAndMakeVisible(reportOverlay.get());
}

void CaptureComponent::drawSlide(juce::Graphics &g, float scale) {
  ImageViewComponent::drawSlide(g, scale);

  if (view->isEmpty() || MRImageSet->empty()) {
    return;
  }

  cv::Rect_<float> frameBox;
  bool showAsCircle;
  std::string magLabel;

#ifdef WITH_SPINNAKER
  if (recording) {
    int ignore;
    bcam->sCam->get_last_frame(frameBox, showAsCircle, ignore, magLabel);
  }
#endif

  if (simulating) {
    int ignore;
    sCam->get_last_frame(frameBox, showAsCircle, ignore, magLabel);
  }

  if (!simulating && !recording) { return; }

  auto bounds = RectCtoJ<float>(frameBox);
  bounds.setPosition(bounds.getPosition() - view->getPosition());

  bounds *= view2screenScale(*view) * scale;

  g.setColour(juce::Colours::red);

  if (showAsCircle) {
    auto center = bounds.getCentre();
    fPoint radius = scopeRadius * view2screenScale(*view) * scale;;
    center -= radius;
    g.drawEllipse(center.getX(), center.getY(), 2 * radius.getX(), 2 * radius.getY(), 3);
  } else {
    g.drawRect(bounds, 3);
  }

  //print objective level
  g.setColour(juce::Colours::red);

  g.setFont(15);
  g.drawText("current objective", 5, getHeight() - 30, 110, Justification::centredLeft, true);

  g.setFont(40.0);
  g.drawText(magLabel, 20, getHeight() - 50, 100, Justification::centredLeft, true);
}

void CaptureComponent::startRecording() {
  recording = true;
#ifdef WITH_SPINNAKER
  if (!bcam) {
    bcam.reset(new pathCam::SpinPath(config));
    bcam->add_observer(parent);
    scopeRadius = bcam->sCam->get_scope_radius();
    parent->MRimage = bcam->get_image_reference();
    parent->sCam = bcam->sCam;
  }
  //aiOverlay->set_sCam(bcam->sCam);
#endif

  //bcam->run();
  parent->imageview->setImage(parent->MRimage);
  parent->capture->setImage(parent->MRimage);
  parent->annotate->setImage(parent->MRimage);

  compositeThread.start(new bcamPocoRunnable(this));
  updateDrawThread.start(new drawThreadRunnable(parent, compositeThread));

  repaint();
}

void CaptureComponent::startSimulating() {
  simulating = true;

  if (!sCam) {
    sCam = std::make_shared<pathCam::StreamCam>(config);
    sCam->add_observer(parent);
    scopeRadius = sCam->get_scope_radius();
    aiOverlay->set_sCam(sCam);
    parent->sCam = sCam.get();
  }

  parent->MRimage = sCam->get_image_reference();
  parent->imageview->setImage(parent->MRimage);
  parent->capture->setImage(parent->MRimage);
  parent->annotate->setImage(parent->MRimage);

  // std::thread t([this]() {
  //   sCam->run();
  // });
  // t.detach();
  compositeThread.start(new sCamPocoRunnable(this));
  updateDrawThread.start(new drawThreadRunnable(parent, compositeThread));

  aiOverlay->resized();
  //reportOverlay->resized();
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
  compositeThread.join();
  //TODO
  repaint();
}
