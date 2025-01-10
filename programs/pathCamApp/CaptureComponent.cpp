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


  captureOverlay.reset(new CaptureOverlay(this, iconNames, iconsFromZipFile));
  addAndMakeVisible(captureOverlay.get());
  aiOverlay.reset(new AIOverlay(this, iconNames, iconsFromZipFile));
  addAndMakeVisible(aiOverlay.get());

                                                          
}

void CaptureComponent::resized() {

    ImageViewComponent::resized();

    {
        const ScopedLock lock(mutex);
        juce::Rectangle<int> b = getLocalBounds();
        int width = 300;
        captureOverlay->setBounds(juce::Rectangle<int>(b.getWidth() - width - 20, 20, width, 60));
        aiOverlay->setBounds(juce::Rectangle<int>(b.getWidth() - 100 - 20,
                                                  b.getHeight() - 100 - 20, 100, 100));

    }

}

void CaptureComponent::drawSlide(juce::Graphics& g, float scale) {
    ImageViewComponent::drawSlide(g, scale);

    if (view->isEmpty() || MRImage->empty()) { return; }

    cv::Rect_<float> frameBox;
    bool showAsCircle;

#ifdef WITH_SPINNAKER
    if (recording) {
        bcam->sCam->get_last_frame(frameBox, showAsCircle);
    }
#endif

    if (simulating) {
        sCam->get_last_frame(frameBox, showAsCircle);
    }

    auto bounds = RectCtoJ < float >(frameBox);
    bounds.setPosition(bounds.getPosition() - view->getPosition());

    bounds *= view2screenScale(*view) * scale;

    g.setColour(juce::Colours::red);

    if (showAsCircle) {
        auto center = bounds.getCentre();
        fPoint radius = scopeRadius * view2screenScale(*view) * scale;;
        center -= radius;
        g.drawEllipse(center.getX(), center.getY(), 2 * radius.getX(), 2 * radius.getY(), 3);

    }
    else {
        g.drawRect(bounds, 3);
    }

}

void CaptureComponent::startRecording() {
  recording = true;
#ifdef WITH_SPINNAKER

  bcam.reset(new pathCam::SpinPath(config));
  bcam->add_observer(parent);
  scopeRadius = bcam->sCam->get_scope_radius();
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

  sCam.reset(new pathCam::StreamCam(config));
  sCam->add_observer(parent);
  scopeRadius = sCam->get_scope_radius();

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

