//
//  CaptureComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include <memory>

#include "JuceHeader.h"


class sCamPocoRunnable : public Poco::Runnable {
public:
  sCamPocoRunnable(CaptureComponent *cptcmp) : cptcmp(cptcmp) {
  };
  CaptureComponent *cptcmp;

  void run() override {
    cptcmp->sCam->run();
    MessageManager::callAsync(
      [safeParent = Component::SafePointer(cptcmp->parent)]() mutable {
        if (safeParent != nullptr)
          safeParent->stopCompositingUIUpdates();
      });
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
    MessageManager::callAsync(
      [safeParent = Component::SafePointer(cptcmp->parent)]() mutable {
        if (safeParent != nullptr)
          safeParent->stopCompositingUIUpdates();
      });
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

  uncacheThread = std::thread(&ImageViewComponent::uncacher, this);
  cacheThread = std::thread(&ImageViewComponent::cacher, this);
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
    aiOverlay->set_sCam(bcam->sCam);
    parent->sCam = bcam->sCam;
  }
  //aiOverlay->set_sCam(bcam->sCam);
  bcam->sCam->set_slide_label();
#endif

  parent->MRimage = sCam->get_MRimage_reference();
  setImage(parent->MRimage);
  parent->annotate->setImage(parent->MRimage);

  recentlyViewedSlides.push_unique(parent->MRimage);

  compositeThread.start(new bcamPocoRunnable(this));
  parent->startCompositingUIUpdates();

  captureOverlay->resized();
  aiOverlay->resized();
  repaint();
}

void CaptureComponent::startSimulating() {
  if (compositeThread.isRunning()) {
    compositeThread.join();
  }

  simulating = true;

  if (!sCam) {
    sCam = std::make_shared<pathCam::StreamCam>(config);
    sCam->add_observer(parent);
    scopeRadius = sCam->get_scope_radius();
    aiOverlay->set_sCam(sCam);
    parent->sCam = sCam;
  }

  if (!inputPath.empty()) {
    sCam->set_input_file(inputPath);
  }
  sCam->set_slide_label();

  parent->MRimage = sCam->get_MRimage_reference();
  setImage(parent->MRimage);
  parent->annotate->setImage(parent->MRimage);

  recentlyViewedSlides.push_unique(parent->MRimage);


  compositeThread.start(new sCamPocoRunnable(this));
  parent->startCompositingUIUpdates();

  captureOverlay->resized();
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


  compositeThread.join();
  recording = false;
  repaint();
}

void CaptureComponent::stopSimulating() {
  simulating = false;
  // parent->stopCompositingUIUpdates();
  repaint();
}

void CaptureComponent::set_input(const FileChooser &fc) {
  File result = fc.getResult();
  if (result.exists()) {
    inputPath = result.getFullPathName().toStdString();
    procedureMode = 1;
    ready = true;
    if (MRImageSet) {
      // parent->imageview->setImage(nullptr);
      parent->capture->setImage(nullptr); //parent->capture is just this
      parent->annotate->setImage(nullptr);
    }
    captureOverlay->resized();
    repaint();
  }
}

bool CaptureComponent::keyPressed(const juce::KeyPress &key, juce::Component *originatingComponent) {
  ImageViewComponent::keyPressed(key, originatingComponent);

  if (!isVisible()) { return false; }

  if (key.getKeyCode() == KeyPress::spaceKey) {
    if (procedureMode == 0) {
      // begin selecting or setting up input
#ifdef WITH_SPINNAKER
      //open camera barcode reader
#else
      //selecting input
      parent->fc.reset(new FileChooser("Choose an image to open...", File("/home/cm/Documents/data/blur_test/config/"),
                                       "*.png,*.jpeg,*.tiff"));

      parent->fc->launchAsync(FileBrowserComponent::openMode
                              | FileBrowserComponent::canSelectFiles,
                              std::bind(&CaptureComponent::set_input, this, std::placeholders::_1));
      return true;
#endif
    }
    if (procedureMode == 1) {
      // begin an actual recording/simulation

#ifdef WITH_SPINNAKER
      if (!ready) {
        std::cout << "slide barcode not read" << std::endl;
        return true;
      }
      startRecording();
#else
      startSimulating();
#endif
      ready = false;
      procedureMode = 2;
      return true;
    }
    if (procedureMode == 2) {
      stop();
      captureOverlay->resized();
      procedureMode = 0;
      return true;
    }
    std::cout << "invalid procedure mode, resetting" << std::endl;
    procedureMode = 0;
    return true;

    if (recording || simulating) { stop(); }
  }
  return false; // Key press not handled
}
