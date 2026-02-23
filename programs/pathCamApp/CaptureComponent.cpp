//
//  CaptureComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"

#include <filesystem>


inline std::filesystem::path makeTempWavInCwd(const std::string& prefix = "recording")
{
  namespace fs = std::filesystem;
  fs::path p = fs::current_path() /(prefix + ".wav");
  return p;
}


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
    cptcmp->spinpath->run();
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

  uncacheThread = std::thread(&ImageViewComponent::uncacher, this);
  cacheThread = std::thread(&ImageViewComponent::cacher, this);

  auto start = std::chrono::high_resolution_clock::now();
  if (parent->audioDictationOn) {
    wavRecorder.init();
  }
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
  std::chrono::high_resolution_clock::now() - start).count();

#ifdef WITH_SPINNAKER
  if (!spinpath) {
    spinpath.reset(new pathCam::SpinPath(config));
    spinpath->add_observer(parent);
    scopeRadius = spinpath->sCam->get_scope_radius();
    aiOverlay->set_sCam(spinpath->sCam);
    parent->sCam = spinpath->sCam;
    sCam = spinpath->sCam;
    if (!parent->labelList) {
      setup_listbox();
    }
  }


#endif
  if (!sCam) {
    sCam = std::make_shared<pathCam::StreamCam>(config);
    sCam->add_observer(parent);
    scopeRadius = sCam->get_scope_radius();
    aiOverlay->set_sCam(sCam);
    parent->sCam = sCam;
    if (!parent->labelList) {
      setup_listbox();
    }

  }
  int k = 0;
}

void CaptureComponent::setup_listbox() const {
  parent->setup_listbox();
}

void CaptureComponent::drawSlide(juce::Graphics &g, float scale) {
  ImageViewComponent::drawSlide(g, scale);

  if (view->isEmpty() || MRImageSet->empty()) {
    return;
  }

  cv::Rect_<float> frameBox;
  bool showAsCircle;
  std::string magLabel;
  float lastScale;

#ifdef WITH_SPINNAKER
  if (recording) {
    int ignore;
    spinpath->sCam->get_last_frame(frameBox, showAsCircle, ignore, magLabel);
  }
#endif

  if (simulating) {
    int ignore;
    sCam->get_last_frame(frameBox, showAsCircle, ignore, magLabel, lastScale);
  }

  if (!simulating && !recording) { return; }

  auto bounds = RectCtoJ<float>(frameBox);
  bounds.setPosition(bounds.getPosition() - view->getPosition());

  bounds *= view2screenScale(*view) * scale;

  g.setColour(juce::Colours::red);

  if (showAsCircle) {
    auto center = bounds.getCentre();
    fPoint radius = scopeRadius * view2screenScale(*view) * scale * lastScale;
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
  if (!spinpath) {
    spinpath.reset(new pathCam::SpinPath(config));
    spinpath->add_observer(parent);
    scopeRadius = spinpath->sCam->get_scope_radius();
    aiOverlay->set_sCam(spinpath->sCam);
    parent->sCam = spinpath->sCam;
    sCam = spinpath->sCam;
    if (!parent->labelList) {
      setup_listbox();
    }
  }
  //aiOverlay->set_sCam(bcam->sCam);
  sCam->set_slide_label();
  parent->MRimage = sCam->get_MRimage_reference();

#endif

  setImage(parent->MRimage);
  parent->annotate->setImage(parent->MRimage);

  recentlyViewedSlides.push_unique(parent->MRimage);


  if (/*parent->audioDictationOn*/false) {
    if (!wavRecorder.initialised) {
      wavRecorder.init(1);
    }
    wavRecorder.startRecording(juce::File(makeTempWavInCwd("dictation").string()));
  }

  compositeThread.start(new bcamPocoRunnable(this));
  parent->startCompositingUIUpdates();

  captureOverlay->resized();
  aiOverlay->resized();
  resized();
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
      if (!parent->labelList) {
        setup_listbox();
      }

  }

  if (!inputPath.empty()) {
    sCam->set_input_file(inputPath);
  }
  sCam->set_slide_label();

  parent->MRimage = sCam->get_MRimage_reference();
  setImage(parent->MRimage);
  parent->annotate->setImage(parent->MRimage);

  recentlyViewedSlides.push_unique(parent->MRimage);


  if (parent->audioDictationOn) {
    if (!wavRecorder.initialised) {
      wavRecorder.init(1);
    }
    wavRecorder.startRecording(juce::File(makeTempWavInCwd("dictation").string()));
  }

  compositeThread.start(new sCamPocoRunnable(this));
  parent->startCompositingUIUpdates();

  captureOverlay->resized();
  aiOverlay->resized();
  //reportOverlay->resized();
  resized();
  repaint();
}


void CaptureComponent::stop() {
  Poco::Path finalAudio;
  if (wavRecorder.isRecording()) {
    wavRecorder.stop();
    finalAudio = MRImageSet->cwd;
    finalAudio.makeDirectory();
    finalAudio.setFileName("dictation");
    finalAudio.setExtension("wav");
    wavRecorder.writeFile.moveFileTo(juce::File(finalAudio.toString()));
  }
  if (recording) { stopRecording(); }
  if (simulating) { stopSimulating(); }

  if (parent->audioDictationOn) {
  // if (true){
    Poco::FastMutex::ScopedLock lock(parent->annotate->voiceAnnoMutex);
    parent->annotate->voiceAnnoOutstanding.push(std::pair(MRImageSet->index,juce::File(finalAudio.toString())));
    // parent->annotate->voiceAnnoOutstanding.push(std::pair(MRImageSet->index,juce::File("/home/cm/Documents/data/low_feat_10x/dictation.wav")));
    // parent->annotate->voiceAnnoOutstanding.push(std::pair(MRImageSet->index,juce::File("/home/cm/Documents/data/blur_test/config/0/dictation.wav")));
    parent->annotate->newVoiceAnnotation.set();
  }
}

void CaptureComponent::stopRecording() {
#ifdef WITH_SPINNAKER
  spinpath->stopCamera();
#endif

  recording = false;
  compositeThread.join();
  repaint();
}

void CaptureComponent::stopSimulating() {
  simulating = false;
  compositeThread.join();
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
      /*parent->capture->*/setImage(nullptr); //literally this
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
      procedureMode = 1;
      ready = sCam->pathcamReady;
#else
      //selecting input
      parent->fc.reset(new FileChooser("Choose an input file...", File("/home/cm/Documents/data/"),
                                       "*.txt"));

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
        std::cout << "not read" << std::endl;

        // Brief red flash overlay
        struct FlashOverlay : public juce::Component, public juce::Timer {
          FlashOverlay(juce::Component* parent) {
            parent->addAndMakeVisible(this);
            setBounds(parent->getLocalBounds());
            setAlwaysOnTop(true);
            startTimer(150);
          }
          void paint(juce::Graphics& g) override {
            g.setColour(juce::Colour::fromFloatRGBA(1.0f, 0.0f, 0.0f, 0.4f));
            g.fillRect(getLocalBounds());
          }
          void timerCallback() override {
            stopTimer();
            if (auto* p = getParentComponent()) p->removeChildComponent(this);
            delete this;
          }
        };
        new FlashOverlay(this);

        return true;
      }
      startRecording();
#else
      startSimulating();
#endif
      ready = false;
      sCam->pathcamReady = false;
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
