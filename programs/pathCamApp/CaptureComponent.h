//
//  CaptureComponent.hpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef CaptureComponent_h
#define CaptureComponent_h

#include "JuceHeader.h"

#ifdef WITH_SPINNAKER
class SpinPath;
#else
class StreamCam;
#endif


class CaptureComponent : public ImageViewComponent {
  friend class CaptureOverlay;
  friend class AIOverlay;

public:
  CaptureComponent(std::shared_ptr<fRectangle> view,
                   StringArray &iconNames,
                   OwnedArray<Drawable> &iconsFromZipFile, Poco::Util::LayeredConfiguration::Ptr config,
                   MainComponent *parent);

  Poco::Util::LayeredConfiguration::Ptr config;


  bool recording;
  bool simulating;
  bool ready = false;
  std::string inputPath;
  std::string caseLabel;

#ifdef WITH_SPINNAKER
  std::shared_ptr<pathCam::SpinPath> spinpath;
#endif

  std::shared_ptr<pathCam::StreamCam> sCam;

  MainComponent *parent;

  void set_input(const FileChooser &fc);

  void resized() override{
    auto area = getLocalBounds();

    // IMPORTANT: lay out ImageViewComponent *inside remaining area*
    layoutIn(area);

    {
      const ScopedLock lock(mutex);
      auto b = getLocalBounds();
      int width = 300;
      captureOverlay->setBounds({ b.getWidth() - width - 20, 20, width, 60 });
      aiOverlay->setBounds({ b.getWidth() - 100 - 20,
                             b.getHeight() - 100 - 20, 100, 100 });
    }
  }


  bool keyPressed(const juce::KeyPress &key, juce::Component *originatingComponent) override;

  void startRecording();

  void startSimulating();

  void stop();

  void stopRecording();

  void stopSimulating();

  void drawSlide(juce::Graphics &g, float scale) override;

  void setup_listbox();


  void paint(juce::Graphics &g) override {
    ImageViewComponent::paint(g);

    if (ready) {
      g.setColour(Colours::green);
      g.setFont(30);
      auto b = getLocalBounds().removeFromRight(175).removeFromBottom(50);
      g.drawText("Ready", b.getX(), b.getY(), 200, 40, Justification::left);
    }

    if (recording) {
      g.setColour(juce::Colours::red);
      g.drawRect(getLocalBounds(), 3);
      std::string r = "Recording...";
      g.setFont(30);
      auto b = getLocalBounds().removeFromRight(175).removeFromBottom(50);
      g.drawText(r, b.getX(), b.getY(), 200, 40, Justification::left);
    }
  }

private:
  std::unique_ptr<CaptureOverlay> captureOverlay;
  std::unique_ptr<AIOverlay> aiOverlay;
  //std::unique_ptr<ReportOverlay> reportOverlay;

  Poco::Thread compositeThread;
  Poco::Thread updateDrawThread;

  int procedureMode = 0;
  float scopeRadius;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CaptureComponent)
};

#endif /* CaptureComponent_h */
