//
//  CaptureOverlay.h
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef AIOverlay_h
#define AIOverlay_h

#include "JuceHeader.h"

class CaptureComponent;

class AIClassify : public Poco::Runnable {
private:
  std::shared_ptr<pathCam::StreamCam> sCam;
public:
  AIClassify(std::shared_ptr<pathCam::StreamCam> sCam) : sCam(sCam){};
  virtual void run(){
    sCam->run_agg_classify();
  };
};

class AIOverlay final : public Component,
                        public Button::Listener, private juce::Timer {
public:
  AIOverlay(CaptureComponent *parent,
            StringArray &iconNames,
            OwnedArray<Drawable> &iconsFromZipFile);

  ~AIOverlay() {

  }

  void resized() override;

  void paint(juce::Graphics &g) override;

  inline void set_sCam(std::shared_ptr<pathCam::StreamCam> _sCam){ sCam= _sCam;}

private:
  std::shared_ptr<pathCam::StreamCam> sCam;

  bool AIready;

  void timerCallback() override;

  float currentOpacity = 1.0f;
  bool increasingOpacity = false;

  void buttonClicked(juce::Button *button) override;

  std::unique_ptr<SvgButton> AIthinkingButton;
  std::unique_ptr<SvgButton> AIreadyButton;

  CaptureComponent *parent;
  Poco::Thread classifyThread;
  AIClassify *aic_r;

  std::thread t;
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AIOverlay)
};


#endif /* AIOverlay_h */
