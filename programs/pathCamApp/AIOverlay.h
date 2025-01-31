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

class AIOverlay final : public Component,
public Button::Listener, private juce::Timer
{
public:
  AIOverlay (CaptureComponent* parent,
             StringArray &iconNames,
             OwnedArray<Drawable> &iconsFromZipFile) : parent(parent)
  {
    
    
    for (int i = 0; i < iconNames.size(); i++) {
      
      if(iconNames[i] == "AI_ready.svg"){
        AIreadyButton.reset( new SvgButton ("AI_ready", iconsFromZipFile[i]) );
        AIreadyButton->addListener(this);
        addAndMakeVisible(*AIreadyButton);
      }

      if(iconNames[i] == "AI_thinking.svg"){
        AIthinkingButton.reset( new SvgButton ("AI_thinking", iconsFromZipFile[i]) );
        AIthinkingButton->addListener(this);
        addAndMakeVisible(*AIthinkingButton);
      }
      
    }
  
    startTimerHz(30);
    
  }
  
  ~AIOverlay(){

  }
  
  void resized() override;
  
  void paint(juce::Graphics &g) override;
  
private:
  
  void timerCallback() override;
  
  float currentOpacity = 1.0f;
  bool increasingOpacity = false;
  
  void buttonClicked(juce::Button* button) override;
  
  std::unique_ptr < SvgButton > AIthinkingButton;
  std::unique_ptr < SvgButton > AIreadyButton;
  
  CaptureComponent* parent;
    
  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AIOverlay)
};


#endif /* AIOverlay_h */
