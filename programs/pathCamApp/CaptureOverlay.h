//
//  CaptureOverlay.h
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef CaptureOverlay_h
#define CaptureOverlay_h

#include "JuceHeader.h"

class CaptureComponent;

class CaptureOverlay final : public Component,
public Button::Listener
{
public:
  CaptureOverlay (CaptureComponent* parent,
                  StringArray &iconNames,
                  OwnedArray<Drawable> &iconsFromZipFile) : parent(parent)
  {
    
    
    for (int i = 0; i < iconNames.size(); i++) {
      
      if(iconNames[i] == "record.svg"){
        recordButton.reset( new SvgButton ("record", iconsFromZipFile[i]) );
        recordButton->addListener(this);
        addAndMakeVisible(*recordButton);
      }
      
      if(iconNames[i] == "stop.svg"){
        stopButton.reset( new SvgButton ("stop", iconsFromZipFile[i]) );
        stopButton->addListener(this);
        addAndMakeVisible(*stopButton);
      }
      
    }
  

    
  }
  
  ~CaptureOverlay(){

  }
  
  void resized() override
  {
    auto area = getLocalBounds().reduced (4);
    stopButton->setBounds(area.removeFromRight(100).reduced(20,0));
    recordButton->setBounds(area.removeFromRight(100).reduced(20,0));
  }
  
private:
  
  void buttonClicked(juce::Button* button) override;
  
  std::unique_ptr < SvgButton > recordButton;
  std::unique_ptr < SvgButton > stopButton;
  
  CaptureComponent* parent;
    
  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CaptureOverlay)
};


#endif /* CaptureOverlay_h */
