//
//  AnnotateOverlay.h
//  pathCamApp
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef AnnotateOverlay_h
#define AnnotateOverlay_h

#include "JuceHeader.h"

class AnnoViewComponent;

class AnnotateOverlay final : public Component,
public Button::Listener
{
public:
  AnnotateOverlay (AnnoViewComponent* parent,
                  StringArray &iconNames,
                  OwnedArray<Drawable> &iconsFromZipFile) : parent(parent)
  {
    
    
    for (int i = 0; i < iconNames.size(); i++) {
      if(iconNames[i] == "polygon.svg"){
        polygonButton.reset( new SvgButton ("polygon", iconsFromZipFile[i]) );
        polygonButton->addListener(this);
        addAndMakeVisible(*polygonButton);
      }
      
      if(iconNames[i] == "segment.svg"){
        segmentButton.reset( new SvgButton ("segment", iconsFromZipFile[i]) );
        segmentButton->addListener(this);
        addAndMakeVisible(*segmentButton);
      }
      
      if(iconNames[i] == "dictate.svg"){
        dictateButton.reset( new SvgButton ("dictate", iconsFromZipFile[i]) );
        dictateButton->addListener(this);
        addAndMakeVisible(*dictateButton);
      }
      
      if(iconNames[i] == "measure.svg"){
        measureButton.reset( new SvgButton ("measure", iconsFromZipFile[i]) );
        measureButton->addListener(this);
        addAndMakeVisible(*measureButton);
      }
      
    }
  

    
  }
  
  ~AnnotateOverlay(){

  }
  
  void resized() override
  {
    auto area = getLocalBounds().reduced (4);
    dictateButton->setBounds(area.removeFromRight(100).reduced(20,0));
    segmentButton->setBounds(area.removeFromRight(100).reduced(20,0));
    polygonButton->setBounds(area.removeFromRight(100).reduced(20,0));
    measureButton->setBounds(area.removeFromRight(100).reduced(20,0));
  }
  
private:
  
  void buttonClicked(juce::Button* button) override;
  
  std::unique_ptr < SvgButton > polygonButton;
  std::unique_ptr < SvgButton > segmentButton;
  std::unique_ptr < SvgButton > dictateButton;
  std::unique_ptr < SvgButton > measureButton;

  AnnoViewComponent* parent;
    
  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AnnotateOverlay)
};


#endif /* AnnotateOverlay_h */
