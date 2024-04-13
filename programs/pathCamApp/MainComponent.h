#pragma once

#include "JuceHeader.h"

//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
class MainComponent final : public juce::Component, public juce::KeyListener 
{
public:
  //==============================================================================
  MainComponent(std::shared_ptr< pathCam::StreamCam > bcam);
  ~MainComponent() override;
  
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override;
  
  bool keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) override {
      if (key == juce::KeyPress::createFromDescription("spacebar")) {
        std::cout << "space\n" << "\n";
          // Do something when space bar is pressed
          return true;  // Key press handled
      }
    
      return false;  // Key press not handled
  }
  
private:
  
  std::shared_ptr< pathCam::StreamCam > bcam;
  std::shared_ptr< MRTiledImage >  MRimage;
  
  
  ToolbarComp * toolbar;
  ImageViewComponent * imageview;
  
  juce::ProgressBar * progressBar;
  

  CriticalSection mutex;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
