#pragma once

#include "JuceHeader.h"

//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
class MainComponent final : public juce::Component
{
public:
  //==============================================================================
  MainComponent(std::shared_ptr< pathCam::StreamCam > bcam);
  ~MainComponent() override;
  
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override;
  
  void createShaders();

private:
  
  std::shared_ptr< pathCam::StreamCam > bcam;
  
  ToolbarDemoComp * toolbar;
  ImageViewComponent * imageview;
  

  CriticalSection mutex;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
