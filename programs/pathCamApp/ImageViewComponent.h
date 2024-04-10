#pragma once

#include "JuceHeader.h"

//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
class ImageViewComponent final : public juce::Component
{
public:
  //==============================================================================
  ImageViewComponent(std::shared_ptr< pathCam::StreamCam > bcam);
  ~ImageViewComponent() override;
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override;
  
  //void createShaders();

private:
  
  std::shared_ptr< pathCam::StreamCam > bcam;
  
  void mouseDown(const juce::MouseEvent& event) override;
  void mouseDrag(const juce::MouseEvent& event) override;
  
    
  TiledImage timage;
  juce::Image checkerboard;
  
  
  juce::Point<int> imagePosition;
  juce::Point<int> lastMousePosition;

  juce::Rectangle<int> bounds;
  float desktopScale;
  juce::Rectangle<int> pixel_bounds;
  
  CriticalSection mutex;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ImageViewComponent)
};
