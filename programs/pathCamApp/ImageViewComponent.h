#pragma once

#include "JuceHeader.h"

//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
class ImageViewComponent final : public juce::Component, public juce::ScrollBar::Listener
{
public:
  //==============================================================================
  ImageViewComponent(std::shared_ptr< MRTiledImage > MRImage);
  ~ImageViewComponent() override;
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override;
  
  //void createShaders();

private:
  
  std::shared_ptr< MRTiledImage> MRImage;
  
  void mouseDown(const juce::MouseEvent& event) override;
  void mouseDrag(const juce::MouseEvent& event) override;
  void mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) override;
  void scrollBarMoved(juce::ScrollBar* scrollBar, double newRangeStart) override;
  
  juce::Image createCheckerboardImage(int width, int height, int squareSize,
                                      juce::Colour colour1, juce::Colour colour2);

    
  static inline fRectangle zoomBoundsCenter(fRectangle r, float scale){
    float center_x = r.getCentreX();
    float center_y = r.getCentreY();
    r -= fPoint(center_x, center_y);
    r *= scale;
    r += fPoint(center_x, center_y);
    return r;
  }
  
  static inline fRectangle translateBounds(fRectangle r, fPoint delta){
    r += delta;
    return r;
  }

  juce::Image checkerboard;
  
  juce::Point<int> imagePosition;
  juce::Point<int> lastMousePosition;

  fRectangle bounds;
  
  juce::ScrollBar horizontalScrollBar{ false }; // Horizontal scroll bar
  juce::ScrollBar verticalScrollBar{ true }; // Vertical scroll bar

  float scale;
  
  CriticalSection mutex;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ImageViewComponent)
};
