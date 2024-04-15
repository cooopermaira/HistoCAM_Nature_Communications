#pragma once

#include "JuceHeader.h"

//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
class ImageViewComponent final : public juce::Component, public juce::ScrollBar::Listener, public juce::KeyListener
{
public:
  //==============================================================================
  ImageViewComponent(std::shared_ptr< MRTiledImage > MRImage);
  ~ImageViewComponent() override;
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override;
  
  bool keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) override;
  
private:
  std::shared_ptr< MRTiledImage> MRImage;
  
  void mouseDown(const juce::MouseEvent& event) override;
  void mouseDrag(const juce::MouseEvent& event) override;
  void mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) override;
  void scrollBarMoved(juce::ScrollBar* scrollBar, double newRangeStart) override;
  
  inline void scaleCenter(fPoint scale){
    fPoint center = view.getCentre();
    view -= center;
    view *= scale;
    view += center;
  }
  
  inline fPoint screen2view(){
    return fPoint(view.getHorizontalRange().getLength()/getBounds().getHorizontalRange().getLength(),
                  view.getVerticalRange().getLength()/getBounds().getVerticalRange().getLength());
  }
  
  inline fPoint view2screen(){
    return fPoint(getBounds().getHorizontalRange().getLength()/view.getHorizontalRange().getLength(),
                  getBounds().getVerticalRange().getLength()/view.getVerticalRange().getLength());
  }
  
  juce::Image createCheckerboardImage(int width, int height, int squareSize,
                                      juce::Colour colour1, juce::Colour colour2);
  
  
  juce::Image checkerboard;
  
  juce::Point<int> imagePosition;
  juce::Point<int> lastMousePosition;
  
  fRectangle view;
  juce::Rectangle<int> old_bounds;
  
  juce::ScrollBar horizontalScrollBar{ false }; // Horizontal scroll bar
  juce::ScrollBar verticalScrollBar{ true }; // Vertical scroll bar
  
  float scale;
  
  CriticalSection mutex;
  
  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ImageViewComponent)
};
