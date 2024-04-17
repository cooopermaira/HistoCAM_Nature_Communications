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
  ImageViewComponent(MainComponent *parent);
  ~ImageViewComponent() override;
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void drawSlide(juce::Graphics& g, float scale);
  void resized() override;
  
  bool keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) override;
  
  void setImage(std::shared_ptr< MRTiledImage > image);
  
private:
  std::shared_ptr< MRTiledImage> MRImage;
  
  void mouseDown(const juce::MouseEvent& event) override;
  void mouseDrag(const juce::MouseEvent& event) override;
  void mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) override;

  void scrollBarMoved(juce::ScrollBar* scrollBar, double newRangeStart) override;
  void updateScrollbar();
  
  void mouseMagnify (const MouseEvent&, float magnifyAmmount) override;
  
  inline void translate(fPoint amount){
    view += amount;
    updateScrollbar();
  }
  
  
  inline void scaleCenter(fPoint scale){
    fPoint center = view.getCentre();
    view -= center;
    view *= scale;
    view += center;
    updateScrollbar();
  }
  
  inline fPoint screen2view(){
    return fPoint(view.getHorizontalRange().getLength()/getLocalBounds().getHorizontalRange().getLength(),
                  view.getVerticalRange().getLength()/getLocalBounds().getVerticalRange().getLength());
  }
  
  inline fPoint view2screen(){
    return fPoint(getLocalBounds().getHorizontalRange().getLength()/view.getHorizontalRange().getLength(),
                  getLocalBounds().getVerticalRange().getLength()/view.getVerticalRange().getLength());
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
    
  CriticalSection mutex;
  
  MainComponent *parent;
  std::unique_ptr<ImageViewOverlay> controlsOverlay;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ImageViewComponent)
};
