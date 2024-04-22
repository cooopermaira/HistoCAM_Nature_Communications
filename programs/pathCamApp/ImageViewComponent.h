#pragma once

#include "JuceHeader.h"


//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
class ImageViewComponent : public juce::Component, public juce::ScrollBar::Listener, public juce::KeyListener
{
  
  friend class ImageViewOverlay;
  
  
public:
  //==============================================================================
  ImageViewComponent(MainComponent *parent,
                     std::shared_ptr < fRectangle > view,
                     StringArray &iconNames,
                     OwnedArray<Drawable> &iconsFromZipFile);
  ~ImageViewComponent() override;
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void drawSlide(juce::Graphics& g, float scale);
  void resized() override;
  
  bool keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) override;
  
  void setImage(std::shared_ptr< MRTiledImage > image);
  
  void fixAspectRatio(){
    if(!MRImage || !isVisible()){ return; }
    
    fRectangle b = fRectangle(getLocalBounds().getX(), getLocalBounds().getY(), getLocalBounds().getWidth(), getLocalBounds().getHeight());
    fPoint center = view->getCentre();
    
    if(b.getAspectRatio() != view->getAspectRatio()){
      
      int newWidthView, newHeightView;
      if (view->getWidth() / view->getHeight() < b.getAspectRatio()) {
        newWidthView   =   view->getHeight() * b.getAspectRatio();
        newHeightView  =   view->getHeight();
      } else {
        newWidthView = view->getWidth();
        newHeightView = view->getWidth() / b.getAspectRatio();
      }
      
      view->setSize( newWidthView, newHeightView);
      view->setCentre(center);
    }
  }
  
private:
  std::shared_ptr< MRTiledImage> MRImage;
  
  void mouseDown(const juce::MouseEvent& event) override;
  void mouseDrag(const juce::MouseEvent& event) override;
  void mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) override;
  
  void scrollBarMoved(juce::ScrollBar* scrollBar, double newRangeStart) override;
  void updateScrollbar();
  
  
  void mouseMagnify (const MouseEvent&, float magnifyAmmount) override;
  

  
  void zoomAndCenter(){
    if(!MRImage || !isVisible()){ return; }
    juce::Rectangle<int> b = getLocalBounds();
    *view = fRectangle(b.getX(),b.getY(),b.getWidth(),b.getHeight());
    //view.reset(new fRectangle(b.getX(),b.getY(),b.getWidth(),b.getHeight()));
    view->setCentre(MRImage->bounds.getCentre());
    
    float scale = max((float)MRImage->bounds.getHorizontalRange().getLength()/
                      (float)view->getHorizontalRange().getLength(),
                      (float)MRImage->bounds.getVerticalRange().getLength()/
                      (float)view->getVerticalRange().getLength());
    
    scaleCenter(fPoint(scale,scale));
    
  }
  
  inline void translate(fPoint amount){
    if(!MRImage){ return; }
    if(isVisible()){ *view += amount; }
    updateScrollbar();
  }
  
  inline void scaleCenter(fPoint scale){
    if(!MRImage){ return; }
    fPoint center = view->getCentre();
    if(isVisible()){
      *view -= center;
      *view *= scale;
      *view += center;
    }
    updateScrollbar();
  }
  
protected:
  inline fPoint screen2view(){
    if(!MRImage){ return fPoint(); }
    return fPoint(view->getHorizontalRange().getLength()/getLocalBounds().getHorizontalRange().getLength(),
                  view->getVerticalRange().getLength()/getLocalBounds().getVerticalRange().getLength());
  }
  
  inline fPoint view2screen(){
    if(!MRImage){ return fPoint(); }
    return fPoint(getLocalBounds().getHorizontalRange().getLength()/view->getHorizontalRange().getLength(),
                  getLocalBounds().getVerticalRange().getLength()/view->getVerticalRange().getLength());
  }
  
  
private:
  juce::Image createCheckerboardImage(int width, int height, int squareSize,
                                      juce::Colour colour1, juce::Colour colour2);
  
  
  juce::Image checkerboard;
  
  juce::Point<int> imagePosition;
  juce::Point<int> lastMousePosition;
  
  juce::Rectangle<int> old_bounds;
  
  juce::ScrollBar horizontalScrollBar{ false }; // Horizontal scroll bar
  juce::ScrollBar verticalScrollBar{ true }; // Vertical scroll bar
  
  
  MainComponent *parent;
  
  std::unique_ptr<ImageViewOverlay> controlsOverlay;
  
protected:
  std::shared_ptr < fRectangle > view;

  CriticalSection mutex;
  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ImageViewComponent)
};
