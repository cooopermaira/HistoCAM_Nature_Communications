#pragma once

#include "JuceHeader.h"


class DemoControlsOverlay final : public Component,
                                  private Slider::Listener
{
public:
  DemoControlsOverlay ()
    {
        addAndMakeVisible (statusLabel);
        statusLabel.setJustificationType (Justification::topLeft);
        statusLabel.setFont (Font (14.0f));

        addAndMakeVisible (sizeSlider);
        sizeSlider.setRange (0.0, 1.0, 0.001);
        sizeSlider.addListener (this);

        addAndMakeVisible (zoomLabel);
        zoomLabel.attachToComponent (&sizeSlider, true);

    }

    void initialise()
    {
        speedSlider.setValue (0.01);
        sizeSlider .setValue (0.5);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (4);

        speedSlider         .setBounds (area.removeFromBottom (25));
        sizeSlider          .setBounds (area.removeFromBottom (25));

        statusLabel.setBounds (area);
    }

    Label statusLabel;

private:
    void sliderValueChanged (Slider*) override
    {
//        const ScopedLock lock (demo.mutex);
//
//        demo.scale         = (float) sizeSlider .getValue();
//        demo.rotationSpeed = (float) speedSlider.getValue();
    }



    Label speedLabel  { {}, "Speed:" },
          zoomLabel   { {}, "Zoom:" };

    Slider speedSlider, sizeSlider;

    ToggleButton showBackgroundToggle  { "Draw 2D graphics in background" };

    std::atomic<bool> buttonDown { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DemoControlsOverlay)
};

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
  void drawSlide(juce::Graphics& g);
  void resized() override;
  
  bool keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) override;
  
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
  
  std::unique_ptr<DemoControlsOverlay> controlsOverlay;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ImageViewComponent)
};
