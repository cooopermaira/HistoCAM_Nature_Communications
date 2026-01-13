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
  bool shadeLevels;
  bool shadeClasses;
  int componentSelector = 0;
  std::vector<juce::Colour> levelColors = {
    Colour(66, 91, 176), Colour(120, 154, 175), Colour(190, 217, 201),
    Colour(243, 249, 243)
  };
  //==============================================================================
  ImageViewComponent(std::shared_ptr<fRectangle> view,
                     StringArray &iconNames,
                     OwnedArray<Drawable> &iconsFromZipFile, MainComponent *parent);
  ~ImageViewComponent() override;

  //==============================================================================
  void paint(juce::Graphics &g) override;
  virtual void drawSlide(juce::Graphics &g, float scale);
  void resized() override;

  bool keyPressed(const juce::KeyPress &key, juce::Component *originatingComponent) override;

  void setImage(std::shared_ptr<MRTiledImageSet> image);

  void notify_new_data() { newData = true; }

  void fixAspectRatio()
  {
    if (!MRImageSet || !isVisible())
    {
      return;
    }

    fRectangle b = fRectangle(getLocalBounds().getX(), getLocalBounds().getY(), getLocalBounds().getWidth(), getLocalBounds().getHeight());
    fPoint center = view->getCentre();

    if (b.getAspectRatio() != view->getAspectRatio())
    {

      int newWidthView, newHeightView;
      if (view->getWidth() / view->getHeight() < b.getAspectRatio())
      {
        newWidthView = view->getHeight() * b.getAspectRatio();
        newHeightView = view->getHeight();
      }
      else
      {
        newWidthView = view->getWidth();
        newHeightView = view->getWidth() / b.getAspectRatio();
      }

      view->setSize(newWidthView, newHeightView);
      view->setCentre(center);
    }
  }

  void refreshImage();

protected:
  std::shared_ptr<MRTiledImageSet> MRImageSet;
  MainComponent *parent;
  std::atomic<bool> newData;
  cv::Mat greenShade;
  cv::Mat holding1;
  cv::Mat holding2;
  cv::Mat channelHolding;
  std::vector<cv::Mat> channels;

  OpenGLContext gl;

  void drawLayer(Graphics &g, float scale, std::shared_ptr<MRTiledImage>tiledImage);

  void mouseDown(const juce::MouseEvent &event) override;
  void mouseDrag(const juce::MouseEvent &event) override;
  void mouseWheelMove(const MouseEvent &event, const MouseWheelDetails &fwheel) override;

  void scrollBarMoved(juce::ScrollBar *scrollBar, double newRangeStart) override;
  void updateScrollbar();

  void mouseMagnify(const MouseEvent &, float magnifyAmmount) override;

  void zoomAndCenter();

  inline void translate(fPoint amount)
  {
    if (!MRImageSet)
    {
      return;
    }
    if (isVisible())
    {
      *view += amount;
    }
    updateScrollbar();
  }

  inline void scaleCenter(fPoint scale)
  {
    if (!MRImageSet)
    {
      return;
    }
    fPoint center = view->getCentre();
    if (isVisible())
    {
      *view -= center;
      *view *= scale;
      *view += center;
    }
    updateScrollbar();
  }

  inline fPoint screen2viewScale(fRectangle myview)
  {
    if (!MRImageSet)
    {
      return fPoint();
    }
    return fPoint(myview.getHorizontalRange().getLength() / getLocalBounds().getHorizontalRange().getLength(),
                  myview.getVerticalRange().getLength() / getLocalBounds().getVerticalRange().getLength());
  }

  inline fPoint view2screenScale(fRectangle myview)
  {
    if (!MRImageSet)
    {
      return fPoint();
    }
    return fPoint(getLocalBounds().getHorizontalRange().getLength() / myview.getHorizontalRange().getLength(),
                  getLocalBounds().getVerticalRange().getLength() / myview.getVerticalRange().getLength());
  }

  inline fPoint screen2view(fPoint p, fRectangle myview)
  {
    if (!MRImageSet)
    {
      return p;
    }
    return p * screen2viewScale(myview) + myview.getPosition();
  }

  inline fPoint view2screen(fPoint p, fRectangle myview)
  {
    if (!MRImageSet)
    {
      return p;
    }
    return (p - myview.getPosition()) * view2screenScale(myview);
  }

private:
  juce::Image createCheckerboardImage(int width, int height, int squareSize,
                                      juce::Colour colour1, juce::Colour colour2);

  juce::Image checkerboard;

  juce::Point<int> imagePosition;
  juce::Point<int> lastMousePosition;

  juce::Rectangle<int> old_bounds;

  juce::ScrollBar horizontalScrollBar{false}; // Horizontal scroll bar
  juce::ScrollBar verticalScrollBar{true};    // Vertical scroll bar

  std::unique_ptr<ImageViewOverlay> controlsOverlay;

  juce::Time lastRepaintTime;
  static const int TARGET_FPS = 60;
  static const int MIN_REPAINT_INTERVAL_MS = 1000 / TARGET_FPS;

protected:
  std::shared_ptr<fRectangle> view;

  CriticalSection mutex;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ImageViewComponent)
};
