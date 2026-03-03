#pragma once

#include "JuceHeader.h"

//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
template <class T,
          class Hash = std::hash<T>,
          class KeyEqual = std::equal_to<T>>
struct UniqueFifo {
private:
  mutable std::mutex m_;
  std::deque<T> fifo_;
  std::unordered_set<T, Hash, KeyEqual> seen_;

public:
  UniqueFifo() = default;

  explicit UniqueFifo(std::size_t reserve_count) {
    // safe in ctor (no concurrent access yet)
    seen_.reserve(reserve_count);
  }

  bool empty() const noexcept {
    std::lock_guard<std::mutex> lk(m_);
    return fifo_.empty();
  }

  std::size_t size() const noexcept {
    std::lock_guard<std::mutex> lk(m_);
    return fifo_.size();
  }

  void clear() {
    std::lock_guard<std::mutex> lk(m_);
    fifo_.clear();
    seen_.clear();
  }

  // Enqueue only if not already present
  bool push_unique(const T& v) {
    std::lock_guard<std::mutex> lk(m_);
    if (seen_.insert(v).second) {
      fifo_.push_back(v);
      return true;
    }
    return false;
  }

  bool push_unique(T&& v) {
    std::lock_guard<std::mutex> lk(m_);
    // Need a stable key for the set; we cannot "move into both".
    // So: insert a copy into the set; move into deque only if inserted.
    auto [it, inserted] = seen_.emplace(v);
    if (inserted) {
      fifo_.push_back(std::move(v));
      return true;
    }
    return false;
  }

  // Pop front; returns false if empty
  bool pop(T& out) {
    std::lock_guard<std::mutex> lk(m_);
    if (fifo_.empty()) return false;
    out = std::move(fifo_.front());
    fifo_.pop_front();
    seen_.erase(out); // allows re-enqueue later
    return true;
  }

  // Convenience: pop and return by value
  T pop_or_throw() {
    std::lock_guard<std::mutex> lk(m_);
    if (fifo_.empty()) throw std::runtime_error("UniqueFifo: pop on empty");
    T out = std::move(fifo_.front());
    fifo_.pop_front();
    seen_.erase(out);
    return out;
  }

  bool contains(const T& v) const {
    std::lock_guard<std::mutex> lk(m_);
    return seen_.find(v) != seen_.end();
  }

  // Optional: peek without popping
  bool peek(T& out) const {
    std::lock_guard<std::mutex> lk(m_);
    if (fifo_.empty()) return false;
    out = fifo_.front(); // copy
    return true;
  }
};



class ImageViewComponent : public juce::Component, public juce::ScrollBar::Listener, public juce::KeyListener
{

  friend class ImageViewOverlay;
  friend class CaptureComponent;


public:
  bool shadeLevels;
  bool shadeClasses;
  int componentSelector = 0;
  int MRImageSetSelector = 0;

  UniqueFifo<std::shared_ptr<MRTiledImageSet>> recentlyViewedSlides;

  std::vector<juce::Colour> levelColors = {
    /*Colour(66, 91, 176), Colour(120, 154, 175), Colour(190, 217, 201),
    Colour(243, 249, 243)*/
    Colour(0, 255, 0), Colour(255, 255, 0), Colour(255, 0, 0),
    Colour(211, 211, 211) //20x green, 10x yellow, 4x red, 2x gray
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

  std::shared_ptr<MRTiledImageSet> MRImageSet;
  MainComponent *parent;
  std::atomic<bool> newData;
  std::atomic<bool> cacherKeepGoing = true;
  cv::Mat greenShade;
  cv::Mat holding1;
  cv::Mat holding2;
  cv::Mat channelHolding;
  std::vector<cv::Mat> channels;

  OpenGLContext gl;

  Poco::FastMutex loadASAPMutex;
  std::queue<std::shared_ptr<MRTiledImageSet>> loadMRImageSetsASAP;
  Poco::Event loadMRImageSetASAPEvent;
  std::thread cacheThread, uncacheThread;

  void drawLayer(Graphics &g, float scale, std::shared_ptr<MRTiledImage>tiledImage);

  void mouseDown(const juce::MouseEvent &event) override;
  void mouseDrag(const juce::MouseEvent &event) override;
  void mouseWheelMove(const MouseEvent &event, const MouseWheelDetails &fwheel) override;

  void scrollBarMoved(juce::ScrollBar *scrollBar, double newRangeStart) override;
  void updateScrollbar();

  void mouseMagnify(const MouseEvent &, float magnifyAmmount) override;

  void zoomAndCenter();

  void adjust_MRImageSet(int mode);

  void save_slide_set() const;

  void cacher();

  void uncacher();

  void q_cache();

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

  inline void scaleCenter(fPoint scale,bool _uncache = true){
    if (!MRImageSet)
    {
      return;
    }
    fPoint center = view->getCentre();
    if (isVisible()){
      if (_uncache) {
        q_cache();
      }
      *view -= center;
      *view *= scale;
      *view += center;
    }
    updateScrollbar();
  }

  void layoutIn(juce::Rectangle<int> area);

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

  juce::Image createCheckerboardImage(int width, int height, int squareSize,
                                      juce::Colour colour1, juce::Colour colour2);

  juce::Image checkerboard;

  juce::Point<int> imagePosition;
  juce::Point<int> lastMousePosition;

  juce::Rectangle<int> old_bounds;

  juce::ScrollBar horizontalScrollBar{false}; // Horizontal scroll bar
  juce::ScrollBar verticalScrollBar{true};    // Vertical scroll bar


  juce::Time lastRepaintTime;
  static const int TARGET_FPS = 60;
  static const int MIN_REPAINT_INTERVAL_MS = 1000 / TARGET_FPS;

  juce::Rectangle<int> oldBounds;


  std::unique_ptr<ImageViewOverlay> controlsOverlay;

  std::shared_ptr<fRectangle> view;

  CriticalSection mutex;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ImageViewComponent)
};
