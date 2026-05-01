//
//  AnnotateComponent.h
//  pathCamApp
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef AnnotateComponent_h
#define AnnotateComponent_h


#include "JuceHeader.h"
struct tsWord {
  std::string word;
  long startMS = 0;
  long endMS = 0;
};


class AnnotateComponent : public juce::Component {
public:
  int segID = 0;
  MainComponent *parent;

  std::thread voiceHandlerThread;
  std::queue<std::pair<int,juce::File>> voiceAnnoOutstanding;
  Poco::Event newVoiceAnnotation;
  Poco::FastMutex voiceAnnoMutex;
  std::atomic<bool> voiceHandlerShouldContinue = true;

  ~AnnotateComponent() override {
    voiceHandlerShouldContinue = false;
    voiceHandlerThread.join();
  }

  void silly_test();


  std::vector<std::string> get_preconfig_anno();

  AnnotateComponent(std::shared_ptr<fRectangle> view,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile,
                    MainComponent *parent) : parent(parent), mode(Annotation::_NONE), newVoiceAnnotation(Poco::Event::EVENT_MANUALRESET){
    activeAnnotations.reset(new std::vector<std::shared_ptr<Annotation> >());
    allSlideAnnotations.push_back(activeAnnotations);

    leftComponent.reset(new AnnoListComponent(this, activeAnnotations, iconNames, iconsFromZipFile));
    addChildComponent(leftComponent.get());

    navPathList.reset(new NavPathListComponent(this));
    addChildComponent(navPathList.get());

    rightComponent.reset(new AnnoViewComponent(this, view, iconNames, iconsFromZipFile, activeAnnotations));

    addChildComponent(rightComponent.get());

    layout.setItemLayout(0, 100, -1, -0.20); // left component takes half the space initially
    layout.setItemLayout(1, 10, 10, 10); // resizer bar with a fixed size
    layout.setItemLayout(2, 100, -1, -0.80); // right component also takes half the space initially

    resizerBar.reset(new juce::StretchableLayoutResizerBar(&layout, 1, true));
    addChildComponent(resizerBar.get());

    voiceHandlerThread = std::thread(&AnnotateComponent::voice_annotation_handler,this);

    // silly_test();
  }

  void voice_annotation_handler();


  void update_active_annotations(int _index) {
    if (_index < 0) return;

    Poco::FastMutex::ScopedLock lock(allSlideAnnotationMutex);

    if ((int)allSlideAnnotations.size() <= _index) {
      allSlideAnnotations.resize(_index + 1);
    }

    if (!allSlideAnnotations[_index]) {
      allSlideAnnotations[_index] = std::make_shared<std::vector<std::shared_ptr<Annotation>>>();
    }

    activeAnnotations = allSlideAnnotations[_index];
    rightComponent->updateAnnotations(activeAnnotations);
    leftComponent->updateAnnotations(activeAnnotations);
  }

  void setImage(std::shared_ptr<MRTiledImageSet> image);

  void refreshImage() { rightComponent->refreshImage(); }

  void resized() override {
    auto area = getLocalBounds();
    juce::Component *components[] = {leftComponent.get(), resizerBar.get(), rightComponent.get()};

    layout.layOutComponents(components, 3, area.getX(), area.getY(), area.getWidth(), area.getHeight(), false, true);

    // Split the left column: leftComponent gets top 3/4, navPathList gets bottom 1/4.
    auto leftBounds = leftComponent->getBounds();
    int navHeight = leftBounds.getHeight() / 4;
    leftComponent->setBounds(leftBounds.withTrimmedBottom(navHeight));
    navPathList->setBounds(leftBounds.removeFromBottom(navHeight));

    rightComponent->resized();
  }

  void setVisible(bool shouldBeVisible) override {
    Component::setVisible(shouldBeVisible);

    leftComponent->setVisible(shouldBeVisible);
    navPathList->setVisible(shouldBeVisible);
    rightComponent->setVisible(shouldBeVisible);
    resizerBar->setVisible(shouldBeVisible);

    if (shouldBeVisible && rightComponent->MRImageSet) {
      getNavPathList()->setPaths(rightComponent->MRImageSet->navPaths);
    }

  }

  void fixAspectRatio() { rightComponent->fixAspectRatio(); }

  AnnoViewComponent *getViewComp() { return rightComponent.get(); }
  AnnoListComponent *getListComp() { return leftComponent.get(); }
  NavPathListComponent *getNavPathList() { return navPathList.get(); }

  const NavigationPath *getSelectedNavPath() const {
    if (ephemeralNavPath.has_value())
      return &ephemeralNavPath.value();
    return navPathList ? navPathList->getSelectedPath() : nullptr;
  }

  void toggleMode(int mode_in) {
    if (mode_in == mode) {
      mode = Annotation::_NONE;
    } else {
      mode = mode_in;
    }
  }

  void setMode(int mode_in) { mode = mode_in; }


  int getMode() { return mode; }

  void annotationsUpdated() { leftComponent->updatelist(); }

  std::shared_ptr<Annotation> getSelected() { return selected; }

  void setSelected(std::shared_ptr<Annotation> annotation) {
    selected = annotation;
    updateEphemeralNavPath();
  }

  void updateEphemeralNavPath() {
    ephemeralNavPath.reset();

    auto *vpp = dynamic_cast<VoicePointPoly *>(selected.get());
    if (!vpp || vpp->startFrameIdx < 0 || !rightComponent->MRImageSet) {
      getListComp()->setDistancePerFrame(std::nullopt);
      rightComponent->repaint();
      return;
    }

    long pathStartIdx = (long) ((1.0f - pathHistory) * (float) vpp->startFrameIdx);
    pathStartIdx = std::max(0L, pathStartIdx);

    auto centers = rightComponent->MRImageSet->frame_centers_from_frame_interval(
      pathStartIdx, vpp->startFrameIdx);

    if (!centers.empty()) {
      NavigationPath path;
      path.frameCenters.reserve(centers.size());
      for (const auto &c : centers)
        path.frameCenters.emplace_back((float) c.x, (float) c.y);
      path.distancePerFrame = path.calc_dist_per_frame();
      getListComp()->setDistancePerFrame(path.distancePerFrame, path.frameCenters.size());
      ephemeralNavPath = std::move(path);
    } else {
      getListComp()->setDistancePerFrame(std::nullopt);
    }

    rightComponent->repaint();
  }

  void removeSelected();

  std::shared_ptr<std::vector<std::shared_ptr<Annotation> > > activeAnnotations;
  std::vector<std::shared_ptr<std::vector<std::shared_ptr<Annotation> > > > allSlideAnnotations;

  Poco::FastMutex allSlideAnnotationMutex;

  std::shared_ptr<Annotation> selected;

  std::unique_ptr<AnnoListComponent> leftComponent;
  std::unique_ptr<NavPathListComponent> navPathList;
  std::unique_ptr<AnnoViewComponent> rightComponent;
  std::unique_ptr<juce::StretchableLayoutResizerBar> resizerBar;
  juce::StretchableLayoutManager layout;


  int mode;

  float annotationVisibility = 0.5f;
  float pathHistory = 0.0f;
  std::optional<NavigationPath> ephemeralNavPath;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AnnotateComponent)
};

#endif /* AnnotateComponent_h */
