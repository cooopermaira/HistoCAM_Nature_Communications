#include "JuceHeader.h"


//==============================================================================
//MainComponent::MainComponent(std::shared_ptr<pathCam::StreamCam> bcam) : bcam(bcam) {
MainComponent::MainComponent(Poco::Util::LayeredConfiguration::Ptr config) : config(config), clientHasData(false) {
  //Won't work for deployment, but ok for now
  std::stringstream ss;
  ss << PROJECT_SOURCE_DIR << "/resources/hud_icons.zip";

  ZipFile icons(File(ss.str().c_str()));

  for (int i = 0; i < icons.getNumEntries(); ++i) {
    std::unique_ptr<InputStream> svgFileStream(icons.createStreamForEntry(i));

    if (svgFileStream.get() != nullptr) {
      iconNames.add(icons.getEntry(i)->filename);
      iconsFromZipFile.add(Drawable::createFromImageDataStream(*svgFileStream));
    }
  }

  //imagePyramid.reset(new MRTiledImage);
  MRimage.reset(new MRTiledImageSet);
  toolbar = new ToolbarComponent(this);
  view.reset(new fRectangle());
  // imageview = new ImageViewComponent(view, iconNames, iconsFromZipFile,this);
  capture = new CaptureComponent(view, iconNames, iconsFromZipFile, config, this); //pass reference to bcam

  //capture->bcam->set_MainComponent_reference(this);
  annotate = new AnnotateComponent(view, iconNames, iconsFromZipFile, this);

  // addAndMakeVisible(imageview);
  addAndMakeVisible(toolbar);

  addChildComponent(capture);
  addChildComponent(annotate);

  capture->setup_listbox();

  cwd = juce::File("/home/");
  dirFilter = std::make_unique<juce::WildcardFileFilter>("*", "*", "All Files");
  dirBrowser = std::make_unique<juce::FileBrowserComponent>(
    juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
    cwd, dirFilter.get(), nullptr);
  addAndMakeVisible(*dirBrowser);
  dirSelectButton.setButtonText("Select");
  dirSelectButton.addListener(this);
  addAndMakeVisible(dirSelectButton);

  // Initialize slideListButton
  for (int i = 0; i < iconNames.size(); i++) {
    if (iconNames[i] == "slideList.svg") {
      slideListButton.reset(new SvgButton("slides", iconsFromZipFile[i]));
      slideListButton->addListener(this);
      addAndMakeVisible(*slideListButton);
      slideListButton->setVisible(false); // Initially hidden until sCam is available
    }
  }

  setWantsKeyboardFocus(true);
  // addKeyListener(imageview);
  addKeyListener(capture);
  addKeyListener(annotate->getViewComp());


  setSize(1024, 768);
}


MainComponent::~MainComponent() {
  delete progressBar;
  delete toolbar;
  // delete imageview;
  delete capture;
  delete annotate;
}

//==============================================================================
class LoadingThread final : public juce::ThreadWithProgressWindow {
public:
  explicit LoadingThread(MainComponent *parent, std::string path)
    : juce::ThreadWithProgressWindow("Opening Image", true, true), parent(parent), path(path) {
    setStatusMessage("Opening image ...");
  }

  void run() override {
    setProgress(-1.0); // setting a value beyond the range 0 -> 1 will show a spinning bar..
    setStatusMessage("Reading Image");

    cv::Mat image_in = imread(path);
    std::cout << "Read OpenCV image: " << image_in.cols << "X" << image_in.rows << "\n";
    parent->imagePyramid.reset(new MRTiledImage());

    setStatusMessage("Building Hierarchy");

    unsigned int height = image_in.rows;
    unsigned int width = image_in.cols;

    parent->imagePyramid->bounds = cv::Rect_<float>(-5000, -5000, width, height);


    unsigned int tile_size = parent->imagePyramid->tile_size;

    double total_levels = ceil(max(log2(width), log2(height)) - log2(tile_size) + 1);
    double total_pixels = 0.0;
    for (unsigned int i = 0; i < total_levels; i++) {
      total_pixels += (width / (pow(2, i))) * (height / (pow(2, i)));
    }

    unsigned int num_levels = 1;
    std::shared_ptr<TiledImage> current = std::make_shared<TiledImage>(parent->imagePyramid, tile_size, tile_size, 0);
    setStatusMessage("Computing level 1");
    current->insertMat(image_in, cv::Rect_<float>(-5000, -5000, width, height));
    double pixels_processed = image_in.cols * image_in.rows;
    setProgress(pixels_processed / total_pixels);
    parent->imagePyramid->level.push_back(current);

    while (image_in.cols > tile_size || image_in.rows > tile_size) {
      std::shared_ptr<TiledImage> current = std::make_shared<TiledImage>(parent->imagePyramid, tile_size,
                                                                         tile_size * pow(2, num_levels), num_levels);
      setStatusMessage("Computing level " + juce::String(num_levels));
      //cv::resize causes a reallocation and is possibly better done with cv::pyrDown()
      cv::resize(image_in, image_in, cv::Size(image_in.cols / 2, image_in.rows / 2));
      pixels_processed += image_in.cols * image_in.rows;
      num_levels += 1;
      current->insertMat(image_in, cv::Rect_<float>(-5000, -5000, width, height));
      setProgress(pixels_processed / total_pixels);
      parent->imagePyramid->level.push_back(current);


      if (threadShouldExit())
        return;
    }
  }

  void threadComplete(bool userPressedCancel) override {
    if (userPressedCancel) { parent->MRimage.reset(new MRTiledImageSet()); } else {
      // parent->imageview->setImage(parent->MRimage);
      parent->capture->setImage(parent->MRimage);
      parent->annotate->setImage(parent->MRimage);
    }
    delete this;
  }

  MainComponent *parent;
  std::string path;
};

void MainComponent::setup_listbox() {
  labelList = std::make_unique<StreamCamLabelList>(sCam,
                                                   annotate,
                                                   [this](int idx, const juce::String &label,
                                                          bool matchedByAnnotation) {
                                                     Poco::FastMutex::ScopedLock lock(sCam->previousSlidesMutex);
                                                     if (idx < 0 || idx >= (int) sCam->previousSlides.size())
                                                       return;
                                                     auto mrImgSet = sCam->previousSlides[idx];
                                                     capture->setImage(mrImgSet);
                                                     annotate->setImage(mrImgSet);
                                                     annotate->update_active_annotations(idx);

                                                     // If matched by annotation, switch to annotate view and copy search text
                                                     if (matchedByAnnotation) {
                                                       GuiEventHandler("annotate");

                                                       // Copy search text from slide selector to annotation selector
                                                       if (labelList && annotate->leftComponent) {
                                                         juce::String searchText = labelList->getCurrentSearchText();
                                                         annotate->leftComponent->setSearchText(searchText);
                                                       }
                                                     }
                                                   });

  addAndMakeVisible(*labelList);
  labelList->setVisible(false);
}

void MainComponent::update() {
  if (clientHasData) {
    const MessageManagerLock mmLock;
    refreshImage();
    clientHasData = false;
  }
}


void MainComponent::loadImage(std::string path) {
  (new LoadingThread(this, path))->launchThread();

  //  cv::Mat cvimage = imread(path);
  //  std::cout << "Read OpenCV image: " << cvimage.cols << "X" << cvimage.rows << "\n";
  //  MRimage.reset(new MRTiledImage());
  //  MRimage->build(cvimage);
}


//==============================================================================
void MainComponent::paint(juce::Graphics &g) {
}

void MainComponent::resized() {
  const ScopedLock lock(mutex);
  juce::Rectangle<int> b = getLocalBounds();
  juce::Rectangle<int> fullBounds = b;
  toolbar->setBounds(b.removeFromTop(50));

  // Handle labelList visibility and positioning
  const int panelWidth = (labelList && labelList->isVisible()) ? 100 : 0;
  auto leftPanel = b.removeFromLeft(panelWidth);

  if (labelList) {
    labelList->setBounds(leftPanel);
  }

  // imageview->setBounds(b);
  capture->setBounds(b);
  annotate->setBounds(b);
  if (dirBrowser) {
    dirBrowser->setBounds(b);
    // Place the select button at the bottom-right, aligned with the path bar
    const int btnW = 80, btnH = 24, margin = 4;
    dirSelectButton.setBounds(b.getRight() - btnW - margin, b.getBottom() - btnH - margin, btnW, btnH);
    dirSelectButton.setVisible(dirBrowser->isVisible());
  }

  repositionSlideListButton();

  // Update labelList visibility - hide if recording/simulating
  if (labelList && (capture->recording || capture->simulating)) {
    labelList->setVisible(false);
  }
}

void MainComponent::repositionSlideListButton() {
  if (!slideListButton) return;

  ImageViewOverlay *overlay = nullptr;
  if (capture && capture->isVisible() && capture->controlsOverlay)
    overlay = capture->controlsOverlay.get();
  else if (annotate && annotate->isVisible() && annotate->rightComponent->controlsOverlay)
    overlay = annotate->rightComponent->controlsOverlay.get();

  if (overlay) {
    auto rLocal = overlay->getCenterButtonBoundsLocal();
    if (!rLocal.isEmpty()) {
      auto pInMain = overlay->localPointToGlobal(rLocal.getPosition());
      pInMain = getLocalPoint(nullptr, pInMain);
      juce::Rectangle<int> rInMain(pInMain.x, pInMain.y, rLocal.getWidth(), rLocal.getHeight());
      slideListButton->setBounds(rInMain.translated(0, rInMain.getHeight() + 8));
    }
  }

  bool shouldShow = (sCam != nullptr) &&
                    capture && !capture->recording &&
                    !capture->simulating &&
                    (sCam->get_num_slides() > 0);
  slideListButton->setVisible(shouldShow);
}

void MainComponent::buttonClicked(juce::Button *button) {
  if (button == slideListButton.get()) {
    if (labelList) {
      labelList->refresh();
      labelList->setVisible(!labelList->isVisible());
      resized(); // Update layout to account for labelList visibility change
    }
  }
  if (button == &dirSelectButton) {
    confirmDirectorySelection();
  }
}

void MainComponent::loadImageDialog(const FileChooser &fc) {
  File result = fc.getResult();
  if (result.exists()) {
    loadImage(result.getFullPathName().toStdString());
  }
}


void MainComponent::confirmDirectorySelection() {
  if (dirBrowser) {
    auto selected = dirBrowser->getSelectedFile(0);
    if (!selected.isDirectory())
      selected = dirBrowser->getRoot();
    if (cwd != selected) {
      cwd = selected;
      capture->save_slide_set();
      sCam->new_case_reset();
      sCam->givenWorkingDirectory = cwd.getFullPathName().toStdString();
      capture->setImage(nullptr);
      annotate->setImage(nullptr);
      findSlideDirectories();
      sCam->keepFrames = keepFrames;
    }
  }
}

void MainComponent::findSlideDirectories() {
  std::vector<juce::File> slideDirs;
  for (const auto &entry: juce::RangedDirectoryIterator(cwd, false, "*", juce::File::findDirectories)) {
    juce::File subdir = entry.getFile();
    if (subdir.getChildFile("slide.pcHdr").existsAsFile())
      slideDirs.push_back(subdir);
  }

  load_case(slideDirs);
}

void MainComponent::load_case(std::vector<juce::File> slideDirs) {
  if (slideDirs.empty()){return;}
  annotate->allSlideAnnotations.resize(slideDirs.size());

  for (auto& ptr : annotate->allSlideAnnotations) {
    ptr = std::make_shared<std::vector<std::shared_ptr<Annotation>>>();
  }
  {
    Poco::FastMutex::ScopedLock lock(sCam->previousSlidesMutex);
    for (auto &p :slideDirs) {
      auto slide = std::make_shared<MRTiledImageSet>();
      slide->index = sCam->previousSlides.size();
      slide->cwd = p.getFullPathName().toStdString();
      slide->read_slide_header();
      sCam->previousSlides.push_back(slide);
      load_annotations(p,slide->index);


      ++sCam->numSlides;
    }
  }

  // Switch to capture view first so components are visible
  GuiEventHandler("capture");

  // Set image after visible so zoomAndCenter works
  capture->setImage(sCam->previousSlides[0]);
  annotate->setImage(sCam->previousSlides[0]);
  capture->fixAspectRatio();

  // Ensure the slide label list is available
  if (!labelList)
    setup_listbox();
  if (labelList)
    labelList->refresh();

  resized();
}

void MainComponent::load_annotations(juce::File dir, int index) {
  auto dPath = dir.getChildFile("dictation.wav");
  if (dPath.existsAsFile()) {
    annotate->voiceAnnoOutstanding.push({index,dPath});
    annotate->newVoiceAnnotation.set();
  }
}

void MainComponent::GuiEventHandler(std::string event) {
  if (event == "open") {
    fc.reset(new FileChooser("Choose an image to open...", File::getCurrentWorkingDirectory(),
                             "*.png,*.jpeg,*.tiff"));

    fc->launchAsync(FileBrowserComponent::openMode
                    | FileBrowserComponent::canSelectFiles,
                    std::bind(&MainComponent::loadImageDialog, this, std::placeholders::_1));

    return;
  }

  if (event == "home") {
    {
      const ScopedLock lock(mutex);
      capture->setVisible(false);
      annotate->setVisible(false);
      if (dirBrowser) dirBrowser->setVisible(true);
    }
    resized();
  }

  if (event == "capture") {
    {
      const ScopedLock lock(mutex);
      capture->setVisible(true);
      capture->fixAspectRatio();
      annotate->setVisible(false);
      if (dirBrowser) dirBrowser->setVisible(false);
    }
    resized();
  }

  if (event == "annotate") {
    {
      const ScopedLock lock(mutex);
      annotate->setVisible(true);

      annotate->fixAspectRatio();
      capture->setVisible(false);
      if (dirBrowser) dirBrowser->setVisible(false);
    }
    resized();
  }
}
