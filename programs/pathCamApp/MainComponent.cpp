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
                                                   [this](int idx, const juce::String &label) {
                                                     Poco::FastMutex::ScopedLock lock(sCam->previousSlidesMutex);
                                                     if (idx < 0 || idx >= (int) sCam->previousSlides.size())
                                                       return;
                                                     auto mrImgSet = sCam->previousSlides[idx];
                                                     capture->setImage(mrImgSet);
                                                     annotate->setImage(mrImgSet);
                                                     annotate->update_active_annotations(idx);
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

  // Position slideListButton as floating overlay below the centerButton
  // controlsOverlay is at (panelWidth + 20, 20) with centerButton at top 60px
  if (slideListButton) {
    // int xPos = panelWidth + 20; // Same x as controlsOverlay
    // int yPos = 50 + 20 + 60; // toolbar(50) + overlay_offset(20) + centerButton_height(60)
    // slideListButton->setBounds(xPos, yPos, 60, 60);
    if (capture->controlsOverlay) {
      ImageViewOverlay *overlay = nullptr;
      if (capture->isVisible()) {
        overlay = capture->controlsOverlay.get();
      } else if (annotate->isVisible()) {
        overlay = annotate->rightComponent->controlsOverlay.get();
      }

      if (overlay) {
        // 1) Get center button bounds in overlay-local coords
        auto rLocal = overlay->getCenterButtonBoundsLocal();
        if (!rLocal.isEmpty()) {
          // 2) Convert overlay-local rect -> MainComponent-local rect
          // (convert top-left point, keep same size)
          auto pInMain = overlay->localPointToGlobal(rLocal.getPosition());
          pInMain = this->getLocalPoint(nullptr, pInMain); // global -> MainComponent local

          juce::Rectangle<int> rInMain(pInMain.x, pInMain.y, rLocal.getWidth(), rLocal.getHeight());

          // 3) Place slideListButton directly below center button, same size
          const int gap = 8; // whatever spacing you want
          auto slideRect = rInMain.translated(0, rInMain.getHeight() + gap);

          slideListButton->setBounds(slideRect);

          // visibility logic unchanged
          bool shouldShowButton = (sCam != nullptr) &&
                                  !capture->recording &&
                                  !capture->simulating &&
                                  (sCam->get_num_slides() > 0);
          slideListButton->setVisible(shouldShowButton);
        }
      }
    }


    // Update slideListButton visibility based on state
    bool shouldShowButton = (sCam != nullptr) &&
                            !capture->recording &&
                            !capture->simulating &&
                            (sCam->get_num_slides() > 0);
    slideListButton->setVisible(shouldShowButton);
  }

  // Update labelList visibility - hide if recording/simulating
  if (labelList && (capture->recording || capture->simulating)) {
    labelList->setVisible(false);
  }
}

void MainComponent::buttonClicked(juce::Button *button) {
  if (button == slideListButton.get()) {
    if (labelList) {
      labelList->refresh();
      labelList->setVisible(!labelList->isVisible());
      resized(); // Update layout to account for labelList visibility change
    }
  }
}

void MainComponent::loadImageDialog(const FileChooser &fc) {
  File result = fc.getResult();
  if (result.exists()) {
    loadImage(result.getFullPathName().toStdString());
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
    const ScopedLock lock(mutex);
    // imageview->setVisible(true);
    // imageview->fixAspectRatio();
    capture->setVisible(false);
    annotate->setVisible(false);
  }

  if (event == "capture") {
    const ScopedLock lock(mutex);
    capture->setVisible(true);
    capture->fixAspectRatio();
    // imageview->setVisible(false);
    annotate->setVisible(false);
  }

  if (event == "annotate") {
    const ScopedLock lock(mutex);
    annotate->setVisible(true);
    annotate->fixAspectRatio();
    // imageview->setVisible(false);
    capture->setVisible(false);
  }
}
