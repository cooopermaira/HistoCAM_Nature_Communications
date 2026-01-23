#include "JuceHeader.h"

static int wrapMod(int x, int m) {
  // result in [0, m-1] even if x is negative
  return (x % m + m) % m;
}

//==============================================================================
ImageViewComponent::ImageViewComponent(std::shared_ptr<fRectangle> _view,
                                       StringArray &iconNames,
                                       OwnedArray<Drawable> &iconsFromZipFile, MainComponent *parent) : parent(parent),
  MRImageSet(NULL),
  view(_view),
  loadMRImageSetASAPEvent(Poco::Event::EVENT_AUTORESET),
  shadeLevels(
    false) {
  setOpaque(true); //telling juce that there is nothingi to render underneath

  controlsOverlay.reset(new ImageViewOverlay(this, iconNames, iconsFromZipFile));
  addAndMakeVisible(controlsOverlay.get());

  juce::Rectangle<int> b = getLocalBounds();

  MRImageSet = NULL;

  horizontalScrollBar.addListener(this);
  verticalScrollBar.addListener(this);

  // Set the initial positions and visibility of the scroll bars
  horizontalScrollBar.setCurrentRange(0, 400);
  verticalScrollBar.setCurrentRange(0, 400);

  horizontalScrollBar.setSingleStepSize(10);
  verticalScrollBar.setSingleStepSize(10);

  addChildComponent(horizontalScrollBar);
  addChildComponent(verticalScrollBar);

  shadeClasses = false;


  //gl.attachTo(*this);
}

ImageViewComponent::~ImageViewComponent() {
}

void ImageViewComponent::uncacher() {
  while (cacherKeepGoing) {
    if (parent->sCam) {
      {
        Poco::FastMutex::ScopedLock lock(loadASAPMutex);
        while (!loadMRImageSetsASAP.empty()) {
          auto slide = loadMRImageSetsASAP.front();
          loadMRImageSetsASAP.pop();
          slide->uncache_from_disk();
          slide->loadFromCacheQueued = false;
        }
      }
      loadMRImageSetASAPEvent.wait();
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  }
}

void ImageViewComponent::cacher() {
  while (cacherKeepGoing) {
    if (parent->sCam) {
      while (recentlyViewedSlides.size() > 1) {
        if (std::shared_ptr<MRTiledImageSet> mrImgSet; recentlyViewedSlides.pop(mrImgSet)) {
          mrImgSet->cache_to_disk();
        }
      }
      parent->sCam->cacheAlert.wait();
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  }
}

void ImageViewComponent::q_cache()  {
  if (!MRImageSet->inMemory && !MRImageSet->loadFromCacheQueued) {
    Poco::FastMutex::ScopedLock lock(loadASAPMutex);
    loadMRImageSetsASAP.push(MRImageSet);
    loadMRImageSetASAPEvent.set();
    MRImageSet->loadFromCacheQueued = true;

    if (MRImageSet->inMemory) {
      int k = 0;
    }
    recentlyViewedSlides.push_unique(MRImageSet);
    parent->sCam->cacheAlert.set();
  }
}

void ImageViewComponent::drawLayer(Graphics &g, float scale, std::shared_ptr<MRTiledImage> tiledImage) {
  //convert bounds from view space to image space
  auto imageview = *view;

  if (tiledImage->scale == 0) {
    if (auto set = tiledImage->MRImageSet.lock()) {
      auto ans = set->get_display_coords_for_zero_scale(tiledImage);
      imageview -= fPoint(ans.x, ans.y);
    } else {
      // MRImageSet no longer exists (shouldn't happen if set owns tiles), but handle safely anyway.
      throw std::runtime_error(
        "MRImageSet no longer exists, but ImageViewComponent is trying to access it in drawLayer");
    }
  } else {
    imageview *= 1.0 / tiledImage->scale;
    imageview -= fPoint(tiledImage->offset.x, tiledImage->offset.y);
  }

  //query tiles within image space bounds
  std::vector<TileQuery> tiles = tiledImage->
      getTiles(RectJtoC(imageview), RectJtoC(getLocalBounds()));

  //draw each tile
  for (unsigned int t = 0; t < tiles.size(); t++) {
    auto tile = tiles[t].image;
    tile->mutex.lock();

    auto bounds = RectCtoJ<float>(tiles[t].bounds);
    bounds *= view2screenScale(imageview) * scale;
    bounds.expand(0.5, 0.5);
    tiles[t].bounds = RectJtoC<float>(bounds);

    if (tile->image.data) {
      if (!tile->usingPreferred) {
        juce::Image *im = new juce::Image(juce::Image::ARGB, tile->image.cols, tile->image.rows, true);
        tile->preferredObj = im;
        tile->destroyPreferredObj = [](void *p) {
          delete static_cast<juce::Image *>(p);
        };
        tile->usingPreferred = true;
      }

      auto im = static_cast<juce::Image *>(tile->preferredObj);

      if (tile->newData) {
        auto img = tile->image;
        juce::Image::BitmapData bitmap_data(*im, juce::Image::BitmapData::ReadWriteMode::writeOnly);
        CHECK_CUDA(cudaMemcpy2D(bitmap_data.data, 4 * img.cols, img.data,
          img.step, 4 * img.cols, img.rows, cudaMemcpyDeviceToHost));
        tile->newData = false;
      }
      if (tile->newAnnoData) {
        for (auto &kv: tile->SAMMasks) {
          if (!kv.second.second) {
            kv.second.second = new juce::Image(juce::Image::SingleChannel, tile->image.cols, tile->image.rows, true);
          }
          auto annoMask = static_cast<juce::Image *>(kv.second.second);
          juce::Image::BitmapData bitmap_data(*annoMask, juce::Image::BitmapData::ReadWriteMode::writeOnly);
          auto img = kv.second.first;
          CHECK_CUDA(cudaMemcpy2D(bitmap_data.data,img.cols,img.data,img.step,img.cols,img.rows,cudaMemcpyDeviceToHost))
          ;
        }
      }
      g.setOpacity(1.f);
      g.drawImage(*im, bounds);
      //
      // //draw tile bounds with owner frame
      // g.setColour(juce::Colours::greenyellow);
      // g.drawRect(bounds, 3);
      //
      // std::string ij;
      // if (tile->owner) {
      //   ij = Poco::format("(%ld,%f)", tile->owner->index, static_cast<double>(tile->owner->motionBlur));
      // }
      // g.setFont(20);
      // g.drawText(ij, bounds.getCentreX() - 250,
      //            bounds.getCentreY() - 15, 500, 30, Justification::centred);

      for (auto &mask: tile->SAMMasks) {
        auto jImg = static_cast<juce::Image *>(mask.second.second);
        g.saveState();
        g.setOpacity(0.5f);

        const float sx = bounds.getWidth() / (float) jImg->getWidth();
        const float sy = bounds.getHeight() / (float) jImg->getHeight();

        AffineTransform maskToCanvas = AffineTransform::scale(sx, sy).translated(bounds.getX(), bounds.getY());

        auto annoColor = (*parent->annotate->annotations.get())[mask.first].get()->getColor();
        //Colour annoColor = parent->annotate

        g.reduceClipRegion(*jImg, maskToCanvas);
        g.setColour(annoColor.withAlpha(0.5f));
        g.fillAll();
        g.restoreState();
      }


      if (shadeLevels) {
        Graphics::ScopedSaveState save(g);

        const float sx = bounds.getWidth() / (float) im->getWidth();
        const float sy = bounds.getHeight() / (float) im->getHeight();
        AffineTransform imgToCanvas = AffineTransform::scale(sx, sy).translated(bounds.getX(), bounds.getY());

        g.reduceClipRegion(*im, imgToCanvas);

        int maglab = tiledImage->magLabel;
        auto color = levelColors[4 - maglab];
        auto overlayColor = Colour(color.getRed(), color.getGreen(), color.getBlue(), (uint8) 100);

        g.setColour(overlayColor);
        g.fillRect(bounds);
      }
    }
    tile->mutex.unlock();
  }

  //tile classification
  /*
    if (MRImage->images.size() > 0 && shadeClasses) {
      auto sCam = MRImage->images[0]->parent;

      auto baseTiles = tiledImage->
          getTiles(RectJtoC(imageview), RectJtoC(getLocalBounds()), true);

      for (auto tile: baseTiles) {
        if (!tile.image->image.data) { continue; }
        //get draw bounds of base level tile
        auto bounds = RectCtoJ<float>(tile.bounds);
        bounds *= view2screenScale(imageview) * scale;
        bounds.expand(0.5, 0.5);
        tile.bounds = RectJtoC<float>(bounds);

        //get class for color
        auto tileCoords = std::tuple<int, int, unsigned>(tile.i, tile.j, i);
        int classScore = tiledImage->get_class_for_tile(tileCoords);
        if (classScore == -1) { continue; }

        //draw it
        Colour tileColor;
        if (classScore == 0) {
          tileColor = Colour(uint8(50), 50, 50, uint8(20));
        } else {
          auto tileClassInfo = sCam->classesInfo[classScore - 1];
          tileColor = Colour(uint8(tileClassInfo.r), tileClassInfo.g, tileClassInfo.b, uint8(100));
        }

        g.setColour(tileColor);
        g.fillRect(bounds);
      }
    }
    */
}

void ImageViewComponent::drawSlide(Graphics &g, float scale) {
  bool canShadeClasses = false;

  std::vector<std::shared_ptr<MRTiledImage>> mrImages;
  {
    Poco::FastMutex::ScopedLock lock(MRImageSet->mutex);
    mrImages = MRImageSet->MRImages;
  }

  const int N = (int) mrImages.size();
  if (N == 0) return;

  const int mode = wrapMod(componentSelector, N + 1); // 0..N
  const bool showAll = (mode == 0);
  const int selectedIdx = showAll ? -1 : (mode - 1); // 0..N-1

  if (!showAll) {
    g.beginTransparencyLayer(0.3f);
  }

  for (int i = 0; i < N; ++i) {
    if (!showAll && i == selectedIdx) continue;

    auto img = mrImages[i];
    if (/*img->scale == 0 ||*/ img->suspended) continue;

    g.setColour(juce::Colours::white);
    drawLayer(g, scale, img);


    // //draws grid on image with indexes
    //   for (unsigned int t = 0; t < tiles.size(); t++) {
    //       auto bounds = RectCtoJ<float>(tiles[t].bounds) * scale;
    //       g.setColour(juce::Colours::greenyellow);
    //       g.drawRect(bounds, 3);
    //       std::string ij = Poco::format("(%i,%i)", tiles[t].i, tiles[t].j);
    //       g.setFont(20);
    //       g.drawText(ij, bounds.getCentreX() - 50,
    //                  bounds.getCentreY() - 45, 100, 30, Justification::centred);
    //    }
  }

  if (!showAll) {
    g.endTransparencyLayer();

    // draw selected at full opacity
    auto img = mrImages[selectedIdx];
    if (img->scale != 0 && !img->suspended) {
      g.setColour(juce::Colours::white);
      drawLayer(g, scale, img);

      if (!parent->sCam->microscopeInput) {
        g.setColour(juce::Colours::red);

        g.setFont(15);
        g.drawText("current objective", 5, getHeight() - 30, 110,
                   Justification::centredLeft, true);

        g.setFont(40.0f);
        g.drawText(pathCam::Image::get_label(img->magLabel),
                   20, getHeight() - 50, 100,
                   Justification::centredLeft, true);
      }
    }
  }


  //Define buffer space from the edges
  if (mrImages.size() > 0 && shadeClasses) {
    auto sCam = mrImages[0]->parent;
    int paddingX = 100;
    int paddingY = 150;
    int squareSize = 30; // Size of the square
    int textPadding = 10; // Space between the square and the text
    int linePadding = 10; // Space between lines


    // Loop through the dynamically loaded classes
    int i = 0;
    do {
      int xPosition = getWidth() - paddingX - squareSize - textPadding;
      int yPosition = getHeight() - paddingY - i * squareSize - i * linePadding;

      // Extract color and name from ClassInfo
      Colour color(uint8(sCam->classesInfo[i].r), sCam->classesInfo[i].g, sCam->classesInfo[i].b);

      // Set the color for the square
      g.setColour(color);
      g.fillRect(xPosition, yPosition, squareSize, squareSize);

      // Set text color and font
      g.setColour(Colours::black);
      g.setFont(Font(24.0));

      // Draw class label
      g.drawText(sCam->classesInfo[i].name, xPosition + squareSize + textPadding, yPosition, 100,
                 squareSize,
                 Justification::centredLeft, true);
      i++;
    } while (i < sCam->classesInfo.size());

    //draw title
    int xPosition = getWidth() - paddingX - squareSize - textPadding;
    int yPosition = getHeight() - paddingY - i * squareSize - i * linePadding;

    // Set text color and font
    g.setColour(Colours::black);
    g.setFont(Font(24.0));

    // Draw class label
    g.drawText(sCam->classes_title, xPosition + squareSize + textPadding, yPosition, 100,
               squareSize,
               Justification::centredLeft, true);
  }
}


void ImageViewComponent::refreshImage() {
  const ScopedLock lock(mutex);
  zoomAndCenter();
  repaint();
}

void ImageViewComponent::setImage(std::shared_ptr<MRTiledImageSet> image) {
  const ScopedLock lock(mutex);

  MRImageSet = image;
  if (!image) { return; }

  horizontalScrollBar.setRangeLimits(MRImageSet->bounds.x, MRImageSet->bounds.width);
  verticalScrollBar.setRangeLimits(MRImageSet->bounds.y, MRImageSet->bounds.height);

  horizontalScrollBar.setVisible(true);
  verticalScrollBar.setVisible(true);

  zoomAndCenter();
  repaint();
}

void ImageViewComponent::mouseDown(const juce::MouseEvent &event) {
  if (event.mods.isLeftButtonDown()) {
    lastMousePosition = event.getPosition();
  }
}

/*
void ImageViewComponent::mouseDrag(const juce::MouseEvent &event) {
  if (event.mods.isLeftButtonDown()) {
    juce::Point<int> idelta = event.getPosition() - lastMousePosition;
    fPoint delta = fPoint(idelta.x, idelta.y) * screen2viewScale(*view);
    translate(-delta);
    lastMousePosition = event.getPosition();
    repaint();
  }
}
*/

void ImageViewComponent::mouseDrag(const juce::MouseEvent &event) {
  if (event.mods.isLeftButtonDown()) {
    juce::Point<int> idelta = event.getPosition() - lastMousePosition;
    fPoint delta = fPoint(idelta.x, idelta.y) * screen2viewScale(*view);
    translate(-delta);
    lastMousePosition = event.getPosition();

    // FRAME RATE LIMITING HERE
    auto now = juce::Time::getCurrentTime();
    if ((now - lastRepaintTime).inMilliseconds() >= MIN_REPAINT_INTERVAL_MS) {
      repaint();
      lastRepaintTime = now;
    }
    // Mouse position is still updated, just fewer repaints
  }
}


void ImageViewComponent::mouseWheelMove(const MouseEvent &event, const MouseWheelDetails &wheel) {
  scaleCenter(fPoint(1.0 - wheel.deltaY, 1.0 - wheel.deltaY));
  repaint();
}

void ImageViewComponent::mouseMagnify(const MouseEvent &, float magnifyAmmount) {
  scaleCenter(fPoint(1.0 / magnifyAmmount, 1.0 / magnifyAmmount));
  repaint();
}

void ImageViewComponent::adjust_MRImageSet(int mode) {
  const bool showCurrent = (mode == 0);
  const int selectedIdx = showCurrent ? -1 : (mode - 1); // 0..N-1
  std::shared_ptr<MRTiledImageSet> mrImgSet;
  if (showCurrent) {
    mrImgSet = parent->sCam->get_MRimage_reference();
    // parent->imageview->setImage(parent->sCam->get_MRimage_reference());
    // parent->capture->setImage(parent->sCam->get_MRimage_reference());
    // parent->annotate->setImage(parent->sCam->get_MRimage_reference());
  } else {
    mrImgSet = parent->sCam->previousSlides[selectedIdx];
    // parent->imageview->setImage(parent->sCam->previousSlides[selectedIdx]);
    // parent->capture->setImage(parent->sCam->previousSlides[selectedIdx]);
    // parent->annotate->setImage(parent->sCam->previousSlides[selectedIdx]);
  }
  setImage(mrImgSet);
  parent->annotate->setImage(mrImgSet);
}

bool ImageViewComponent::keyPressed(const juce::KeyPress &key, juce::Component *originatingComponent) {
  if (key.getTextCharacter() == '>') {
    //cant change to past slide while compositing
    if (parent->capture->simulating || parent->capture->recording) { return true; }

    if (parent->sCam) {
      Poco::FastMutex::ScopedLock lock(parent->sCam->previousSlidesMutex);

      const int N = static_cast<int>(parent->sCam->previousSlides.size());

      if (N == 0) { return true; }
      int mode = wrapMod(MRImageSetSelector + 1, N + 1);
      if (mode == 0 && !parent->sCam->MRImageSet) {
        mode = wrapMod(mode + 1, N + 1);
      }
      MRImageSetSelector = mode;
      adjust_MRImageSet(mode);
    }
    return true;
  }

  if (key.getTextCharacter() == '<') {
    //cant change to past slide while compositing
    if (parent->capture->simulating || parent->capture->recording) { return true; }

    if (parent->sCam) {
      Poco::FastMutex::ScopedLock lock(parent->sCam->previousSlidesMutex);

      const int N = static_cast<int>(parent->sCam->previousSlides.size());

      if (N == 0) { return true; }
      int mode = wrapMod(MRImageSetSelector - 1, N + 1);
      if (mode == 0 && !parent->sCam->MRImageSet) {
        mode = wrapMod(mode - 1, N + 1);
      }
      MRImageSetSelector = mode;
      adjust_MRImageSet(mode);
    }
    return true;
  }

  if (key == juce::KeyPress::createFromDescription("-")) {
    scaleCenter(fPoint(2.0, 2.0));
    repaint();
    return true; // Key press handled
  }
  if (key == juce::KeyPress::createFromDescription("=")) {
    //Really "+"
    scaleCenter(fPoint(0.5, 0.5));
    repaint();
    return true; // Key press handled
  }
  if (key == juce::KeyPress::createFromDescription("a")) {
    shadeClasses = !shadeClasses;
    repaint();
    return true;
  }
  if (key == KeyPress::createFromDescription("c")) {
    const int N = (int) MRImageSet->MRImages.size();
    if (N == 0) return true;

    const int M = N + 1; // 0..N  (0 = show all)
    int mode = wrapMod(componentSelector + 1, M);

    // If we're in "show one component" mode, skip suspended components.
    if (mode != 0) {
      int guard = 0;
      while (guard++ < M) {
        const int idx = mode - 1; // 0..N-1
        if (!MRImageSet->MRImages[idx]->suspended) break;

        mode = wrapMod(mode + 1, M); // advance within 0..N
        if (mode == 0) {
          // landed on "show all" -> always allowed
          break;
        }
      }

      // If we failed to find a non-suspended component (all suspended), show all.
      if (guard >= M && mode != 0) mode = 0;
    }

    componentSelector = mode;
    repaint();
    return true;
  }

  if (key == juce::KeyPress::createFromDescription("s")) {
    shadeLevels = !shadeLevels;
    repaint();
    return true; // Key press handled
  }
  // // segment anything
  // if (key == juce::KeyPress::createFromDescription("q")) {
  //   if (!MRImageSet->MRImages.empty()) {
  //     //MRImage->images[0]->parent->as->create_segmentation()
  //   }
  // }
  if (key.getKeyCode() == KeyPress::escapeKey) {
    JUCEApplication::getInstance()->systemRequestedQuit();
  }
  return false; // Key press not handled
}

void ImageViewComponent::updateScrollbar() {
  if (!view->isEmpty()) {
    horizontalScrollBar.setCurrentRangeStart(view->getCentreX());
    verticalScrollBar.setCurrentRangeStart(view->getCentreY());
  }
}


void ImageViewComponent::scrollBarMoved(juce::ScrollBar *scrollBar, double newRangeStart) {
  // This method is called when the scroll bar is moved
  if (scrollBar == &horizontalScrollBar) {
    if (isVisible()) { view->setCentre(newRangeStart, view->getCentreY()); }
    repaint();
  } else if (scrollBar == &verticalScrollBar) {
    if (isVisible()) { view->setCentre(view->getCentreX(), newRangeStart); }
    repaint();
  }
}


//==============================================================================
void ImageViewComponent::paint(juce::Graphics &g) {
  g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));

  g.drawImageAt(checkerboard, 0, 0);
  //g.fillAll(juce::Colours::white);

  if (MRImageSet) {
    drawSlide(g, 1.0);
  }


  if (shadeLevels) {
    // Define the buffer space from the edges
    int padding = 100;

    int squareSize = 30; // Size of the square
    int textPadding = 10; // Space between the square and the text
    int linePadding = 10; // Space between lines

    std::vector<std::string> objectives = {"20x", "10x", "4x", "2x"};


    for (unsigned int i = 0; i < 4; i++) {
      int xPosition = getWidth() - padding - squareSize - textPadding;
      int yPosition = getHeight() - padding - (i * squareSize) - (i * linePadding);

      g.setColour(levelColors[i]);
      g.fillRect(xPosition, yPosition, squareSize, squareSize);

      g.setColour(Colours::black);
      g.setFont(Font(24.0));
      g.drawText(objectives[i], xPosition + squareSize + textPadding, yPosition, 40, squareSize,
                 Justification::centredLeft, true);
    }
  }
}


void ImageViewComponent::resized() {
  // This is called when the ImageViewComponent is resized.
  // If you add any child components, this is where you should
  // update their positions.
  const ScopedLock lock(mutex);

  juce::Rectangle<int> b = getLocalBounds();

  horizontalScrollBar.setBounds(b.removeFromBottom(20));
  verticalScrollBar.setBounds(b.removeFromRight(20));

  controlsOverlay->setBounds(juce::Rectangle<int>(20, 20, 60, 60));


  b = getLocalBounds();

  if (MRImageSet && isVisible()) {
    scaleCenter(fPoint((float) b.getHorizontalRange().getLength() /
                       (float) old_bounds.getHorizontalRange().getLength(),
                       (float) b.getVerticalRange().getLength() /
                       (float) old_bounds.getVerticalRange().getLength()));
  }

  old_bounds = getLocalBounds();


  checkerboard = createCheckerboardImage(getLocalBounds().getWidth(),
                                         getLocalBounds().getHeight(),
                                         64,
                                         juce::Colours::lightgrey,
                                         juce::Colours::white);
}

void ImageViewComponent::zoomAndCenter() {
  bool isEmpty;
  {
    Poco::FastMutex::ScopedLock lock(MRImageSet->mutex);
    isEmpty = MRImageSet->MRImages.empty();
  }
  if (!MRImageSet || isEmpty || !isVisible()) { return; }

  //  Rect_<float> bounds;
  //  bool showAsCircle;
  //  int component;
  //  std::string magLabel;
  //  parent->capture->sCam->get_last_frame(bounds, showAsCircle, component, magLabel);
  //  auto lastComponentImg = MRImage->images[component];

  //  horizontalScrollBar.setRangeLimits((*lastComponentImg).scale * (*lastComponentImg).offset.x + (*lastComponentImg).bounds.x, (*lastComponentImg).scale * (*lastComponentImg).bounds.width);
  //  verticalScrollBar.setRangeLimits((*lastComponentImg).scale * (*lastComponentImg).offset.y + (*lastComponentImg).bounds.y, (*lastComponentImg).scale * (*lastComponentImg).bounds.height);
  //
  //  horizontalScrollBar.setVisible(true);
  //  verticalScrollBar.setVisible(true);

  juce::Rectangle<int> b = getLocalBounds();
  *view = fRectangle(b.getX(), b.getY(), b.getWidth(), b.getHeight());


  //  auto compBounds = Rect_<float>((*lastComponentImg).scale * (*lastComponentImg).offset.x + (*lastComponentImg).bounds.x,
  //                             (*lastComponentImg).scale * (*lastComponentImg).offset.y + (*lastComponentImg).bounds.y,
  //                             (*lastComponentImg).scale * (*lastComponentImg).bounds.width,
  //                             (*lastComponentImg).scale * (*lastComponentImg).bounds.height);
  //  //view->setCentre(RectCtoJ((*lastComponentImg).bounds).getCentre());
  //  view->setCentre(RectCtoJ(compBounds).getCentre());
  //
  //  float scale = max((float) (*lastComponentImg).scale * (*lastComponentImg).bounds.width /
  //                    (float) view->getHorizontalRange().getLength(),
  //                    (float) (*lastComponentImg).scale * (*lastComponentImg).bounds.height /
  //                    (float) view->getVerticalRange().getLength());
  //
  //  scaleCenter(fPoint(scale, scale));

  view->setCentre(RectCtoJ(MRImageSet->bounds).getCentre());

  float scale = max((float) MRImageSet->bounds.width /
                    (float) view->getHorizontalRange().getLength(),
                    (float) MRImageSet->bounds.height /
                    (float) view->getVerticalRange().getLength());

  scaleCenter(fPoint(scale, scale),false);
}

juce::Image ImageViewComponent::createCheckerboardImage(int width,
                                                        int height,
                                                        int squareSize,
                                                        juce::Colour colour1,
                                                        juce::Colour colour2) {
  juce::Image checkerboard(juce::Image::PixelFormat::RGB, width, height, true);

  juce::Graphics g(checkerboard);

  for (int y = 0; y < height; y += squareSize) {
    for (int x = 0; x < width; x += squareSize) {
      // Determine the color based on the position
      bool isColour1 = ((x / squareSize) % 2 == 0) ^ ((y / squareSize) % 2 == 0);
      g.setColour(isColour1 ? colour1 : colour2);

      // Draw the square
      g.fillRect(x, y, squareSize, squareSize);
    }
  }

  return checkerboard;
}
