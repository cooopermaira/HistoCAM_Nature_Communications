#include "JuceHeader.h"


//==============================================================================
ImageViewComponent::ImageViewComponent(std::shared_ptr<fRectangle> view,
                                       StringArray &iconNames,
                                       OwnedArray<Drawable> &iconsFromZipFile, MainComponent *parent) : parent(parent),
  MRImage(NULL),
  view(view),
  shadeLevels(
    false) {
  setOpaque(true); //telling juce that there is nothingi to render underneath

  controlsOverlay.reset(new ImageViewOverlay(this, iconNames, iconsFromZipFile));
  addAndMakeVisible(controlsOverlay.get());

  juce::Rectangle<int> b = getLocalBounds();

  MRImage = NULL;

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

void ImageViewComponent::refreshImage() {
  const ScopedLock lock(mutex);
  zoomAndCenter();
  repaint();
}

void ImageViewComponent::setImage(std::shared_ptr<MRTiledImageSet> image) {
  const ScopedLock lock(mutex);

  MRImage = image;

  horizontalScrollBar.setRangeLimits(MRImage->bounds.x, MRImage->bounds.width);
  verticalScrollBar.setRangeLimits(MRImage->bounds.y, MRImage->bounds.height);

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


bool ImageViewComponent::keyPressed(const juce::KeyPress &key, juce::Component *originatingComponent) {
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
  if (key == juce::KeyPress::createFromDescription("s")) {
    shadeLevels = !shadeLevels;
    repaint();
    return true; // Key press handled
  }
  if (key==juce::KeyPress::createFromDescription("q")) {
    if (!MRImage->images.empty()) {
      //MRImage->images[0]->parent->as->create_segmentation()
    }
  }
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

void ImageViewComponent::drawSlide(juce::Graphics &g, float scale) {
  bool canShadeClasses = false;

  for (unsigned int i = 0; i < MRImage->images.size(); i++) {
    g.setColour(juce::Colours::white);

    if (MRImage->images[i]->scale == 0 || MRImage->images[i]->suspended) { continue; }

    //convert bounds from view space to image space
    auto imageview = *view;
    imageview *= 1.0 / MRImage->images[i]->scale;
    imageview -= fPoint(MRImage->images[i]->offset.x, MRImage->images[i]->offset.y);

    //query tiles within image space bounds
    std::vector<TileQuery> tiles = MRImage->images[i]->
        getTiles(RectJtoC(imageview), RectJtoC(getLocalBounds()));

    //draw each tile
    for (unsigned int t = 0; t < tiles.size(); t++) {
      auto tile = tiles[t].image;
      auto bounds = RectCtoJ<float>(tiles[t].bounds);
      bounds *= view2screenScale(imageview) * scale;
      bounds.expand(0.5, 0.5);
      tiles[t].bounds = RectJtoC<float>(bounds);

      if (tile->image.data) {
        if (!tile->usingPreferred) {
          juce::Image *im = new juce::Image(juce::Image::ARGB, tile->image.cols, tile->image.rows, true);
          tile->preferredObj = im;
          tile->destroyPreferredObj = [](void* p) {
            delete static_cast<juce::Image*>(p);
          };
          tile->usingPreferred = true;
        }

        auto im = static_cast<juce::Image *>(tile->preferredObj);
        tile->mutex.lock();

        if (tile->newData) {
          auto img = tile->image;
          juce::Image::BitmapData bitmap_data(*im, juce::Image::BitmapData::ReadWriteMode::writeOnly);
          CHECK_CUDA(cudaMemcpy2D(bitmap_data.data, 4 * img.cols, img.data,
                       img.step, 4 * img.cols, img.rows, cudaMemcpyDeviceToHost));
          tile->newData = false;
        }
        if (tile->newAnnoData) {
          for (auto &kv : tile->SAMMasks) {
            if (!kv.second.second) {
              kv.second.second = new juce::Image(juce::Image::SingleChannel, tile->image.cols, tile->image.rows, true);
            }
            auto annoMask = static_cast<juce::Image *>(kv.second.second);
            juce::Image::BitmapData bitmap_data(*annoMask,juce::Image::BitmapData::ReadWriteMode::writeOnly);
            auto img = kv.second.first;
            CHECK_CUDA(cudaMemcpy2D(bitmap_data.data,img.cols,img.data,img.step,img.cols,img.rows,cudaMemcpyDeviceToHost));
          }
        }
        g.setOpacity(1.f);
        g.drawImage(*im, bounds);

        //draw tile bounds with owner frame
        g.setColour(juce::Colours::greenyellow);
        g.drawRect(bounds, 3);

        std::string ij;
        if (tile->owner) {
          ij = Poco::format("(%ld,%f)", tile->owner->index, static_cast<double>(tile->motionBlur));
        }
        g.setFont(20);
        g.drawText(ij, bounds.getCentreX() - 250,
                   bounds.getCentreY() - 15, 500, 30, Justification::centred);

        for (auto & mask : tile->SAMMasks) {
          auto jImg = static_cast<juce::Image*>(mask.second.second);
          g.saveState();
          g.setOpacity(0.5f);

          const float sx = bounds.getWidth()  / (float) jImg->getWidth();
          const float sy = bounds.getHeight() / (float) jImg->getHeight();

          AffineTransform maskToCanvas = AffineTransform::scale(sx, sy).translated(bounds.getX(), bounds.getY());

          auto annoColor = (*parent->annotate->annotations.get())[mask.first].get()->getColor();
          //Colour annoColor = parent->annotate

          g.reduceClipRegion(*jImg,maskToCanvas);
          g.setColour(annoColor.withAlpha(0.5f));
          g.fillAll();
          g.restoreState();
        }

        tile->mutex.unlock();

        if (shadeLevels) {
          Graphics::ScopedSaveState save(g);

          const float sx = bounds.getWidth()  / (float) im->getWidth();
          const float sy = bounds.getHeight() / (float) im->getHeight();
          AffineTransform imgToCanvas = AffineTransform::scale(sx, sy).translated(bounds.getX(), bounds.getY());

          g.reduceClipRegion(*im, imgToCanvas);

          int maglab = MRImage->images[i]->magLabel;
          auto color = levelColors[4 - maglab];
          auto overlayColor = Colour(color.getRed(), color.getGreen(), color.getBlue(), (uint8)100);

          g.setColour(overlayColor);
          g.fillRect(bounds);

        }
      }
    }

    //tile classification
    if (MRImage->images.size() > 0 && shadeClasses) {
      auto sCam = MRImage->images[0]->parent;

      auto baseTiles = MRImage->images[i]->
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
        int classScore = MRImage->images[i]->get_class_for_tile(tileCoords);
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
  //Define buffer space from the edges
  if (MRImage->images.size() > 0 && shadeClasses) {
    auto sCam = MRImage->images[0]->parent;
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


//==============================================================================
void ImageViewComponent::paint(juce::Graphics &g) {
  g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));

  g.drawImageAt(checkerboard, 0, 0);
  //g.fillAll(juce::Colours::white);

  if (MRImage) {
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

  if (MRImage && isVisible()) {
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
  if (!MRImage || MRImage->images.empty() || !isVisible()) { return; }

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

  view->setCentre(RectCtoJ(MRImage->bounds).getCentre());

  float scale = max((float) MRImage->bounds.width /
                    (float) view->getHorizontalRange().getLength(),
                    (float) MRImage->bounds.height /
                    (float) view->getVerticalRange().getLength());

  scaleCenter(fPoint(scale, scale));
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
