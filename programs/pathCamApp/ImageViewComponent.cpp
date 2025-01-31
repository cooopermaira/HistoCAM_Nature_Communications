#include "JuceHeader.h"


//==============================================================================
ImageViewComponent::ImageViewComponent(std::shared_ptr<fRectangle> view,
                                       StringArray &iconNames,
                                       OwnedArray<Drawable> &iconsFromZipFile) : MRImage(NULL), view(view),
                                                                                 shade_levels(false) {

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

}

ImageViewComponent::~ImageViewComponent() {
}

void ImageViewComponent::refreshImage() {
  const ScopedLock lock(mutex);
  horizontalScrollBar.setRangeLimits(MRImage->bounds.x, MRImage->bounds.width);
  verticalScrollBar.setRangeLimits(MRImage->bounds.y, MRImage->bounds.height);

  horizontalScrollBar.setVisible(true);
  verticalScrollBar.setVisible(true);

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

void ImageViewComponent::mouseDrag(const juce::MouseEvent &event) {
  if (event.mods.isLeftButtonDown()) {
    juce::Point<int> idelta = event.getPosition() - lastMousePosition;
    fPoint delta = fPoint(idelta.x, idelta.y) * screen2viewScale(*view);
    translate(-delta);
    lastMousePosition = event.getPosition();
    repaint();
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
    return true;  // Key press handled
  }
  if (key == juce::KeyPress::createFromDescription("=")) { //Really "+"
    scaleCenter(fPoint(0.5, 0.5));
    repaint();
    return true;  // Key press handled
  }
  if (key == juce::KeyPress::createFromDescription("s")) {
    shade_levels = !shade_levels;
    repaint();
    return true;  // Key press handled
  }
  if (key.getKeyCode() == KeyPress::escapeKey) {
    JUCEApplication::getInstance()->systemRequestedQuit();
  }
  return false;  // Key press not handled
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
  bool shadeClasses = true;

  for (unsigned int i = 0; i < MRImage->images.size(); i++) {
    if (MRImage->images[i]->scale == 0) { continue; }
    auto imageview = *view;
    imageview *= 1.0 / MRImage->images[i]->scale;
    imageview -= fPoint(MRImage->images[i]->offset.x, MRImage->images[i]->offset.y);
    std::vector<TileQuery> tiles = MRImage->images[i]->getTiles(RectJtoC(imageview), RectJtoC(getLocalBounds()));
    for (unsigned int t = 0; t < tiles.size(); t++) {
      cv::Mat tile = tiles[t].image;
      auto bounds = RectCtoJ<float>(tiles[t].bounds);
      bounds *= view2screenScale(imageview) * scale;
      bounds.expand(0.5, 0.5);
      tiles[t].bounds = RectJtoC<float>(bounds);
      if (tile.data) {

        juce::Image im = juce::Image(juce::Image::ARGB, tile.cols, tile.rows, true);
        juce::Image::BitmapData bitmap_data(im, juce::Image::BitmapData::ReadWriteMode::writeOnly);

        jassert(tile.step == bitmap_data.lineStride);

        if (shade_levels) {
          if (!greenShade.data) {
            greenShade = Mat(tile.rows, tile.cols, CV_8UC3, cv::Scalar(0, 255, 0));
            channels.resize(2);
            channels[0] = greenShade;
          }
          double beta = (log2(1.0 / MRImage->images[i]->scale) / 3.4) * 0.7 + 0.05;

          greenShade.setTo(cv::Scalar(0, 0, 0));
          cv::extractChannel(tile, channels[1], 3);
          greenShade.setTo(cv::Scalar(200 * beta, 150 * (1 - beta), 100 * beta), channels[1]);
          cv::merge(channels, holding1);

          holding2 = beta * holding1 + (1.0 - beta) * tile;
          //holding2 = 0.5 * holding1 + 0.5 * tile;
          memcpy(bitmap_data.data, holding2.data, tile.cols * tile.rows * 4);
        } else {
          memcpy(bitmap_data.data, tile.data, tile.cols * tile.rows * 4);
        }
        g.drawImage(im, bounds);
      }

    }

    if (MRImage->images[0]->parent->classifyingComplete && shadeClasses) {
      auto baseTiles = MRImage->images[i]->getTiles(RectJtoC(imageview), RectJtoC(getLocalBounds()),true);
      for (auto tile : baseTiles) {
        if (tile.i == MRImage->images[0]->parent->im->minx && tile.j == MRImage->images[0]->parent->im->miny) {
          int k = 0;
        }

        //get draw bounds of base level tile
        auto bounds = RectCtoJ<float>(tile.bounds);
        bounds *= view2screenScale(imageview) * scale;
        bounds.expand(0.5, 0.5);
        tile.bounds = RectJtoC<float>(bounds);

        //get class for color
        auto tileCoords = Point2i(tile.i,tile.j);
        int classScore = MRImage->images[i]->get_class_for_tile(tileCoords);

        //draw it
        auto tileColor = Colour(uint8(0),0,0,uint8(0));
        switch (classScore) {
          case 1:
            tileColor = Colour(0, uint8(255), 0,uint8(50));
            break;
          case 2:
            tileColor = Colour(uint8(0), 0, 255,  uint8(50));
            break;
          case 3:
            tileColor = Colour(255,uint8(0), 0,  uint8(50));
            break;
          case 0:
            tileColor = Colour(uint8(50), 50, 50, uint8(20));
            break;
        }
        g.setColour(tileColor);
        g.fillRect(bounds);

      }
    }


// #ifdef DEBUG
//     for (unsigned int t = 0; t < tiles.size(); t++) {
//         auto bounds = RectCtoJ < float >(tiles[t].bounds) * scale;
//         g.setColour(juce::Colours::greenyellow);
//         g.drawRect(bounds, 3);
//         std::string ij = Poco::format("(%i,%i)", tiles[t].i, tiles[t].j);
//         g.setFont(20);
//         g.drawText(ij, bounds.getCentreX() - 50,
//                    bounds.getCentreY() - 15, 100, 30, Justification::centred);
//     }
// #endif


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

  if (shade_levels) {

    // Define the buffer space from the edges
    int padding = 100;

    int squareSize = 30; // Size of the square
    int textPadding = 10; // Space between the square and the text
    int linePadding = 10;  // Space between lines

    std::vector<std::string> objectives = {"20x", "10x", "4x", "2x"};
    std::vector<juce::Colour> colors = {Colour(66, 91, 176), Colour(120, 154, 175), Colour(190, 217, 201),
                                        Colour(243, 249, 243)};

    for (unsigned int i = 0; i < 4; i++) {
      int xPosition = getWidth() - padding - squareSize - textPadding;
      int yPosition = getHeight() - padding - (i * squareSize) - (i * linePadding);

      // Set the color for the square (purple)
      g.setColour(colors[i]);
      // Draw the purple square
      g.fillRect(xPosition, yPosition, squareSize, squareSize);

      // Set the color for the text (you can choose any color)
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
  if (!MRImage || !isVisible()) { return; }
  juce::Rectangle<int> b = getLocalBounds();
  *view = fRectangle(b.getX(), b.getY(), b.getWidth(), b.getHeight());
  //view.reset(new fRectangle(b.getX(),b.getY(),b.getWidth(),b.getHeight()));
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
