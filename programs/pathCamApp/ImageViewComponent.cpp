#include "JuceHeader.h"


//==============================================================================
ImageViewComponent::ImageViewComponent(std::shared_ptr<fRectangle> view,
                                       StringArray &iconNames,
                                       OwnedArray<Drawable> &iconsFromZipFile) : MRImage(NULL), view(view) {

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

void ImageViewComponent::refreshImage(){
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
  
  for(unsigned int i=0; i < MRImage->images.size(); i++){
    if(MRImage->images[i]->scale == 0){continue;}
    auto imageview = *view;
    imageview *= 1.0/MRImage->images[i]->scale;
    imageview -= fPoint(MRImage->images[i]->offset.x, MRImage->images[i]->offset.y);
    std::vector<TileQuery> tiles = MRImage->images[i]->getTiles(RectJtoC(imageview), RectJtoC(getLocalBounds()));
    for (unsigned int t = 0; t < tiles.size(); t++) {
      cv::Mat tile = tiles[t].image;
      auto bounds = RectCtoJ < float >(tiles[t].bounds);
      bounds *= view2screenScale(imageview) * scale;
      bounds.expand(0.5, 0.5);
      tiles[t].bounds = RectJtoC <float> (bounds);
      if (tile.data) {
        juce::Image im = juce::Image(juce::Image::ARGB, tile.cols, tile.rows, true);
        juce::Image::BitmapData bitmap_data(im, juce::Image::BitmapData::ReadWriteMode::writeOnly);
        
        jassert(tile.step == bitmap_data.lineStride);
        memcpy(bitmap_data.data, tile.data, tile.cols*tile.rows*4);
        
        g.drawImage(im, bounds);
      }
    }

#ifdef DEBUG
    for (unsigned int t = 0; t < tiles.size(); t++) {
        auto bounds = RectCtoJ < float >(tiles[t].bounds) * scale;
        g.setColour(juce::Colours::greenyellow);
        g.drawRect(bounds, 3);
        std::string ij = Poco::format("(%i,%i)", tiles[t].i, tiles[t].j);
        g.setFont(20);
        g.drawText(ij, bounds.getCentreX() - 50,
                   bounds.getCentreY() - 15, 100, 30, Justification::centred);
    }
#endif
    
    
  }






}


//==============================================================================
void ImageViewComponent::paint(juce::Graphics &g) {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));

    g.drawImageAt(checkerboard, 0, 0);

    if (MRImage) {
        drawSlide(g, 1.0);
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
  if(!MRImage || !isVisible()){ return; }
  juce::Rectangle<int> b = getLocalBounds();
  *view = fRectangle(b.getX(),b.getY(),b.getWidth(),b.getHeight());
  //view.reset(new fRectangle(b.getX(),b.getY(),b.getWidth(),b.getHeight()));
  view->setCentre(RectCtoJ(MRImage->bounds).getCentre());

  float scale = max((float)MRImage->bounds.width/
                    (float)view->getHorizontalRange().getLength(),
                    (float)MRImage->bounds.height/
                    (float)view->getVerticalRange().getLength());

  scaleCenter(fPoint(scale,scale));

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
