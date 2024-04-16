#include "JuceHeader.h"


//==============================================================================
ImageViewComponent::ImageViewComponent(std::shared_ptr< MRTiledImage > MRImage): MRImage(MRImage)
{
  
  setOpaque (true);
  controlsOverlay.reset (new ImageViewOverlay ());
  addAndMakeVisible (controlsOverlay.get());

  
  juce::Rectangle<int> b = getLocalBounds();


  horizontalScrollBar.setRangeLimits(0, MRImage->bounds.getWidth());
  verticalScrollBar.setRangeLimits(0, MRImage->bounds.getHeight());

  horizontalScrollBar.addListener(this);
  verticalScrollBar.addListener(this);

  // Set the initial positions and visibility of the scroll bars
  horizontalScrollBar.setCurrentRange(0, 400);
  verticalScrollBar.setCurrentRange(0, 400);

  horizontalScrollBar.setSingleStepSize(10);
  verticalScrollBar.setSingleStepSize(10);
  
  addAndMakeVisible(horizontalScrollBar);
  addAndMakeVisible(verticalScrollBar);


}

ImageViewComponent::~ImageViewComponent()
{
}

void ImageViewComponent::mouseDown(const juce::MouseEvent& event)
{
    lastMousePosition = event.getPosition();
}

void ImageViewComponent::mouseDrag(const juce::MouseEvent& event)
{
    juce::Point<int> idelta = event.getPosition() - lastMousePosition;
    fPoint delta = fPoint(idelta.x, idelta.y)* screen2view();
    translate(-delta);
    lastMousePosition = event.getPosition();
    repaint();
}

void ImageViewComponent::mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) {
    scaleCenter(fPoint(1.0-wheel.deltaY,1.0-wheel.deltaY));
    repaint();
}

void ImageViewComponent::mouseMagnify (const MouseEvent&, float magnifyAmmount)
{
  scaleCenter(fPoint(1.0/magnifyAmmount,1.0/magnifyAmmount));
  repaint();
}

bool ImageViewComponent::keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) {
  if (key == juce::KeyPress::createFromDescription("-")) {
    scaleCenter(fPoint(2.0,2.0));
    repaint();
    return true;  // Key press handled
  }
  if (key == juce::KeyPress::createFromDescription("=")) { //Really "+"
    scaleCenter(fPoint(0.5,0.5));
    repaint();
    return true;  // Key press handled
  }
  if (key.getKeyCode() == KeyPress::escapeKey)
  {
      JUCEApplication::getInstance()->systemRequestedQuit();
  }
  return false;  // Key press not handled
}

void ImageViewComponent::updateScrollbar(){

  horizontalScrollBar.setCurrentRangeStart(view.getCentreX());
  verticalScrollBar.setCurrentRangeStart(view.getCentreY());
  
}


void ImageViewComponent::scrollBarMoved(juce::ScrollBar* scrollBar, double newRangeStart)
{
    // This method is called when the scroll bar is moved
    if (scrollBar == &horizontalScrollBar)
    {
      view.setCentre(newRangeStart, view.getCentreY());
      repaint();
    }
    else if (scrollBar == &verticalScrollBar)
    {
      view.setCentre(view.getCentreX(), newRangeStart);
      repaint();
    }
}

void ImageViewComponent::drawSlide(juce::Graphics& g, float scale){
  
  std::vector < TileQuery >  tiles = MRImage->getTiles(view, getLocalBounds());
  
  for(unsigned int i = 0; i < tiles.size(); i++){
    juce::Image *im = tiles[i].image;
    tiles[i].bounds *= view2screen()*scale;
    tiles[i].bounds.expand(0.5, 0.5);
    if(im != NULL){
      g.drawImage(*im, tiles[i].bounds);
    }
  }
  
  

#ifdef DEBUG
for(unsigned int i = 0; i < tiles.size(); i++){
  auto bounds = tiles[i].bounds*scale;
  g.setColour (juce::Colours::greenyellow);
  g.drawRect(bounds, 3);
  std::string ij = Poco::format("(%i,%i)", tiles[i].i, tiles[i].j);
  g.setFont (20);
  g.drawText ( ij, bounds.getCentreX()-50,
              bounds.getCentreY()-15, 100, 30, Justification::centred);
}
#endif
}


//==============================================================================
void ImageViewComponent::paint (juce::Graphics& g)
{

  g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
  
  g.drawImageAt(checkerboard, 0, 0);
  
  if(false){
    
    auto scale = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()->scale;

    juce::Image screenBuffer(juce::Image::PixelFormat::ARGB, getLocalBounds().getWidth()*scale, getLocalBounds().getHeight()*scale, true);
    
    Graphics b(screenBuffer);
    
    drawSlide(b, scale);
  
    
    g.drawImage(screenBuffer, fRectangle(getLocalBounds().getX(),
                                         getLocalBounds().getY(),
                                         getLocalBounds().getWidth(),
                                         getLocalBounds().getHeight()));
  }else{
    drawSlide(g, 1.0);
  }
  
}

void ImageViewComponent::resized()
{
  // This is called when the ImageViewComponent is resized.
  // If you add any child components, this is where you should
  // update their positions.
  const ScopedLock lock (mutex);

  juce::Rectangle<int> b = getLocalBounds();
  
  horizontalScrollBar.setBounds(b.removeFromBottom(20));
  verticalScrollBar.setBounds(b.removeFromRight(20));
  
  controlsOverlay->setBounds(juce::Rectangle<int>(40,40, 60, 120));

  
  b = getLocalBounds();
  
  if(!old_bounds.isEmpty()){
    scaleCenter(fPoint((float)b.getHorizontalRange().getLength()/
                       (float)old_bounds.getHorizontalRange().getLength(),
                       (float)b.getVerticalRange().getLength()/
                       (float)old_bounds.getVerticalRange().getLength()));
  
  }else{
    view = fRectangle(b.getX(),b.getY(),b.getWidth(),b.getHeight());
    view.setCentre(MRImage->bounds.getCentre());
    
    float scale = max((float)MRImage->bounds.getHorizontalRange().getLength()/
                      (float)view.getHorizontalRange().getLength(),
                      (float)MRImage->bounds.getVerticalRange().getLength()/
                      (float)view.getVerticalRange().getLength());
    
    scaleCenter(fPoint(scale,scale));
  }
  
  
  checkerboard = createCheckerboardImage(getLocalBounds().getWidth(),
                                         getLocalBounds().getHeight(),
                                         64,
                                         juce::Colours::lightgrey,
                                         juce::Colours::white);
   
  old_bounds = getLocalBounds();
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
