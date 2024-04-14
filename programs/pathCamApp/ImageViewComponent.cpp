#include "JuceHeader.h"



//==============================================================================
ImageViewComponent::ImageViewComponent(std::shared_ptr< MRTiledImage > MRImage): MRImage(MRImage), scale(1.0)
{
  addAndMakeVisible(horizontalScrollBar);
  addAndMakeVisible(verticalScrollBar);
  
  juce::Rectangle<int> b = getLocalBounds();
  
  horizontalScrollBar.setRangeLimits(0, b.getWidth());
  verticalScrollBar.setRangeLimits(0, b.getHeight());

  horizontalScrollBar.addListener(this);
  verticalScrollBar.addListener(this);

  // Set the initial positions and visibility of the scroll bars
  horizontalScrollBar.setCurrentRange(0, 1);
  verticalScrollBar.setCurrentRange(0, 1);

  horizontalScrollBar.setSingleStepSize(10);
  verticalScrollBar.setSingleStepSize(10);

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
    fPoint delta = fPoint(idelta.x, idelta.y);
  
    delta *= view.getVerticalRange().getLength()/getBounds().getVerticalRange().getLength();
    view = view-delta;
    lastMousePosition = event.getPosition();
    repaint();
}

void ImageViewComponent::mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) {
  scaleCenter(1.0+wheel.deltaY);
  repaint();
}

bool ImageViewComponent::keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) {
  if (key == juce::KeyPress::createFromDescription("-")) {
    scaleCenter(2.0);
    repaint();
    return true;  // Key press handled
  }
  if (key == juce::KeyPress::createFromDescription("=")) { //Really "+"
    scaleCenter(0.5);
    repaint();
    return true;  // Key press handled
  }
  
  return false;  // Key press not handled
}


void ImageViewComponent::scrollBarMoved(juce::ScrollBar* scrollBar, double newRangeStart)
{
    // This method is called when the scroll bar is moved
    if (scrollBar == &horizontalScrollBar)
    {
        // Update the content's horizontal position
    }
    else if (scrollBar == &verticalScrollBar)
    {
        // Update the content's vertical position
    }
}



//==============================================================================
void ImageViewComponent::paint (juce::Graphics& g)
{

  g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
  
  g.drawImageAt(checkerboard, 0, 0);
    
  std::vector < TileQuery >  tiles = MRImage->getTiles(view);
  
  for(unsigned int i = 0; i < tiles.size(); i++){
    juce::Image *im = tiles[i].image;
    tiles[i].bounds *= getBounds().getVerticalRange().getLength()/view.getVerticalRange().getLength();
    if(im != NULL){
      g.drawImage(*im, tiles[i].bounds);
    }
  }
#ifdef DEBUG
  for(unsigned int i = 0; i < tiles.size(); i++){
    g.setColour (juce::Colours::greenyellow);
    g.drawRect(tiles[i].bounds, 3);
    std::string ij = Poco::format("(%i,%i)", tiles[i].i, tiles[i].j);
    g.setFont (20);
    g.drawText ( ij, tiles[i].bounds.getCentreX()-50,
                tiles[i].bounds.getCentreY()-15, 100, 30, Justification::centred);
  }
#endif
  
}

void ImageViewComponent::resized()
{
  // This is called when the ImageViewComponent is resized.
  // If you add any child components, this is where you should
  // update their positions.
  const ScopedLock lock (mutex);

  
  juce::Rectangle<int> b = getLocalBounds();
  
  horizontalScrollBar.setBounds(b.removeFromBottom(18));
  verticalScrollBar.setBounds(b.removeFromRight(18));
  
  view = fRectangle(b.getX(),b.getY(),b.getWidth(),b.getHeight());
  view += MRImage->bounds.getCentre();
  
  
  checkerboard = createCheckerboardImage(getBounds().getWidth(), getBounds().getHeight(),
                                         64, juce::Colours::lightgrey,
                                         juce::Colours::white);
   
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
