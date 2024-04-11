#include "JuceHeader.h"



//==============================================================================
ImageViewComponent::ImageViewComponent(std::shared_ptr< pathCam::StreamCam > bcam): bcam(bcam)
{
  
#ifdef DEBUG
  std::stringstream ss;
  ss <<  PROJECT_SOURCE_DIR << "/resources/screen_texture_test.png";

  cv::Mat image = imread(ss.str());
  std::cout << "Read OpenCV image: " << image.cols << "X" << image.rows << "\n";
  
  timage.insertMat(image, pCApp::Rectangle(0,0, image.cols, image.rows));

#endif
  
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
    juce::Point<int> delta = event.getPosition() - lastMousePosition;
    bounds -= delta;
    lastMousePosition = event.getPosition();
    repaint();
}


//==============================================================================
void ImageViewComponent::paint (juce::Graphics& g)
{
//  g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
  
  g.drawImageAt(checkerboard, 0, 0);
    
  std::vector < TileQueryElem >  tiles = timage.getTiles(bounds);
  
  for(unsigned int i = 0; i < tiles.size(); i++){
    juce::Image *im = timage.getTile(tiles[i].i,tiles[i].j);
    
    if(im != NULL){
      g.drawImageAt(*im,
                    tiles[i].i*timage.getTileSize() - bounds.getX(),
                    tiles[i].j*timage.getTileSize() - bounds.getY());
    }

#ifdef DEBUG
    g.setColour (juce::Colours::greenyellow);
    int x = (int)(tiles[i].i*timage.getTileSize() - bounds.getX());
    int y = (int)(tiles[i].j*timage.getTileSize() - bounds.getY());
    g.drawRect(x, y,
               (int)timage.getTileSize(), (int)timage.getTileSize(), 3);
    std::string ij = Poco::format("(%i,%i)", tiles[i].i, tiles[i].j);
    g.setFont (20);
    g.drawText ( ij,x + timage.getTileSize()/2-50, y + timage.getTileSize()/2-15, 100, 30, Justification::centred);
#endif
  }
}

void ImageViewComponent::resized()
{
  // This is called when the ImageViewComponent is resized.
  // If you add any child components, this is where you should
  // update their positions.
  const ScopedLock lock (mutex);
  bounds = getLocalBounds();
  
  checkerboard = createCheckerboardImage(bounds.getWidth(), bounds.getHeight(),
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
