#include "JuceHeader.h"



//==============================================================================
ImageViewComponent::ImageViewComponent(std::shared_ptr< pathCam::StreamCam > bcam): bcam(bcam)
{
  
  // Make sure you set the size of the component after
  // you add any child components.
  //setSize (1024, 768);
  
  std::stringstream ss;
  ss <<  PROJECT_SOURCE_DIR << "/resources/screen_texture_test.png";

  cv::Mat image = imread(ss.str());
  std::cout << "Read OpenCV image: " << image.cols << "X" << image.rows << "\n";
  
  
  timage.insertMat(image, pCApp::Rectangle(0,0, image.cols, image.rows));
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
  // You can add your component specific drawing code here!
  // This will draw over the top of the openGL background.
  
  // Clear the background
  g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
  
  g.drawImageAt(checkerboard, 0, 0);
  
  //bounds = pCApp::Rectangle(-100,-400, 1024, 708);
  
  std::vector < TileQueryElem >  tiles = timage.getTiles(bounds);
  

  for(unsigned int i = 0; i < tiles.size(); i++){
    juce::Image *im = timage.getTile(tiles[i].i,tiles[i].j);
    std::cout << tiles[i].i << " , " << tiles[i].j  << "\t\t";
    
    
    if(im != NULL){
      g.drawImageAt(*im,
                    tiles[i].i*timage.getTileSize() - bounds.getX(),
                    tiles[i].j*timage.getTileSize() - bounds.getY());
    }
    g.setColour (juce::Colours::greenyellow);
    int x = (int)(tiles[i].i*timage.getTileSize() - bounds.getX());
    int y = (int)(tiles[i].j*timage.getTileSize() - bounds.getY());
    g.drawRect(x, y,
               (int)timage.getTileSize(), (int)timage.getTileSize(), 3);
    std::string ij = Poco::format("(%i,%i)", tiles[i].i, tiles[i].j);
    g.setFont (20);
    g.drawText ( ij,x + timage.getTileSize()/2-50, y + timage.getTileSize()/2-15, 100, 30, Justification::centred);
  }
  std::cout << "\n\n";
}

juce::Image createCheckerboardImage(int width, int height, int squareSize, juce::Colour colour1, juce::Colour colour2) {
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

//juce::Image convertOpenCVMatToJUCEImage(const cv::Mat& mat) {
//    // Ensure the source Mat is in BGR format
//    jassert(mat.type() == CV_8UC3);
//
//    // Create a JUCE image of the same dimensions (note: JUCE uses ARGB format internally)
//    juce::Image image(juce::Image::PixelFormat::RGB, mat.cols, mat.rows, true);
//
//    juce::Image::BitmapData imageData(image, juce::Image::BitmapData::writeOnly);
//
//    for (int y = 0; y < mat.rows; ++y) {
//        auto* srcRow = mat.ptr<cv::Vec3b>(y);
//        for (int x = 0; x < mat.cols; ++x) {
//            // Convert BGR to RGB
//            auto& pixel = srcRow[x];
//            imageData.setPixelColour(x, y, juce::Colour(pixel[2], pixel[1], pixel[0]));
//        }
//    }
//
//    return image;
//}
//
//juce::Image createCheckerboardImage(int width, int height, int squareSize, juce::Colour colour1, juce::Colour colour2) {
//    juce::Image checkerboard(juce::Image::PixelFormat::RGB, width, height, true);
//
//    juce::Graphics g(checkerboard);
//
//    for (int y = 0; y < height; y += squareSize) {
//        for (int x = 0; x < width; x += squareSize) {
//            // Determine the color based on the position
//            bool isColour1 = ((x / squareSize) % 2 == 0) ^ ((y / squareSize) % 2 == 0);
//            g.setColour(isColour1 ? colour1 : colour2);
//
//            // Draw the square
//            g.fillRect(x, y, squareSize, squareSize);
//        }
//    }
//
//    return checkerboard;
//}
//
//
//void ImageViewComponent::createShaders()
//{
//  vertexShader =
//  "attribute vec4 position;\n"
//  "attribute vec2 textureCoordIn;\n"
//  "\n"
//  "uniform mat4 projectionMatrix;\n"
//  "\n"
//  "varying vec2 textureCoordOut;\n"
//  "\n"
//  "void main()\n"
//  "{\n"
//  "    textureCoordOut = textureCoordIn;\n"
//  "    gl_Position = projectionMatrix*position;\n"
//  "}\n";
//
//  fragmentShader =
//#if JUCE_OPENGL_ES
//  "varying lowp vec2 textureCoordOut;\n"
//#else
//  "varying vec2 textureCoordOut;\n"
//#endif
//  "uniform sampler2D demoTexture;\n"
//  "\n"
//  "void main()\n"
//  "{\n"
//  "    gl_FragColor = texture2D (demoTexture, textureCoordOut);\n"
//  "}\n";
//
//  std::unique_ptr<OpenGLShaderProgram> newShader (new OpenGLShaderProgram (openGLContext));
//  juce::String statusText;
//
//  if (newShader->addVertexShader (OpenGLHelpers::translateVertexShaderToV3 (vertexShader))
//      && newShader->addFragmentShader (OpenGLHelpers::translateFragmentShaderToV3 (fragmentShader))
//      && newShader->link())
//  {
//    squareBuffer     .reset();
//    attributes.reset();
//    uniforms  .reset();
//
//    shader = std::move (newShader);
//    shader->use();
//
//    squareBuffer     .reset (new SquareBuffer());
//    attributes.reset (new Attributes (*shader));
//    uniforms  .reset (new Uniforms (*shader));
//
//    //To get this to work need to copy into JUCE image format.
//    //Inefficient and can probably be fixed if needed.
//    //juce::Image temp;
//    //copyMatToImage(image, temp);
//    //texture.loadImage(temp);
//
//    statusText = "GLSL: v" + juce::String (OpenGLShaderProgram::getLanguageVersion(), 2);
//  }
//  else
//  {
//    statusText = newShader->getLastError();
//  }
//}
