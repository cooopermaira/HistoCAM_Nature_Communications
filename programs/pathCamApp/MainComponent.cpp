#include "MainComponent.h"



//==============================================================================
MainComponent::MainComponent(std::shared_ptr< StreamCam > bcam): bcam(bcam)
{
  // Make sure you set the size of the component after
  // you add any child components.
  setSize (1024, 768);
  
  std::stringstream ss;
  ss <<  PROJECT_SOURCE_DIR << "/resources/screen_texture_test.png";

  image = imread(ss.str());
  std::cout << "Read OpenCV image: " << image.cols << "X" << image.rows << "\n";
}

MainComponent::~MainComponent()
{
  // This shuts down the GL system and stops the rendering calls.
  shutdownOpenGL();
}

//==============================================================================
void MainComponent::initialise()
{
  // Initialise GL objects for rendering here.
  using namespace ::juce::gl;
  createShaders();
  
}

void MainComponent::shutdown()
{
  // Free any GL objects created for rendering here.
  using namespace ::juce::gl;
  
  texture.release();
  
  shader    .reset();
  squareBuffer     .reset();
  attributes.reset();
  uniforms  .reset();

}

Matrix3D<float>  getOrthoMatrix( const GLfloat left, const GLfloat right,
                                const GLfloat bottom, const GLfloat top,
                                const GLfloat zNear, const GLfloat zFar )
{
  Matrix3D<float> c;
  
  c.mat[0] = 2.0/(right - left);
  c.mat[5] = 2.0/(top - bottom);
  c.mat[10] = -2.0/(zNear - zFar);
  c.mat[12] = -(right + left)/(right - left);
  c.mat[13] = -(top + bottom)/(top - bottom);
  c.mat[14] = -(zFar + zNear)/(zFar - zNear);
  return c;
}


void MainComponent::render()
{
  
  using namespace ::juce::gl;
  
  jassert (OpenGLHelpers::isContextActive());
  
  //Need the opengl context to get the right answer here.
  desktopScale = (float) openGLContext.getRenderingScale();
  pixel_bounds = juce::Rectangle(roundToInt (desktopScale * (float) bounds.getWidth()),
                           roundToInt (desktopScale * (float) bounds.getHeight()));
  
  OpenGLHelpers::clear (getLookAndFeel().findColour (ResizableWindow::backgroundColourId));
  
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  {
    const ScopedLock lock (mutex);
    glViewport (0, 0,
                roundToInt (desktopScale * (float) bounds.getWidth()),
                roundToInt (desktopScale * (float) bounds.getHeight()));
  }
  
  shader->use();
  
  if (uniforms->projectionMatrix != nullptr)
    uniforms->projectionMatrix->setMatrix4 (getOrthoMatrix(-1.0, 1.0,
                                                           1.0, -1.0,
                                                           -1.0, 1.0).mat, 1, true);
  
  if (uniforms->texture != nullptr)
      uniforms->texture->set ((GLint) 0);
  
  texture.bind();

  squareBuffer->bind();
  attributes->enable();
  glDrawElements (GL_TRIANGLES, squareBuffer->numIndices, GL_UNSIGNED_INT, nullptr);
  attributes->disable();
  
  texture.unbind();

  // Reset the element buffers so child Components draw correctly
  glBindBuffer (GL_ARRAY_BUFFER, 0);
  glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);
  
}

//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
  // You can add your component specific drawing code here!
  // This will draw over the top of the openGL background.
  
  g.setColour (getLookAndFeel().findColour (Label::textColourId));
  g.setFont (20);
  g.drawText ("Example for Cooper", 25, 20, 300, 30, Justification::left);
  g.drawLine (20, 20, 190, 20);
  g.drawLine (20, 50, 190, 50);
}

void MainComponent::resized()
{
  // This is called when the MainComponent is resized.
  // If you add any child components, this is where you should
  // update their positions.
  const ScopedLock lock (mutex);
  bounds = getLocalBounds();
   
}

juce::Image convertOpenCVMatToJUCEImage(const cv::Mat& mat) {
    // Ensure the source Mat is in BGR format
    jassert(mat.type() == CV_8UC3);

    // Create a JUCE image of the same dimensions (note: JUCE uses ARGB format internally)
    juce::Image image(juce::Image::PixelFormat::RGB, mat.cols, mat.rows, true);

    juce::Image::BitmapData imageData(image, juce::Image::BitmapData::writeOnly);

    for (int y = 0; y < mat.rows; ++y) {
        auto* srcRow = mat.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mat.cols; ++x) {
            // Convert BGR to RGB
            auto& pixel = srcRow[x];
            imageData.setPixelColour(x, y, juce::Colour(pixel[2], pixel[1], pixel[0]));
        }
    }

    return image;
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


void MainComponent::createShaders()
{
  vertexShader =
  "attribute vec4 position;\n"
  "attribute vec2 textureCoordIn;\n"
  "\n"
  "uniform mat4 projectionMatrix;\n"
  "\n"
  "varying vec2 textureCoordOut;\n"
  "\n"
  "void main()\n"
  "{\n"
  "    textureCoordOut = textureCoordIn;\n"
  "    gl_Position = projectionMatrix*position;\n"
  "}\n";
  
  fragmentShader =
#if JUCE_OPENGL_ES
  "varying lowp vec2 textureCoordOut;\n"
#else
  "varying vec2 textureCoordOut;\n"
#endif
  "uniform sampler2D demoTexture;\n"
  "\n"
  "void main()\n"
  "{\n"
  "    gl_FragColor = texture2D (demoTexture, textureCoordOut);\n"
  "}\n";
  
  std::unique_ptr<OpenGLShaderProgram> newShader (new OpenGLShaderProgram (openGLContext));
  juce::String statusText;
  
  if (newShader->addVertexShader (OpenGLHelpers::translateVertexShaderToV3 (vertexShader))
      && newShader->addFragmentShader (OpenGLHelpers::translateFragmentShaderToV3 (fragmentShader))
      && newShader->link())
  {
    squareBuffer     .reset();
    attributes.reset();
    uniforms  .reset();
    
    shader = std::move (newShader);
    shader->use();
    
    squareBuffer     .reset (new SquareBuffer());
    attributes.reset (new Attributes (*shader));
    uniforms  .reset (new Uniforms (*shader));
    
    //To get this to work need to copy into JUCE image format.
    //Inefficient and can probably be fixed if needed.
    juce::Image temp = convertOpenCVMatToJUCEImage(image);
    texture.loadImage(temp);
    
    statusText = "GLSL: v" + juce::String (OpenGLShaderProgram::getLanguageVersion(), 2);
  }
  else
  {
    statusText = newShader->getLastError();
  }
}
