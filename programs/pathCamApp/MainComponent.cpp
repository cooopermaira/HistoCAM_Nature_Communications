#include "JuceHeader.h"



//==============================================================================
MainComponent::MainComponent(std::shared_ptr< pathCam::StreamCam > bcam): bcam(bcam)
{
  
  std::stringstream ss;
  ss <<  PROJECT_SOURCE_DIR << "/resources/stream_working.png";

  cv::Mat cvimage = imread(ss.str());
  std::cout << "Read OpenCV image: " << cvimage.cols << "X" << cvimage.rows << "\n";

  MRimage = std::make_shared< MRTiledImage >();
  MRimage->build(cvimage);
  
  
  toolbar = new ToolbarComp();
  imageview = new ImageViewComponent(MRimage);
  
  addAndMakeVisible(imageview);
  addAndMakeVisible(toolbar);
  
  setWantsKeyboardFocus(true);
  addKeyListener(this);
  
  setSize (1024, 768);

}

MainComponent::~MainComponent()
{
  delete toolbar;
  delete imageview;
}


//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{

}

void MainComponent::resized()
{
  const ScopedLock lock (mutex);
  toolbar->setBounds(0, 0, getWidth(), 60);
  imageview->setBounds(0, 60, getWidth(), getHeight()-60);
}

