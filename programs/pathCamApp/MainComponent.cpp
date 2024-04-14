#include "JuceHeader.h"



//==============================================================================
MainComponent::MainComponent(std::shared_ptr< pathCam::StreamCam > bcam): bcam(bcam)
{
  
  double progress = 0.0;
  progressBar = new ProgressBar(progress);
  
  
  std::stringstream ss;
  ss <<  PROJECT_SOURCE_DIR << "/resources/stream_working.png";
  //ss <<  PROJECT_SOURCE_DIR << "/resources/screen_texture_test.png";

  

  cv::Mat cvimage = imread(ss.str());
  std::cout << "Read OpenCV image: " << cvimage.cols << "X" << cvimage.rows << "\n";

  MRimage = std::make_shared< MRTiledImage >();
  MRimage->build(cvimage);
  
  
  toolbar = new ToolbarComp();
  imageview = new ImageViewComponent(MRimage);
  
  addAndMakeVisible(imageview);
  addAndMakeVisible(toolbar);
  
  setWantsKeyboardFocus(true);
  addKeyListener(imageview);

  
  //addAndMakeVisible(progressBar);

  
  setSize (1024, 768);

}

MainComponent::~MainComponent()
{
  delete progressBar;
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
  progressBar->setBounds(50, getHeight()-40, getWidth() - 100, 30);

}

