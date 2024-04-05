#include "MainComponent.h"



//==============================================================================
MainComponent::MainComponent(std::shared_ptr< pathCam::StreamCam > bcam): bcam(bcam)
{
  
  toolbar = new ToolbarComp();
  imageview = new ImageViewComponent(bcam);
  
  addAndMakeVisible(imageview);
  addAndMakeVisible(toolbar);
  

  // Make sure you set the size of the component after
  // you add any child components.
  setSize (1024, 768);
  
//  std::stringstream ss;
//  ss <<  PROJECT_SOURCE_DIR << "/resources/screen_texture_test.png";
//
//  image = imread(ss.str());
//  std::cout << "Read OpenCV image: " << image.cols << "X" << image.rows << "\n";
}

MainComponent::~MainComponent()
{
  delete toolbar;
  delete imageview;
}


//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
  // You can add your component specific drawing code here!
  // This will draw over the top of the openGL background.
  
//  g.setColour (getLookAndFeel().findColour (Label::textColourId));
//  g.setFont (20);
//  g.drawText ("Example for Cooper", 25, 20, 300, 30, Justification::left);
//  g.drawLine (20, 20, 190, 20);
//  g.drawLine (20, 50, 190, 50);
}

void MainComponent::resized()
{
  // This is called when the MainComponent is resized.
  // If you add any child components, this is where you should
  // update their positions.
  const ScopedLock lock (mutex);
  toolbar->setBounds(0, 0, getWidth(), 60);
  imageview->setBounds(0, 60, getWidth(), getHeight()-60);
}

