#include "JuceHeader.h"



//==============================================================================
MainComponent::MainComponent(std::shared_ptr< pathCam::StreamCam > bcam): bcam(bcam)
{
  
  
  toolbar = new ToolbarComponent(this);
  imageview = new ImageViewComponent(this);
  
  addAndMakeVisible(imageview);
  addAndMakeVisible(toolbar);
  
  setWantsKeyboardFocus(true);
  addKeyListener(imageview);
  
  
  setSize (1024, 768);

}

MainComponent::~MainComponent()
{
  delete progressBar;
  delete toolbar;
  delete imageview;
}

void MainComponent::loadImage(std::string path){
  
  cv::Mat cvimage = imread(path);
  std::cout << "Read OpenCV image: " << cvimage.cols << "X" << cvimage.rows << "\n";
  MRimage.reset(new MRTiledImage());
  MRimage->build(cvimage);
}


//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{

}

void MainComponent::resized()
{
  const ScopedLock lock (mutex);
  Rectangle < int > b = getLocalBounds();
  toolbar->setBounds(b.removeFromTop(50));
  imageview->setBounds(b);
  
  
//#if DEBUG
//  std::stringstream ss;
//  ss <<  PROJECT_SOURCE_DIR << "/resources/stream_working.png";
//  loadImage(ss.str());
//  imageview->setImage(MRimage);
//#endif

}


void MainComponent::loadImageDialog(const FileChooser& fc){

  File result = fc.getResult();
  if (result.exists()){
    loadImage(result.getFullPathName().toStdString());
    imageview->setImage(MRimage);
  }
}




void MainComponent::GuiEventHandler(std::string event){
  if(event == "open"){
    fc.reset (new FileChooser ("Choose an image to open...", File::getCurrentWorkingDirectory(), "*.png,*.jpeg,*.tiff"));

    fc->launchAsync (FileBrowserComponent::openMode
                     | FileBrowserComponent::canSelectFiles,
                     std::bind(&MainComponent::loadImageDialog, this, std::placeholders::_1));

    return;
  }
  
  
}
