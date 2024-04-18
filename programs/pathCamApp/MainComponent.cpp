#include "JuceHeader.h"



//==============================================================================
MainComponent::MainComponent(std::shared_ptr< pathCam::StreamCam > bcam): bcam(bcam)
{
  //Won't work for deployment, but ok for now
  std::stringstream ss;
  ss <<  PROJECT_SOURCE_DIR << "/resources/hud_icons.zip";
  
  ZipFile icons (File(ss.str().c_str()));

  for (int i = 0; i < icons.getNumEntries(); ++i)
  {
    std::unique_ptr<InputStream> svgFileStream (icons.createStreamForEntry (i));
    
    if (svgFileStream.get() != nullptr)
    {
      iconNames.add (icons.getEntry (i)->filename);
      iconsFromZipFile.add (Drawable::createFromImageDataStream (*svgFileStream));
    }
  }
  
  
  toolbar = new ToolbarComponent(this);
  imageview = new ImageViewComponent(this, iconNames, iconsFromZipFile);
  capture = new CaptureComponent(this, iconNames, iconsFromZipFile);
  
  addAndMakeVisible(imageview);
  addAndMakeVisible(toolbar);
  
  addChildComponent(capture);
  
  setWantsKeyboardFocus(true);
  addKeyListener(imageview);
  addKeyListener(capture);

  
  setSize (1024, 768);

}

MainComponent::~MainComponent()
{
  delete progressBar;
  delete toolbar;
  delete imageview;
  delete capture;
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
  capture->setBounds(b);
  
  
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
    capture->setImage(MRimage);
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
  
  if(event == "home"){
    imageview->setVisible(true);
    capture->setVisible(false);
  }
  
  if(event == "capture"){
    imageview->setVisible(false);
    capture->setVisible(true);
  }
  
  
}
