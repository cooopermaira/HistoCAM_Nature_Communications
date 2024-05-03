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
  view.reset( new fRectangle() );
  imageview = new ImageViewComponent(view, iconNames, iconsFromZipFile);
  capture = new CaptureComponent(view, iconNames, iconsFromZipFile);
  annotate = new AnnotateComponent(view, iconNames, iconsFromZipFile);

  addAndMakeVisible(imageview);
  addAndMakeVisible(toolbar);
  
  addChildComponent(capture);
  addChildComponent(annotate);

  
  setWantsKeyboardFocus(true);
  addKeyListener(imageview);
  addKeyListener(capture);
  addKeyListener(annotate->getViewComp());

  
  setSize (1024, 768);

}

MainComponent::~MainComponent()
{
  delete progressBar;
  delete toolbar;
  delete imageview;
  delete capture;
  delete annotate;
}

//==============================================================================
class DemoBackgroundThread final : public juce::ThreadWithProgressWindow
{
public:
    explicit DemoBackgroundThread ()
        : juce::ThreadWithProgressWindow ("busy doing some important things...", true, true)
    {
      setStatusMessage("Getting ready ...");
    }

    void run() override
    {
        setProgress (-1.0); // setting a value beyond the range 0 -> 1 will show a spinning bar..
        setStatusMessage ("Preparing to do some stuff...");
        wait (2000);

        int thingsToDo = 10;

        for (int i = 0; i < thingsToDo; ++i)
        {
            // must check this as often as possible, because this is
            // how we know if the user's pressed 'cancel'
            if (threadShouldExit())
                return;

            // this will update the progress bar on the dialog box
            setProgress (i / (double) thingsToDo);

            setStatusMessage (juce::String (thingsToDo - i) + " things left to do...");

            wait (500);
        }

        setProgress (-1.0); // setting a value beyond the range 0 -> 1 will show a spinning bar..
        setStatusMessage ("Finishing off the last few bits and pieces!");
        wait (2000);
    }

    // This method gets called on the message thread once our thread has finished..
    void threadComplete (bool userPressedCancel) override
    {
//        const juce::String messageString (userPressedCancel ? "You pressed cancel!" : "Thread finished ok!");
//
//        if (owner != nullptr)
//        {
//            owner->messageBox = AlertWindow::showScopedAsync (MessageBoxOptions()
//                                                                  .withIconType (MessageBoxIconType::InfoIcon)
//                                                                  .withTitle ("Progress window")
//                                                                  .withMessage (messageString)
//                                                                  .withButton ("OK"),
//                                                              nullptr);
//        }
//
//        // ..and clean up by deleting our thread object..
        delete this;
    }

   // Component::SafePointer<MessageBoxOwnerComponent> owner;
};


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
  juce::Rectangle < int > b = getLocalBounds();
  toolbar->setBounds(b.removeFromTop(50));
  imageview->setBounds(b);
  capture->setBounds(b);
  annotate->setBounds(b);

}

void MainComponent::loadImageDialog(const FileChooser& fc){

  File result = fc.getResult();
  if (result.exists()){
    loadImage(result.getFullPathName().toStdString());
    imageview->setImage(MRimage);
    capture->setImage(MRimage);
    annotate->setImage(MRimage);
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
    const ScopedLock lock (mutex);
    imageview->setVisible(true);
    imageview->fixAspectRatio();
    capture->setVisible(false);
    annotate->setVisible(false);
  }
  
  if(event == "capture"){
    const ScopedLock lock (mutex);
    capture->setVisible(true);
    capture->fixAspectRatio();
    imageview->setVisible(false);
    annotate->setVisible(false);
  }
  
  if(event == "annotate"){
    const ScopedLock lock (mutex);
    annotate->setVisible(true);
    annotate->fixAspectRatio();
    imageview->setVisible(false);
    capture->setVisible(false);
  }
  
  
}
