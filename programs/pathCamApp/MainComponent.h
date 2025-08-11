#pragma once

#include "JuceHeader.h"
//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
class MainComponent final : public juce::Component, public DataObserver
{
  friend class LoadingThread;
    
public:
  //==============================================================================
  //MainComponent(std::shared_ptr< pathCam::StreamCam > bcam);
    MainComponent(Poco::Util::LayeredConfiguration::Ptr config);
  ~MainComponent() override;
  unsigned int ImageWidth, ImageHeight;
  pathCam::StreamCam* sCam;
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override;
    
  void GuiEventHandler(std::string event);

    ImageViewComponent * imageview;
    std::shared_ptr< MRTiledImageSet > MRimage;
    std::shared_ptr< MRTiledImage >  imagePyramid;
    CaptureComponent * capture;
    AnnotateComponent * annotate;
  
  inline void update() override;

  inline void notify_new_data(){
    clientHasData = true;
  }
  


  inline void refreshImage(){
    imageview->refreshImage();
    capture->refreshImage();
    annotate->refreshImage();
  }
  
private:
  
  void loadImage(std::string path);
  void loadImageDialog(const FileChooser& fc);

  Poco::Util::LayeredConfiguration::Ptr config;
  //std::shared_ptr< pathCam::StreamCam > bcam;

    std::unique_ptr<FileChooser> fc;
  
  ToolbarComponent * toolbar;

    //Shared view

  
  juce::ProgressBar * progressBar;

  CriticalSection mutex;
  
  StringArray iconNames;
  OwnedArray<Drawable> iconsFromZipFile;

  std::atomic<bool> clientHasData;

public:
  std::shared_ptr < fRectangle > view;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
