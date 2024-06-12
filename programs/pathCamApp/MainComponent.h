#pragma once

#include "JuceHeader.h"
//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
class MainComponent final : public juce::Component
{
  friend class LoadingThread;
    
public:
  //==============================================================================
  //MainComponent(std::shared_ptr< pathCam::StreamCam > bcam);
    MainComponent(Poco::Util::LayeredConfiguration::Ptr config);
  ~MainComponent() override;
  
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override;
    
  void GuiEventHandler(std::string event);

    ImageViewComponent * imageview;
    std::shared_ptr< MRTiledImage >  MRimage;
    CaptureComponent * capture;
    AnnotateComponent * annotate;
  
  void refreshImage(){
    repaint();
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
  
public:
  std::shared_ptr < fRectangle > view;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
