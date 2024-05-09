/*
 ==============================================================================
 
 This file contains the basic startup code for a JUCE application.
 
 ==============================================================================
 */

#include "JuceHeader.h"


//==============================================================================
class PathCamApplication  : public juce::JUCEApplication
{
public:
  //==============================================================================
  PathCamApplication() {}
  
  const juce::String getApplicationName() override       { return ProjectInfo::projectName; }
  const juce::String getApplicationVersion() override    { return ProjectInfo::versionString; }
  bool moreThanOneInstanceAllowed() override             { return true; }
  
  //==============================================================================
  void initialise (const juce::String& commandLine) override
  {
    StringArray args = StringArray::fromTokens(commandLine, true);
    
    Poco::Util::LayeredConfiguration::Ptr config;
    
    Poco::AutoPtr<Poco::ConsoleChannel> consoleChannel(new Poco::ConsoleChannel);
    Poco::Logger& logger = Poco::Logger::create("PathCamLogger", consoleChannel, Poco::Message::PRIO_INFORMATION);

    logger.information("Poco Logger Initialized.");

    
    for (auto arg : args)
    {
      if (arg.startsWith("--config-file="))
      {
        juce::String value = arg.substring(14);
        try {
          Poco::AutoPtr<Poco::Util::XMLConfiguration> pXMLConfig(new Poco::Util::XMLConfiguration(value.toStdString()));
          config = new Poco::Util::LayeredConfiguration;
          config->add(pXMLConfig);
        }catch (const Poco::Exception& ex) {
          // Handle any errors here
          std::cerr << "Error loading configuration: " << ex.displayText() << std::endl;
        }
      }
    }
    
    mainWindow.reset (new MainWindow (getApplicationName(), config));
  }
  
  void shutdown() override
  {
    // Add your application's shutdown code here..
    
    mainWindow = nullptr; // (deletes our window)
  }
  
  //==============================================================================
  void systemRequestedQuit() override
  {
    // This is called when the app is being asked to quit: you can ignore this
    // request and let the app carry on running, or call quit() to allow the app to close.
    quit();
  }
  
  void anotherInstanceStarted (const juce::String& commandLine) override
  {
    // When another instance of the app is launched while this one is running,
    // this method is invoked, and the commandLine parameter tells you what
    // the other instance's command-line arguments were.
  }
  
  //==============================================================================
  /*
   This class implements the desktop window that contains an instance of
   our MainComponent class.
   */
  class MainWindow    : public juce::DocumentWindow
  {
  public:
    MainWindow (juce::String name, Poco::Util::LayeredConfiguration::Ptr config)
    : DocumentWindow (name,
                      juce::Desktop::getInstance().getDefaultLookAndFeel()
                      .findColour (juce::ResizableWindow::backgroundColourId),
                      DocumentWindow::allButtons)
    {
      
      
      bcam.reset( new pathCam::StreamCam(config) );
      
      setUsingNativeTitleBar (true);
      setContentOwned (new MainComponent(bcam), true);
      
#if JUCE_IOS || JUCE_ANDROID
      setFullScreen (true);
#else
      setResizable (true, true);
      centreWithSize (getWidth(), getHeight());
#endif
      
      setVisible (true);
    }
    
    void closeButtonPressed() override
    {
      // This is called when the user tries to close this window. Here, we'll just
      // ask the app to quit when this happens, but you can change this to do
      // whatever you need.
      JUCEApplication::getInstance()->systemRequestedQuit();
    }
    
    /* Note: Be careful if you override any DocumentWindow methods - the base
     class uses a lot of them, so by overriding you might break its functionality.
     It's best to do all your work in your content component instead, but if
     you really have to override any DocumentWindow methods, make sure your
     subclass also calls the superclass's method.
     */
    
  private:
    std::shared_ptr< pathCam::StreamCam > bcam;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainWindow)
  };
  
private:
  
  std::unique_ptr<MainWindow> mainWindow;
};

//==============================================================================
// This macro generates the main() routine that launches the app.
START_JUCE_APPLICATION (PathCamApplication)
