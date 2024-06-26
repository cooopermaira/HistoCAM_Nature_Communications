/*
 src/example1.cpp -- C++ version of an example application that shows
 how to use the various widget classes. For a Python implementation, see
 '../python/example1.py'.
 
 NanoGUI was developed by Wenzel Jakob <wenzel.jakob@epfl.ch>.
 The widget drawing code is based on the NanoVG demo application
 by Mikko Mononen.
 
 All rights reserved. Use of this source code is governed by a
 BSD-style license that can be found in the LICENSE.txt file.
 */

#include <nanogui/opengl.h>
#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/label.h>
#include <nanogui/button.h>
#include <nanogui/toolbutton.h>
#include <nanogui/popupbutton.h>
#include <nanogui/progressbar.h>
#include <SFML/Audio.hpp>


#include "pathCam.h"

using Poco::Util::Application;
using Poco::Util::Option;
using Poco::Util::OptionSet;
using Poco::Util::HelpFormatter;
using Poco::Util::AbstractConfiguration;
using Poco::Util::OptionCallback;
using Poco::AutoPtr;

using namespace nanogui;
using namespace pathCam;

class BeginSound: public Poco::Runnable {
public:
  sf::Sound *sound;
  sf::SoundBuffer *buffer;
  
  BeginSound():sound(0), buffer(0){
    
    std::stringstream ss;
    ss <<  PROJECT_SOURCE_DIR << "/resources/start.wav";

    Poco::Path start_wav_path = Poco::Path(ss.str());
    
    buffer = new sf::SoundBuffer();
      
    if (buffer->loadFromFile(start_wav_path.toString())){
      sound = new sf::Sound(*buffer);
    }
  }

  ~BeginSound(){
    if(sound){delete sound;}
    if(buffer){delete buffer;}
  }
  
  virtual void run(){
    if(sound){
      sound->play();
      while (sound->getStatus() == sf::Sound::Playing){
        // Leave some CPU time for other processes
        Poco::Thread::sleep(100);
      }
    }
  }
};

class EndSound: public Poco::Runnable {
public:
  sf::Sound *sound;
  sf::SoundBuffer *buffer;
  
  EndSound():sound(0), buffer(0){
    
    std::stringstream ss;
    ss <<  PROJECT_SOURCE_DIR << "/resources/stop.wav";

    Poco::Path start_wav_path = Poco::Path(ss.str());
    
    buffer = new sf::SoundBuffer();
      
    if (buffer->loadFromFile(start_wav_path.toString())){
      sound = new sf::Sound(*buffer);
    }
  }

  ~EndSound(){
    if(sound){delete sound;}
    if(buffer){delete buffer;}
  }
  
  virtual void run(){
    if(sound){
      sound->play();
      while (sound->getStatus() == sf::Sound::Playing){
        // Leave some CPU time for other processes
        Poco::Thread::sleep(100);
      }
    }
  }
};



class PathCamApplication : public Screen, public Application {
private:
  nanogui::ref<Window> capture_window;
  Button *capture_button, *capture_set_button;
  
  BeginSound start;
  Poco::Thread thread_start;
  
  EndSound stop;
  Poco::Thread thread_stop;
  
  bool capturing;
  
#ifdef WITH_SPINNAKER
  SpinPath *camera;
#endif
  
  bool _helpRequested;
  
  ProgressBar *m_progress;
  
public:
  PathCamApplication() : Screen(Vector2i(512, 768), "pathCam"), _helpRequested(false){
    inc_ref();
    Window *window = new Window(this, "Toolbox");
    window->set_position(Vector2i(15, 15));
    window->set_layout(new GroupLayout());
    
    Button * b = new Button(window, "Capture");
    b->set_flags(Button::ToggleButton);
    b->set_change_callback([this](bool state) {  // Capture 'this' pointer
      std::cout << "Toggle button state: " << state << std::endl;
      if(state==1){
        openCaptureWindow();
      }else{
        capture_window->dispose();
      }
    });
    
    perform_layout();
    
#ifdef WITH_SPINNAKER
    //camera = new SpinPath();
    Poco::Path root_path = Poco::Path("D:/pcamTest");
    camera->setRootPath(root_path);
    camera->newCaptureSet();
#endif
    capturing = false;
    
  }
  
  ~PathCamApplication(){
#ifdef WITH_SPINNAKER
    delete camera;
#endif
  }
  
  void openCaptureWindow() {
    capture_window = new Window(this, "Capture Toolbox");
    capture_window->set_position(Vector2i(150, 30));
    capture_window->set_layout(new GroupLayout());
    
    
    capture_set_button = new Button(capture_window, "New Capture Set");
    capture_set_button->set_callback([this] {
#ifdef WITH_SPINNAKER
      camera->newCaptureSet();
#endif
    });
    new Label(capture_window, "", "sans-bold");
    capture_button = new Button(capture_window, "Capture");
    capture_button->set_flags(Button::ToggleButton);
    if(capturing){ capture_button->set_pushed(true); }
    capture_button->set_change_callback([this](bool state) {
#ifdef WITH_SPINNAKER
      if(state){
        capturing = (camera->run() != -1);
      }else{
        capturing = false;
        camera->stopCamera();
      }
#endif
    });
    
    
    perform_layout();
    
  }
  
  
  virtual bool keyboard_event(int key, int scancode, int action, int modifiers) override {
    if (Screen::keyboard_event(key, scancode, action, modifiers))
      return true;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      set_visible(false);
      return true;
    }
    if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
#ifdef WITH_SPINNAKER
      std::cout << "Press\n";
      camera->newCaptureSet();
#endif
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS){
      if(capturing){
        thread_start.start(stop);
        //I'm assuming this will just stop when the thread is over, so no blocking
      }else{
        thread_start.start(start);
      }
      
      if(capturing){
        if(capture_button){ capture_button->set_pushed(false); }
        capturing = false;
#ifdef WITH_SPINNAKER
        camera->stopCamera();
#endif
      }else{
#ifdef WITH_SPINNAKER
        int result = camera->run();
#else
        int result = -1;
#endif
        if(result != -1){
          capturing = true;
          if(capture_button){ capture_button->set_pushed(true); }
        }
        
      }
      return true;
    }
    return false;
  }
  
  virtual void draw(NVGcontext *ctx) override {
    /* Draw the user interface */
    Screen::draw(ctx);
  }
  
protected:
  
  void defineOptions(OptionSet& options) override{
    Application::defineOptions(options);

    options.addOption(
      Option("help", "h", "display help information on command line arguments")
        .required(false)
        .repeatable(false)
        .callback(OptionCallback<PathCamApplication>(this, &PathCamApplication::handleHelp)));
        
    options.addOption(
      Option("config-file", "c", "load configuration data from a file")
        .required(false)
        .repeatable(true)
        .argument("file")
        .callback(OptionCallback<PathCamApplication>(this, &PathCamApplication::handleConfig)));

  }
  
  void handleHelp(const std::string& name, const std::string& value){
    _helpRequested = true;
    displayHelp();
    stopOptionsProcessing();
  }
  
  
  void handleConfig(const std::string& name, const std::string& value){
    loadConfiguration(value);
  }
    
  void displayHelp(){
    HelpFormatter helpFormatter(options());
    helpFormatter.setCommand(commandName());
    helpFormatter.setUsage("OPTIONS");
    helpFormatter.setHeader("Pathcam batch image processing application.");
    helpFormatter.format(std::cout);
  }
  
};

int main(int argc , char ** argv) {
  try {
    nanogui::init();
    
    /* scoped variables */ {
      nanogui::ref<PathCamApplication> app = new PathCamApplication();
      
      app->init(argc, argv);
      app->dec_ref();
      app->draw_all();
      app->set_visible(true);
      nanogui::mainloop(1 / 60.f * 1000);
    }
    
    nanogui::shutdown();
  } catch (const std::exception &e) {
    std::string error_msg = std::string("Caught a fatal error: ") + std::string(e.what());
#if defined(_WIN32)
    MessageBoxA(nullptr, error_msg.c_str(), NULL, MB_ICONERROR | MB_OK);
#else
    std::cerr << error_msg << std::endl;
#endif
    return -1;
  } catch (...) {
    std::cerr << "Caught an unknown error!" << std::endl;
  }
  
  return 0;
}
