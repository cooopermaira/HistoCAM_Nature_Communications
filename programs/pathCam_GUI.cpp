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


#include "pathCam.h"


using namespace nanogui;
using namespace pathCam;

class ExampleApplication : public Screen {
  nanogui::ref<Window> capture_window;
  Button *capture_button, *capture_set_button;
//  Button *circleButton;
  bool capturing;
  
//#ifdef WITH_SPINNAKER
  SpinPath *camera;
//#endif
  
public:
  ExampleApplication() : Screen(Vector2i(512, 768), "pathCam") {
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

//#ifdef WITH_SPINNAKER
    camera = new SpinPath();
    Poco::Path root_path = Poco::Path("D:/pcamTest");
    camera->setRootPath(root_path);
    camera->newCaptureSet();
//#endif
    capturing = false;

  }
  
  ~ExampleApplication(){
//#ifdef WITH_SPINNAKER
    delete camera;
//#endif
  }
  
  void openCaptureWindow() {
    capture_window = new Window(this, "Capture Toolbox");
    capture_window->set_position(Vector2i(150, 30));
    capture_window->set_layout(new GroupLayout());

    
    capture_set_button = new Button(capture_window, "New Capture Set");
    capture_set_button->set_callback([this] {
//#ifdef WITH_SPINNAKER
      camera->newCaptureSet();
//#endif
    });
    new Label(capture_window, "", "sans-bold");
//    Widget * tools = new Widget(capture_window);
//    tools->set_layout(new BoxLayout(Orientation::Horizontal,
//                                   Alignment::Middle, 0, 6));
    capture_button = new Button(capture_window, "Capture");
    capture_button->set_flags(Button::ToggleButton);
    if(capturing){ capture_button->set_pushed(true); }
    capture_button->set_change_callback([this](bool state) {
//#ifdef WITH_SPINNAKER
      if(state){
        capturing = camera->startCamera();
      }else{
        capturing = false;
        camera->stopCamera();
      }
//#endif
    });
    

    perform_layout();
    
  }

  
  virtual bool keyboard_event(int key, int scancode, int action, int modifiers) {
    if (Screen::keyboard_event(key, scancode, action, modifiers))
      return true;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      set_visible(false);
      return true;
    }
    if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
        std::cout << "Press\n";
        camera->newCaptureSet();
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS){
      if(capturing){
        if(capture_button){ capture_button->set_pushed(false); }
        capturing = false;
//#ifdef WITH_SPINNAKER
        camera->stopCamera();
//#endif
      }else{
        if(capture_button){ capture_button->set_pushed(true); }
        capturing = true;
//#ifdef WITH_SPINNAKER
        camera->startCamera();
//#endif
      }
      return true;
    }
    return false;
  }
  
  virtual void draw(NVGcontext *ctx) override {
    /* Animate the scrollbar */
    //        m_progress->set_value(std::fmod((float) glfwGetTime() / 10, 1.0f));
    
    /* Draw the user interface */
    Screen::draw(ctx);
    
//    if(circleButton){
//      nvgBeginPath(ctx);
//      nvgCircle(ctx, capture_window->position().x() + circleButton->position().x() + circleButton->width() / 2.0f,
//                capture_window->position().y() + circleButton->position().y() + circleButton->height() / 2.0f,
//                20.0f);
//      
//      if (capturing) {
//        nvgFillColor(ctx, nvgRGBf(0.0f, 1.0f, 0.0f)); // Green color
//      } else {
//        nvgFillColor(ctx, nvgRGBf(1.0f, 0.0f, 0.0f)); // Red color
//      }
//      
//      nvgFill(ctx);
//    }
  }
  

private:
  ProgressBar *m_progress;
};

int main(int /* argc */, char ** /* argv */) {
  try {
    nanogui::init();
    

    
    /* scoped variables */ {
      nanogui::ref<ExampleApplication> app = new ExampleApplication();
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
