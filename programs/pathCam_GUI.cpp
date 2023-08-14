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
  SpinPath *camera;
  
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
    
    camera = new SpinPath();
    Poco::Path root_path = Poco::Path("D:/pcamTest");
    camera->setRootPath(root_path);
    camera->newCaptureSet();

  }
  
  ~ExampleApplication(){
    delete camera;
  }
  
  void openCaptureWindow() {
    capture_window = new Window(this, "Capture Toolbox");
    capture_window->set_position(Vector2i(150, 30));
    capture_window->set_layout(new GroupLayout());

    
    Button * b = new Button(capture_window, "New Capture Set");
    b->set_callback([this] { camera->newCaptureSet(); });
    new Label(capture_window, "", "sans-bold");
    
    b = new Button(capture_window, "Capture");
    b->set_flags(Button::ToggleButton);
    b->set_change_callback([this](bool state) {
      if(state){
        camera->startCamera();
      }else{
        camera->stopCamera();
      }
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
    return false;
  }
  
  virtual void draw(NVGcontext *ctx) {
    /* Animate the scrollbar */
    //        m_progress->set_value(std::fmod((float) glfwGetTime() / 10, 1.0f));
    
    /* Draw the user interface */
    Screen::draw(ctx);
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
