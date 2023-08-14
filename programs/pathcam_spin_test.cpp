//
//  pathcam_spin_test.cpp
//  pathCam
//
//  Created by Brian Summa on 4/28/23.
//

#include "pathCam.h"

using namespace std;
using namespace pathCam;

int main( int argc, char* argv[] ){
  
  SpinPath *camera = new SpinPath();
  
  Poco::Path root_path = Poco::Path("D:/pcamTest");
  
  camera->setRootPath(root_path);
  
  camera->newCaptureSet();
  
  camera->startCamera();
  Poco::Thread::sleep(5000);
  camera->stopCamera();
  
  
  delete camera;
  
   
  return 0;
}

