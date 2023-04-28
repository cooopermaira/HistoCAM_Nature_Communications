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
  
  Poco::Path path = Poco::Path("/Users/bsumma/Documents/pcamTest");
  
  camera->setFolder(path);
  
  camera->RunCamera();
  
  delete camera;
  
   
  return 0;
}

