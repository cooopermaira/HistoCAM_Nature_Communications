//
//  Spinnaker.h
//  pathCamLib
//
//  Created by Brian Summa on 4/27/23.
//

#ifndef Spinnaker_h
#define Spinnaker_h

#include "pathCam.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

namespace pathCam{
  
  class SpinPath{
    CameraPtr pCam;
    SystemPtr system;
    CameraList camList;
        
  public:
    
    SpinPath();
    ~SpinPath(){
      pCam = nullptr;
      camList.Clear();
      system->ReleaseInstance();
    }
    
    int ResetGVCPHeartbeat(){
        return ConfigureGVCPHeartbeat(true);
    }

    int DisableGVCPHeartbeat(){
        return ConfigureGVCPHeartbeat(false);
    }
    
  private:
    
    int ConfigureGVCPHeartbeat(bool enable);
    int PrintDeviceInfo(INodeMap& nodeMap);
    int AcquireImages(INodeMap& nodeMap, INodeMap& nodeMapTLDevice);
    int RunCamera();
    
  };
  
}


#endif /* Spinnaker_h */
