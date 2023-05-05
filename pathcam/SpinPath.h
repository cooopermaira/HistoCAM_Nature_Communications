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
  private:
    CameraPtr pCam;
    SystemPtr system;
    CameraList camList;
    
    Poco::Path path;
        
  public:
    
    SpinPath();
    ~SpinPath();
    int RunCamera();
    
    void setFolder(Poco::Path _path){ path = _path; }
    Poco::Path getPath(){ return path; }
    
  private:
    
    int AcquireImages(INodeMap& nodeMap, INodeMap& nodeMapTLDevice);
    
    int ResetGVCPHeartbeat(){
        return ConfigureGVCPHeartbeat(true);
    }

    int DisableGVCPHeartbeat(){
        return ConfigureGVCPHeartbeat(false);
    }
    
    int ConfigureGVCPHeartbeat(bool enable);
    int PrintDeviceInfo(INodeMap& nodeMap);

    
  };
  
}


#endif /* Spinnaker_h */
