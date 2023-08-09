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

using Poco::Logger;
using Poco::SimpleFileChannel;
using Poco::AutoPtr;

namespace pathCam{

class SpinPath;

class CameraStream: public Poco::Runnable{
public:
  SpinPath * parent;
  bool interrupt;
  CameraStream(SpinPath * parent): parent(parent) {};
  virtual void run();
};





class SpinPath{
  friend class CameraStream;
private:
  CameraPtr pCam;
  SystemPtr system;
  CameraList camList;
  
  Poco::Path root_path;
  
  Logger& camlogger;
  AutoPtr<SimpleFileChannel> pChannel;
  
  Poco::FastMutex cache_mutex;
  
  typedef struct cache_element{
    pathCam::Image * image;
    std::string name;
  } cache_element;
  
  std::queue < cache_element > cache;
  
public:
  
  SpinPath();
  ~SpinPath();
  int RunCamera();
  
  
  
  void setRootPath(Poco::Path _root_path){ root_path = _root_path; }
  Poco::Path getRootPath(){ return root_path; }
  
  void interruptCapture(){ interrupt = true; }
  
private:
  
  bool interrupt;
    
  int spinUpCamera();
  int FileIOThread();
  
  void spinDownCamera();
  
  int ResetGVCPHeartbeat(){
    return ConfigureGVCPHeartbeat(true);
  }
  
  //    int DisableGVCPHeartbeat(){
  //        return ConfigureGVCPHeartbeat(false);
  //    }
  
  int ConfigureGVCPHeartbeat(bool enable);
  int PrintDeviceInfo(INodeMap& nodeMap);
  
  
};


};


#endif /* Spinnaker_h */
