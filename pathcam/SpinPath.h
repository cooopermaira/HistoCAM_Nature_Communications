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

typedef struct cache_element{
  pathCam::Image * image;
  std::string name;
} cache_element;


class CameraStream: public Poco::Runnable{
public:
  SpinPath * parent;
  bool interrupt;
  CameraStream(SpinPath * parent): parent(parent) {};
  virtual void run();
};


class FileStream: public Poco::Runnable{
public:
  SpinPath * parent;
  bool interrupt;
  FileStream(SpinPath * parent): parent(parent) {};
  virtual void run();
};


class SpinPath{
  friend class CameraStream;
  friend class FileStream;
private:
  CameraPtr pCam;
  SystemPtr system;
  CameraList camList;
  
  Poco::Path root_path;
  string captureSetName;
  
  Logger& camlogger;
  AutoPtr<SimpleFileChannel> camChannel;
  
  Logger& IOlogger;
  AutoPtr<SimpleFileChannel> IOChannel;
 
  
  Poco::FastMutex cache_mutex;
  Poco::FastMutex caputure_set_mutex;
  
  std::queue < cache_element > cache;
  
  size_t thread_safe_cache_size(){
    size_t result;
    cache_mutex.lock();
    result = cache.size();
    cache_mutex.unlock();
    return result;
  }
  
public:
  
  SpinPath();
  ~SpinPath();
  int StartCamera();
  void StopCamera();

  
  void newCaptureSet();
  
  void setRootPath(Poco::Path _root_path){ root_path = _root_path; }
  Poco::Path getRootPath(){ return root_path; }
  
  void interruptCapture(){ interrupt = true; }
  
private:
  
  bool interrupt;
    
  int spinUpCamera();
  
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
