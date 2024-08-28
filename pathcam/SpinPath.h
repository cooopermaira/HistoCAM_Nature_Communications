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
using Poco::Util::LayeredConfiguration;

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
  ~CameraStream() {};
  virtual void run();
};


class FileStream: public Poco::Runnable{
public:
  SpinPath * parent;
  bool interrupt;
  FileStream(SpinPath * parent): parent(parent) {};
  ~FileStream() {};
  virtual void run();
};

class ProcessStream : public Poco::Runnable {
public:
    SpinPath* parent;
    bool interrupt;
    ProcessStream(SpinPath* parent) : parent(parent) {};
    ~ProcessStream() {};
    virtual void run();
};

class SpinPath{
  friend class CameraStream;
  friend class FileStream;
  friend class ProcessStream;
private:
  CameraPtr pCam;
  SystemPtr system;
  CameraList camList;
  
  
  
  Poco::Path root_path;
  string captureSetName;
  
  Poco::Logger& camlogger;
  AutoPtr<SimpleFileChannel> camChannel;
  
  Poco::Logger& IOlogger;
  AutoPtr<SimpleFileChannel> IOChannel;
 
 CameraStream * cameraStream;
 FileStream  * fileStream;
 ProcessStream* processStream;

  Poco::Thread thread_cam, thread_file, thread_sCam;
  
  Poco::FastMutex cache_mutex;
  Poco::FastMutex caputure_set_mutex;
  
  std::queue < cache_element > * cache;
  
  size_t thread_safe_cache_size(){
    size_t result;
    cache_mutex.lock();
    result = cache->size();
    cache_mutex.unlock();
    return result;
  }
  
public:
    StreamCam* sCam;
  SpinPath(LayeredConfiguration::Ptr config);
  ~SpinPath();
  int run();
  void stopCamera();

  //void set_MainComponent_reference(MainComponent* parent) { sCam->set_MainComponent_reference(parent); };
  void newCaptureSet();
  
  void setRootPath(Poco::Path _root_path){ root_path = _root_path; }
  Poco::Path getRootPath(){ return root_path; }
  
  void interruptCapture(){ interrupt = true; }

  void add_observer(DataObserver* new_observer) {
      sCam->add_observer(new_observer);
  }

  std::shared_ptr < MRTiledImageSet > get_image_reference() { return sCam->get_image_reference(); }
  
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
