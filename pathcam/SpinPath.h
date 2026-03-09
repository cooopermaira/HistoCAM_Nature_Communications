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

class SerialStream : public Poco::Runnable {
public:
  SpinPath* parent;
  // std::atomic<bool> interrupt{false};

  SerialStream(SpinPath* parent) : parent(parent){}
  ~SerialStream() override {}
  void run() override;
};


class SpinPath{
  friend class CameraStream;
  friend class FileStream;
  friend class ProcessStream;
  friend class SerialStream;
private:
  CameraPtr pCam;
  SystemPtr system;
  CameraList camList;
  
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
  
  Poco::Path root_path;
  string captureSetName;
  
  Poco::Logger& camlogger;
  AutoPtr<SimpleFileChannel> camChannel;
  
  Poco::Logger& IOlogger;
  AutoPtr<SimpleFileChannel> IOChannel;
 
 CameraStream * cameraStream;
 FileStream  * fileStream;
 ProcessStream* processStream;
 SerialStream* serialStream{nullptr};

  Poco::Thread thread_cam, thread_file, thread_sCam, thread_serial;
  
  Poco::FastMutex cache_mutex;
  Poco::FastMutex caputure_set_mutex;
  
  std::queue < Image* > cache;

  int serial_fd{-1}; // linux fd for /dev/ttyUSB0
  std::mutex label_mu;
  std::string latest_label; //make latest label always start at 2 will correct whenever obj is changed


  size_t thread_safe_cache_size() {
    Poco::FastMutex::ScopedLock lock(cache_mutex);
    return cache.size();
  }
  
public:
  std::atomic<bool> cameraDone = false;
  std::shared_ptr<StreamCam> sCam;
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

  std::string getObjectiveLabel() {     // thread safe label storage
    std::lock_guard<std::mutex> lk(label_mu);
    return latest_label;
  }

  std::shared_ptr < MRTiledImageSet > get_image_reference() {
    return sCam->get_MRimage_reference();
  }
  
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
