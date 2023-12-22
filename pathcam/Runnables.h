//
//  Runnables.hpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#ifndef Runnables_h
#define Runnables_h

#include <stdio.h>
//#include <pathCam.h>
#include "Poco/Runnable.h"

namespace pathCam{
class JobQueue;

class QManager: public Poco::Runnable{
private:
  StreamCam *parent;
  JobQueue *queue;
  
public:
  QManager(StreamCam *parent, JobQueue *queue);
  
  virtual void run();
  
};

class DiskStreamer: public Poco::Runnable{
private:
  StreamCam *parent;
  
public:
  
  bool successful;
  
  DiskStreamer(StreamCam *parent);
  //~DiskStreamer();
  
  virtual void run();
  };



class MatchRunnable: public Poco::Runnable{
private:
  StreamCam *parent;
  unsigned long int image_idx;
  
public:
  
  bool successful;
  
  
  MatchRunnable(StreamCam *parent, unsigned long int image_idx);
  
  virtual void run();
  
  };
}
#endif /* Runnables_h */
