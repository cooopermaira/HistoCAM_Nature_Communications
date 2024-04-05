//
//  Runnables.hpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#ifndef Runnables_h
#define Runnables_h

#include <stdio.h>
#include "pathCam.h"
#include "Poco/Runnable.h"

namespace pathCam{
class JobQueue;


class CompositeManager: public Poco::Runnable{
private:
  StreamCam *parent;
  
public:
  bool successful;
  
  CompositeManager(StreamCam *parent);
  
  virtual void run();
  
};

class RegistrationRunnable : public Poco::Runnable {
private:
    StreamCam* parent;
    unsigned long index;
public:
    RegistrationRunnable(StreamCam* parent,unsigned long index) : parent(parent), index(index) {};
    
    virtual void run();
    std::pair<bool, Vec2> trace_to_root(unsigned long index);
};

//loader class takes data from disk streamer/microscope and prepares matchable jobs
class LoaderLogicRunnable : public Poco::Runnable {
private:
    StreamCam* parent;
    Image* image;

public:
    bool successful;
    LoaderLogicRunnable(StreamCam* parent, Image* image) : image(image), parent(parent), successful(true) {};

    virtual void run();
};

/*
//takes matchable pairs and computes final registration locations
class RegManager: public Poco::Runnable{
private:
  pathCam::StreamCam *parent;
  pathCam::JobQueue *queue;
  unsigned int current_index = 0;
  
public:
  bool successful;
  
  RegManager(pathCam::StreamCam *parent, pathCam::JobQueue *queue);
  
  virtual void run();
  
};
*/



class QManager: public Poco::Runnable{
private:
  StreamCam *parent;
  
public:
  QManager(StreamCam *parent);
  
  virtual void run();
  
};



class DiskStreamer: public Poco::Runnable{
private:
  StreamCam *parent;
  
public:
  bool successful;
  
  DiskStreamer(StreamCam *parent);
  
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
