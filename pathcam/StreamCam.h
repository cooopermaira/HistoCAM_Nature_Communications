//
//  StreamCam.h
//  pathCamLib
//
//  Created by cooper maira on 11/24/23.
//

#ifndef StreamCam_h
#define StreamCam_h

#include <stdio.h>
#include "pathCam.h"

using Poco::MemoryPool;
using Poco::Path;
using Poco::Logger;

namespace pathCam{
class Composite;
class CompositeManager;

class StreamCam: public BatchCam{
  
  friend class FeaturesRunnable;
  friend class MatchRunnable;
  friend class DiskStreamer;
  friend class Loader;
  friend class QManager;
  friend class RegManager;
  friend class Composite;
  friend class CompositeManager;
  
private:
  std::queue < std::vector < RegInfo > > compositeQ;
  
  
public:
  StreamCam(Poco::Util::LayeredConfiguration::Ptr config);
  
  ~StreamCam(){delete buffer_mutex, delete image_mutex;}
  
  Poco::FastMutex *buffer_mutex;
  Poco::FastMutex *image_mutex;
  Poco::FastMutex *compositeQ_mutex;
  
  bool run();
  

protected:
  unsigned int increment_and_get_components(){return components++;}
  unsigned long int add_image(Image*);
  
  Image* get_image_ref(unsigned long int);
  
  std::vector < Image* > get_image_refs(std::vector<unsigned long int>);
  std::vector < RegInfo > get_Q_front();
  
  bool compositeQ_empty();
  bool resolve_bboxes();
  
  void push_compositeQ(std::vector < RegInfo >);
  void reg_spanning_tree(unsigned int root_idx, Vec2 offset);
  void add_new_component(unsigned long int);
  
  std::vector < Composite* > composites;
  std::vector < bool > visited;
  

  std::queue < std::string > disk_image;
  std::queue < char* > buffer;
  
  std::atomic < bool > reg_complete = false;
  std::atomic < bool > disk_empty = false;
  std::atomic < bool > jobs_queued = false;
  std::atomic < unsigned int > components = 0;
  
  pathCam::ConsecQ RegistrationConsecQ;
  
  
};

}
#endif /* StreamCam_hpp */
