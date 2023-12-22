//
//  StreamCam.hpp
//  pathCamLib
//
//  Created by cooper maira on 11/24/23.
//

#ifndef StreamCam_hpp
#define StreamCam_hpp

#include <stdio.h>
#include "pathCam.h"

using Poco::MemoryPool;
using Poco::Path;
using Poco::Logger;

namespace pathCam{

class StreamCam: public BatchCam{
  
  friend class FeaturesRunnable;
  friend class MatchRunnable;
  friend class DiskStreamer;
  friend class Loader;
  friend class QManager;
  friend class RegManager;
public:
  StreamCam(Poco::Util::LayeredConfiguration::Ptr config);
  
  ~StreamCam(){delete buffer_mutex, delete image_mutex;}
  
  Poco::FastMutex *buffer_mutex;
  Poco::FastMutex *image_mutex;
  
  bool run();
  

protected:
  unsigned long int add_image(Image*);
  Image* get_image_ref(unsigned long int);
  bool resolve_bboxes();
  void reg_spanning_tree(unsigned int root_idx, Vec2 offset);
  
  std::vector < bool > visited;
  std::queue <std::string> disk_image;
  std::queue <char*> buffer;
  std::atomic <bool> disk_empty = false;
  std::atomic <bool> jobs_queued = false;
  pathCam::ConsecQ ConseqQ;
  Vec2 offset_to_root;
  Vec2 Bbox_max;
  
};

}
#endif /* StreamCam_hpp */
