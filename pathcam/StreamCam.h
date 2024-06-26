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

class MRTiledImage;

namespace pathCam{
class Composite;
class CompositeVoronoi;
class CompositeManager;
class JobQueue;


class StreamCam: public BatchCam{
  
  friend class FeaturesRunnable;
  friend class MatchRunnable;
  friend class DiskReader;
  friend class DiskStreamer;
  friend class Loader;
  friend class SpinLoader;
  friend class QManager;
  friend class RegManager;
  friend class Composite;
  friend class CompositeVoronoi;
  friend class CompositeManager;
  friend class LoaderLogicRunnable;
  friend class RegistrationRunnable;
  friend class SingleMatchRunnable;
  friend class XCompRunnable;
  
private:
std::queue < std::vector < RegInfo > > compositeBatch;

  
public:
  StreamCam(Poco::Util::LayeredConfiguration::Ptr config);
  
  ~StreamCam(){delete buffer_mutex, delete image_mutex;}

  Poco::FastMutex *resize_mmatch_mutex;
  Poco::FastMutex *resize_buffer_mutex;
  Poco::FastMutex *buffer_mutex;
  Poco::FastMutex *image_mutex;
  Poco::FastMutex *compositeQ_mutex;
  Poco::FastMutex *component_mutex;

  std::shared_ptr< MRTiledImage >  imagePyramid;

  bool run();
  bool spin_run();
  void pass_image(Image*, unsigned long sort_order = 0);
  void set_match(unsigned long image_idx, unsigned long prev_idx);

  bool microscopeInput;

  void set_image_reference(std::shared_ptr< MRTiledImage >  MRImage);

  void add_observer(DataObserver * new_observer){
    observers.push_back(new_observer);
  }
  
  void update_observers(){
    for(unsigned int i=0; i < observers.size(); i++){
      observers[i]->update();
    }
  }

protected:

  unsigned int increment_and_get_components(){return components++;}
  unsigned long add_image(Image* image);
  void add_image(Image* image, unsigned long index);
  JobQueue* JobQ;

  Image* get_image_ref(unsigned long int);
  
  std::vector < Image* > get_image_refs(std::vector<unsigned long int>);
  std::vector < RegInfo > get_Q_front();
  Image* get_Q_front_Spin();
  
  bool compositeQ_empty();

  void push_compositeQ(RegInfo index);
  //void reg_spanning_tree(unsigned int root_idx, Vec2 offset);
  void add_new_component(unsigned long image_index, cv::Size image_size);

  //std::vector < double > variancesForDebug;
  std::vector < CompositeVoronoi* > composites;
  std::vector < bool > visited;
  

  std::queue < std::string > disk_image;
  std::queue < char* > buffer;
  std::queue < Image* > spin_image_buffer;

  std::atomic<bool> compositing = true;
  std::atomic < bool > reg_complete = false;
  std::atomic < bool > disk_empty = false;
  std::atomic < bool > jobs_queued = false;
  std::atomic < unsigned int > components = 0;
  std::atomic < unsigned int > diskCount = 0;
  std::atomic < unsigned int > loaderCount = 0;
  std::atomic < unsigned int > matchableCount = 0;
  std::atomic < unsigned int > regCount = 0;
  
  pathCam::ConsecQ RegistrationConsecQ;
  
  std::vector < DataObserver *> observers;

};

}
#endif /* StreamCam_hpp */
