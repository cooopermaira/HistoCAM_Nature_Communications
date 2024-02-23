//
//  StreamCam.cpp
//  pathCamLib
//
//  Created by cooper maira on 11/24/23.
//

#include "pathCam.h"

namespace pathCam{

using Poco::AutoPtr;
using Poco::Path;
using Poco::Util::XMLConfiguration;
using Poco::Util::LayeredConfiguration;
using Poco::Logger;
using Poco::LogStream;
using Poco::Environment;
using Poco::FileChannel;



StreamCam::StreamCam(LayeredConfiguration::Ptr config): BatchCam(config), buffer_mutex(new Poco::FastMutex()),image_mutex(new Poco::FastMutex()),compositeQ_mutex(new Poco::FastMutex()),component_mutex(new Poco::FastMutex()),spin_buffer_mutex(new Poco::FastMutex()){
  
   reg_results.resize(1,RegInfo(false, Vec2(0,0),true,0));
   reg_results[0].index = 0;
}


bool StreamCam::spin_run() {
    Poco::Thread loader_thread, Q_thread, reg_thread, composite_thread;

    std::cout << "spin_run started " << std::endl;

    JobQueue* jq = new JobQueue(3, 3);
    SpinLoader* sl = new SpinLoader(this,jq);
    loader_thread.start(sl);

    QManager* qm = new QManager(this, jq);
    Q_thread.start(qm);

    RegManager* rm = new RegManager(this, jq);
    reg_thread.start(rm);

    CompositeManager* cm = new CompositeManager(this);
    cm->run();
    //composite_thread.start(cm);
    //UI can only be altered from the main thread, this will change when I'm not demoing

    loader_thread.join();
    Q_thread.join();
    reg_thread.join();
    composite_thread.join();

    std::cout << "spin_run done" << std::endl;

    return true;
}

bool StreamCam::run(){
  //cv::namedWindow("display");
  //cv::namedWindow("display2");
  Poco::Thread stream_thread, loader_thread, Q_thread, reg_thread, composite_thread;
  
  DiskStreamer *ds = new DiskStreamer(this);

  /*
  eliminate disk streamer
  create new function on streamcam object class to accept an image pointer
  push image pointer to loader class
  loader extracts features as normal, releases memory

  */

  stream_thread.start(ds);
  
  JobQueue *jq = new JobQueue(10,10);
  
  Loader *loader = new Loader(this,jq);
  loader_thread.start(loader);
  
  QManager *qm = new QManager(this,jq);
  Q_thread.start(qm);
  
  RegManager *rm = new RegManager(this,jq);
  reg_thread.start(rm);
  
  CompositeManager *cm = new CompositeManager(this);
  cm->run();
  //composite_thread.start(cm);
  //UI can only be altered from the main thread, this will change when I'm not demoing
  
  stream_thread.join();
  loader_thread.join();
  Q_thread.join();
  reg_thread.join();
  composite_thread.join();

  return true;
}

void StreamCam::reg_spanning_tree(unsigned int root_idx, Vec2 offset){
  visited[root_idx] = true;
  for(unsigned int j=0; j < images.size(); j++){
    //if(!images[j]->is_good()){ continue; }
    if(matchM.match[root_idx][j]){
      if(!visited[j]){
        Vec2 accum_offset = Vec2(matchM.match[j][root_idx]->t_x+offset.x, matchM.match[j][root_idx]->t_y+offset.y);
        reg_results[j] = RegInfo(true, accum_offset);
        reg_spanning_tree(j, accum_offset);
      }
    }
  }
}


unsigned long int StreamCam::add_image(Image * image){
  unsigned long int index;
  image_mutex->lock();
  images.push_back(image);
  index = images.size() - 1;
  image_mutex->unlock();
  return index;
}

std::vector<Image*> StreamCam::get_image_refs(std::vector<unsigned long int> indexes){
  
  std::vector<Image*> temp;
  
  image_mutex->lock();
  for(unsigned int i = 0; i <indexes.size(); i++){
    temp.push_back(images[indexes[i]]);
  }
  image_mutex->unlock();
  
  return temp;
}

Image* StreamCam::get_image_ref(unsigned long int index){
  Image * temp;
  image_mutex->lock();
  temp = images[index];
  image_mutex->unlock();
  return temp;
}

void StreamCam::add_new_component(unsigned long image_index){
  reg_results[image_index] = RegInfo(false,Vec2(0.0,0.0),true,increment_and_get_components());
  reg_results[image_index].index = image_index;
  //delete these pointers when destroyed
  Composite * temp = new Composite(this);
  component_mutex->lock();
  composites.push_back(temp);
  component_mutex->unlock();
  int k = 0;
}

std::vector < RegInfo > StreamCam::get_Q_front(){
  compositeQ_mutex->lock();
  std::vector < RegInfo > temp = compositeQ.front();
  compositeQ.pop();
  compositeQ_mutex->unlock();
  return temp;
}

Image* StreamCam::get_Q_front_Spin() {
    spin_buffer_mutex->lock();
    Image* temp = spin_image_buffer.front();
    spin_image_buffer.pop();
    spin_buffer_mutex->unlock();
    return temp;
}

void StreamCam::pass_image(Image* image){
    spin_buffer_mutex->lock();
    spin_image_buffer.push(image);
    spin_buffer_mutex->unlock();
}

void StreamCam::push_compositeQ(std::vector<RegInfo> indexes){
  compositeQ_mutex->lock();
  compositeQ.push(indexes);
  compositeQ_mutex->unlock();
}

bool StreamCam::compositeQ_empty(){
  bool temp;
  compositeQ_mutex->lock();
  temp = compositeQ.empty();
  compositeQ_mutex->unlock();
  return temp;
}
}

