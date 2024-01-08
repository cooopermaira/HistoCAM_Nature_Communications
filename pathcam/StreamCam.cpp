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



StreamCam::StreamCam(LayeredConfiguration::Ptr config): BatchCam(config), buffer_mutex(new Poco::FastMutex()),
 image_mutex(new Poco::FastMutex()),compositeQ_mutex(new Poco::FastMutex()){
  
   reg_results.resize(1,RegInfo(false, Vec2(0,0),true,0));
   reg_results[0].index = 0;
}




bool StreamCam::run(){
  
  Poco::Thread stream_thread, loader_thread, Q_thread, reg_thread, composite_thread;
  
  DiskStreamer *ds = new DiskStreamer(this);
  stream_thread.start(ds);
  
  JobQueue *jq = new JobQueue(1,1);
  
  Loader *loader = new Loader(this,jq);
  loader_thread.start(loader);
  
  QManager *qm = new QManager(this,jq);
  Q_thread.start(qm);
  
  RegManager *rm = new RegManager(this,jq);
  reg_thread.start(rm);
  
  CompositeManager *cm = new CompositeManager(this);
  composite_thread.start(cm);
  
  stream_thread.join();
  loader_thread.join();
  Q_thread.join();
  reg_thread.join();
  composite_thread.join();
  //**********************************
  /*
   This logic has been moved to MatchRunnable class
   
  for(unsigned int i=0; i < images.size(); i++){
    for(unsigned int j=0; j < images.size(); j++){
      //logger->information(Poco::format("========== (%u, %u)", i, j));
      if(matchM.match[i][j]){
        matchM.match[j][i] = new pathCam::Match(matchM.match[i][j]);
      }
    }
  }
  */
  
  /*
   This logic has been moved to Loader class
   
  reg_results.resize(images.size(), RegInfo());
  visited.resize(images.size(), false);
*/
  

  
  /*
   Images are not added to image vector if image.is_good() != true so this is no longer necessary
   
  for(start = 0; start < images.size(); start++){
    if(images[start]->is_good()){ break; }
  }
  */
  
  //if(0 == images.size()-1){ return false; }
  
  
  //reg_spanning_tree(0,Vec2(0,0));
  
  //Need to compute spanning tree for images not visited
  //then produce an image for each spanning tree
  
  /* commenting this out for brevity in cout during run
  for(unsigned int i=0; i < images.size(); i++){
    if(images[i]->is_good()){
      logger->information(Poco::format("%s\t%f\t%f\t%s", images[i]->get_ImageFile().getFileName(),  reg_results[i].vec.x, reg_results[i].vec.y, images[i]->get_label()));
    }else{
      logger->information(Poco::format("Bad:%s\t%f\t%f\t%s", images[i]->get_ImageFile().getFileName(),  reg_results[i].vec.x, reg_results[i].vec.y, images[i]->get_label()));
    }
  }
   */
  
  /*
  if(!resolve_bboxes()){ logger->error("Error resolving image bounding boxes."); }
  
  find_overlaps();
  
  
  matchM.output(images);
  
  
  
  if(out_image.toString() != ""){
    logger->information("Compositing Images.");
    
    auto comp_begin = std::chrono::high_resolution_clock::now();
    
    
    bool good;
    
    good = compositing();
    if(!good){ logger->error("Compositing failed.");  return false; }
    
    auto comp_end = std::chrono::high_resolution_clock::now();
    
    logger->information("Done Compositing Images.");
    
    auto comp_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - comp_begin);
    logger->information(Poco::format("%f seconds including I/O", comp_elapsed.count() * 1e-9));
    //auto total_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - reg_begin);
    //logger->information(Poco::format("%f total.", total_elapsed.count() * 1e-9));
  }else{
    logger->information("Not Compositing Images.");
  }
  
  */
  
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


bool StreamCam::resolve_bboxes(){
  box.resize(images.size());
  combined_box = Bbox();
  
  for(unsigned int i=0; i < images.size(); i++){
    if(!images[i]->is_good()){ continue; }
    if(i == 0){
      double t_x = 0.0;
      box[0] = Bbox(0, 0, images[0]->width, images[0]->height);
    }else{
      if(reg_results[i].successful){
        box[i] = Bbox(reg_results[i].vec.x, reg_results[i].vec.y, images[i]->width+reg_results[i].vec.x, images[i]->height+reg_results[i].vec.y);
        
      }
    }
    
    
    if(reg_results[i].successful){
      //constructor sets all these for combined_box to inf
      if(box[i].min_x <  combined_box.min_x){
        combined_box.min_x = box[i].min_x;
      }
      if(box[i].min_y <  combined_box.min_y){
        combined_box.min_y = box[i].min_y;
      }
      if(box[i].max_x >  combined_box.max_x){
        combined_box.max_x = box[i].max_x;
      }
      if(box[i].max_y >  combined_box.max_y){
        combined_box.max_y = box[i].max_y;
      }
    }
  }
  
  for(unsigned int i=0; i < images.size(); i++){
    logger->information(Poco::format("%s %s", images[i]->get_ImageFile().getFileName(), box[i].toString()));
  }
  
  if(combined_box.min_x < 0.0){
    for(unsigned int i=0; i < images.size(); i++){
      if(reg_results[i].successful){
        box[i].min_x -= combined_box.min_x;
        box[i].max_x -= combined_box.min_x;
      }
    }
    combined_box.max_x -= combined_box.min_x;
    combined_box.min_x -= combined_box.min_x; // =0
  }
  
  if(combined_box.min_y < 0.0){
    for(unsigned int i=0; i < images.size(); i++){
      if(reg_results[i].successful){
        box[i].min_y -= combined_box.min_y;
        box[i].max_y -= combined_box.min_y;
      }
    }
    combined_box.max_y -= combined_box.min_y;
    combined_box.min_y -= combined_box.min_y;
  }
  logger->information("=======================");
  logger->information(Poco::format("Image size: %s",combined_box.toString()));
  logger->information("=======================");
  
  
  for(unsigned int i=0; i < images.size(); i++){
    logger->information(Poco::format("%s %s", images[i]->get_ImageFile().getFileName(), box[i].toString()));
  }
  
  return true;
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
    temp.push_back(images[i]);
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
  composites.push_back(temp);
  int k;
}

std::vector < RegInfo > StreamCam::get_Q_front(){
  compositeQ_mutex->lock();
  std::vector < RegInfo > temp = compositeQ.front();
  compositeQ.pop();
  compositeQ_mutex->unlock();
  return temp;
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

