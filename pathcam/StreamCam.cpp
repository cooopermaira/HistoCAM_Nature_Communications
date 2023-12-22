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
 image_mutex(new Poco::FastMutex()),
 offset_to_root(0.0,0.0),Bbox_max(0.0,0.0){
  
  reg_results.resize(1,RegInfo(false, Vec2(0,0)));
  
}



class RegManager: public Poco::Runnable{
private:
  pathCam::StreamCam *parent;
  pathCam::JobQueue *queue;
  unsigned int current_index = 0;
  
  
public:
  bool successful;
  
  RegManager(pathCam::StreamCam *parent, pathCam::JobQueue *queue): parent(parent), queue(queue), successful(false){}
  
  virtual void run(){
    
    bool check_done = false;
    
    while(!(parent->disk_empty and parent->jobs_queued and queue->is_empty() and parent->ConseqQ.is_empty())){
      
      if(parent->ConseqQ.get_run() > 10 or check_done){
        
        check_done = false;
        std::vector < long int > indexes;
        indexes = parent->ConseqQ.return_run(parent->ConseqQ.get_run());
        
        for(int i = 0; i < indexes.size(); i++){

          if(parent->reg_results[indexes[i]].root){continue;} //create new component
          
          Vec2 accum = Vec2(0.0,0.0);
          for(long int j = indexes[i] - 1; j >= 0; j--){
            
            if (parent->matchM.match[j][indexes[i]]){
              
              accum.x = parent->reg_results[j].vec.x + parent->matchM.match[indexes[i]][j]->t_x;
              
              accum.y = parent->reg_results[j].vec.y + parent->matchM.match[indexes[i]][j]->t_y;
              
              parent->offset_to_root.x = min(parent->offset_to_root.x,accum.x);
              parent->offset_to_root.y = min(parent->offset_to_root.y,accum.y);
              
              parent->reg_results[indexes[i]] = RegInfo(true, accum);
              break;
            }
          }
          
        }
      }else if(parent->ConseqQ.get_run() == 0){
        Poco::Thread::sleep(100);
      }else{
        check_done = true;
        Poco::Thread::sleep(100);
      }//end if
    }//end while
  }//end run()
};//end class


class Loader: public Poco::Runnable{
private:
  StreamCam *parent;
  JobQueue *queue;
  unsigned long int image_index = 0;
  
public:
  
  bool successful;
  
  Loader(StreamCam *parent, JobQueue *queue): parent(parent), queue(queue), successful(false){};
  
  virtual void run(){
    
    while(!parent->disk_empty){
      if(parent->buffer.empty()){
        Poco::Thread::sleep(100);
      }else{
        pathCam::Image *image = new pathCam::Image;
        
        parent->buffer_mutex->lock();
        image->set_disk_file(parent->disk_image.front());
        parent->disk_image.pop();
        
        image->copy_in(parent->buffer.front());
        delete [] parent->buffer.front();
        parent->buffer.pop();
        parent->buffer_mutex->unlock();
        
        if(!image->in_memory()){
          successful = false;
          image->label = Image::_BAD_FILE;
          return;
        }
        
        image->find_label();
        
        if(image->is_good()){
          image->create_reg_image(parent->scale_factor,parent->crop_factor,parent->debayer,parent->interpolation, parent->real);
          
          pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(parent->feature_type, parent->use_FREAK);
          
          switch(parent->feature_type){
            case _SIFT:
              detector->set_SIFT_params(parent->SIFT_params);
              break;
            case _SURF:
              detector->set_SURF_params(parent->SURF_params);
              break;
            case _AKAZE:
              detector->set_AKAZE_params(parent->AKAZE_params);
              break;
            case _BRISK:
              detector->set_BRISK_params(parent->BRISK_params);
              break;
            case _ORB:
              detector->set_ORB_params(parent->ORB_params);
              break;
          }
          
          detector->detect_and_compute(image);
          
          if(image->keypoints.size() < 200){
            detector->set_ORB_params();
            detector->detect_and_compute(image);
          }
          
          delete detector;
          
          if(image->keypoints.size() < 200){
            successful = false;
            image->label = Image::_LOWFEAT;
            return;
          }
          
          parent->add_image(image);
          
          if(image_index % 100 == 0){
            parent->matchM.resize(image_index + 100);
            parent->reg_results.resize(image_index + 100, RegInfo());
            parent->visited.resize(image_index + 100, false);
          }
          
          MatchRunnable *matchjob = new MatchRunnable(parent,image_index);
          queue->add_runnable(matchjob);
          
          image_index++;
          successful = true;
          
        }
        
        image->free_memory_RAW();
        
      }//end if
      
    }//end while
    parent->jobs_queued = true;
  }//end run
  
};//end class

bool StreamCam::run(){
  
  Poco::Thread stream_thread, loader_thread, Q_thread, reg_thread;
  
  DiskStreamer *ds = new DiskStreamer(this);
  stream_thread.start(ds);
  
  JobQueue *jq = new JobQueue(10,10);
  
  Loader *loader = new Loader(this,jq);
  loader_thread.start(loader);
  
  QManager *qm = new QManager(this,jq);
  Q_thread.start(qm);
  
  RegManager *rm = new RegManager(this,jq);
  reg_thread.start(rm);
  
  stream_thread.join();
  loader_thread.join();
  Q_thread.join();
  reg_thread.join();
  
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
  
  if(0 == images.size()-1){ return false; }
  
  
  //reg_spanning_tree(0,Vec2(0,0));
  
  //Need to compute spanning tree for images not visited
  //then produce an image for each spanning tree
  
  
  for(unsigned int i=0; i < images.size(); i++){
    if(images[i]->is_good()){
      logger->information(Poco::format("%s\t%f\t%f\t%s", images[i]->get_ImageFile().getFileName(),  reg_results[i].vec.x, reg_results[i].vec.y, images[i]->get_label()));
    }else{
      logger->information(Poco::format("Bad:%s\t%f\t%f\t%s", images[i]->get_ImageFile().getFileName(),  reg_results[i].vec.x, reg_results[i].vec.y, images[i]->get_label()));
    }
  }
  
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

Image* StreamCam::get_image_ref(unsigned long int index){
  Image * temp;
  image_mutex->lock();
  temp = images[index];
  image_mutex->unlock();
  return temp;
}

}

