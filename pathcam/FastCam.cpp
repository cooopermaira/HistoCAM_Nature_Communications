//
//  FastCam.cpp
//  pathCamLib
//
//  Created by Brian on 3/15/23.
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

FastCam::FastCam(LayeredConfiguration::Ptr config): BatchCam(config){ }

class JobQueue{
private:
  Poco::ThreadPool *pool;
  std::queue < Poco::Runnable * > jobQueue;

public:
  JobQueue(int min_threads, int max_threads){
    pool = new Poco::ThreadPool(min_threads,max_threads,60,POCO_THREAD_STACK_SIZE);
  }
  
  bool run_jobs(std::vector < Poco::Runnable * > jobs){
    for(unsigned int i=0; i < jobs.size(); i++){
      jobQueue.push(jobs[i]);
    }
      
    while(!jobQueue.empty()){
      if(pool->available() > 0){
        pool->start(*jobQueue.front());
        jobQueue.pop();
      }else{
        Poco::Thread::sleep(100);
      }
    }
    
    pool->joinAll();
  }
  
};

class FeaturesRunnable: public Poco::Runnable{
private:
  FastCam *parent;
  pathCam::Image * image;
  
public:
  
  bool successful;
  
  FeaturesRunnable(FastCam *parent,  pathCam::Image * image): parent(parent), image(image), successful(false){};
  
  virtual void run(){

    image->load_raw_from_disk();
    
    if(!image->in_memory()){
      successful = false;
      image->label = Image::_LOWFEAT;
      return;
    }

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
    
    if(image->keypoints.size() < 200){
      successful = false;
      image->label = Image::_LOWFEAT;
      return;
    }
    
    successful = true;
    image->label = Image::_UNKOWN;
    
  }

    
};

class MatchRunnable: public Poco::Runnable{
private:
  FastCam *parent;
  unsigned int image_idx;
  
public:
  
  bool successful;

  
  MatchRunnable(FastCam *parent, unsigned int image_idx): parent(parent), image_idx(image_idx), successful(false){};
  
  virtual void run(){
    pathCam::Image * image = parent->images[image_idx];
    
    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);
    pathCam::MotionEstimator *mot = new pathCam::MotionEstimator();
    
    for(long int prev_idx=image_idx-1; prev_idx >=0; prev_idx--){
      pathCam::Image *previous = parent->images[prev_idx];
      parent->matchM.match[prev_idx][image_idx] = new pathCam::Match(previous,image);
      pathCam::Match *m = parent->matchM.match[prev_idx][image_idx];
      matcher->match(m);
      int result = mot->findHomography(m, parent->estimator_type);
      if(result == 1){
        successful = true;
        return;
      }
      if(result == -1){
        delete m;
        parent->matchM.match[image_idx-1][image_idx] = NULL;
      }
      if(result == -2){
        delete m;
        parent->matchM.match[image_idx-1][image_idx] = NULL;
      }
    }
    
    successful = false;
    
    delete matcher;
    delete mot;

  }

    
};


bool FastCam::run(){
  bool good;
  
  good = loadFileList();
  if(!good){ logger->fatal("Exiting run."); return false; }

  logger->information(Poco::format("Performing FastCam Registration w/ %u threads\n", threads));
  auto reg_begin = std::chrono::high_resolution_clock::now();
  
  matchM.resize(images.size());
  
  JobQueue queue = JobQueue(threads, threads);
  
  std::vector < Poco::Runnable * > feat_runnable(images.size());
  
  for(unsigned int i=0; i < images.size(); i++){
    feat_runnable[i] = new FeaturesRunnable(this, images[i]);
  }
  
   
  queue.run_jobs(feat_runnable);
  
  std::vector < Poco::Runnable * > match_runnable(images.size()-1);

  for(unsigned int i=1; i < images.size(); i++){
    match_runnable[i] = new MatchRunnable(this, i);
  }

  queue.run_jobs(match_runnable);

  auto reg_end = std::chrono::high_resolution_clock::now();
  auto reg_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(reg_end - reg_begin);
  logger->information("Done Registering Images.");
  logger->information(Poco::format("%f seconds including I/O", reg_elapsed.count() * 1e-9));

  for(unsigned int i=0; i < images.size(); i++){
    for(unsigned int j=0; j < images.size(); j++){
      if(matchM.match[i][j]){
        matchM.match[j][i] = new pathCam::Match(matchM.match[i][j]);
      }
    }
  }
  
  logger->fatal("Done.");


//  if(!resolve_bboxes()){ logger->error("Error resolving image bounding boxes."); }
//
//  find_overlaps();
//
//  matchM.output(images);
//
//  reg_results.resize(images.size());
//
//  for(unsigned int i=0; i < images.size(); i++){
//    //Need to figure out the reg_results
//  }
//
//
//  if(out_image.toString() != ""){
//    logger->information("Compositing Images.");
//
//    auto comp_begin = std::chrono::high_resolution_clock::now();
//
//    good = compositing();
//    if(!good){ logger->error("Compositing failed.");  return false; }
//
//    auto comp_end = std::chrono::high_resolution_clock::now();
//
//    logger->information("Done Compositing Images.");
//
//    auto comp_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - comp_begin);
//    logger->information(Poco::format("%f seconds including I/O", comp_elapsed.count() * 1e-9));
//    auto total_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - reg_begin);
//    logger->information(Poco::format("%f total.", total_elapsed.count() * 1e-9));
//  }
  
  return true;
}


}
