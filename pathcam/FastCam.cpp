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

FastCam::FastCam(LayeredConfiguration::Ptr config){
  logger = &Logger::get("PathCamLogger");
  
  logger->information(Poco::format("System OS: %s", Environment::osDisplayName()));
  logger->information(Poco::format("System Arch: %s", Environment::osArchitecture()));
  logger->information(Poco::format("System Processor Count: %u\n", Environment::processorCount()));

  //Defaults
  crop_factor = 1.0;
  scale_factor  = 1.0;
  debayer = true;
  real = false;
  interpolation = INTER_CUBIC;
  feature_type = _ORB;
  use_FREAK = false;
  matcher_type = cv::DescriptorMatcher::BRUTEFORCE_HAMMING;
  estimator_type = RANSAC;
  threads = 1;
  results_logger = NULL;

  
  if(!parseConfig(config)){
    logger->fatal("Problem parsing XML. Exiting...");
    exit(-1);
  }

  
  
  mempool.resize(threads);
   
  for(unsigned int i=0; i < threads; i++){
    mempool[i] = new MemoryPool(6464*4852);
  }
  




}

std::string printSubKeys(LayeredConfiguration::Ptr config, std::string short_key, std::string full_key, std::string print_prefix){
  std::vector<std::string> sub_keys;
  config->keys(full_key, sub_keys);
  std::string s = print_prefix + "< " + short_key + " > " + "\n";
  for(unsigned int i=0; i < sub_keys.size(); i++){
    std::string new_prefix = "\t" + print_prefix;
    s += printSubKeys(config, sub_keys[i], full_key + "." + sub_keys[i], new_prefix);
  }
  return s;
}

std::string printConfig(LayeredConfiguration::Ptr config)
{
  std::vector<std::string> root_keys;
  config->keys(root_keys);
  
  std::string s = "";
  
  for(unsigned int i=0; i < root_keys.size(); i++){
    s += printSubKeys(config, root_keys[i], root_keys[i] , "");
  }

  return s;
}

bool FastCam::parseConfig(LayeredConfiguration::Ptr pConf){
  
  logger->information(printConfig(pConf));
  
  if(pConf->has("processing")){
    if(pConf->has("processing[@threads]")){
      try{
        threads = pConf->getUInt("processing[@threads]");
        logger->information(Poco::format("Using %u threads", threads ));
      }catch(std::string bad_input){
        logger->warning("Bad input for threads: " + bad_input + ". Using default.\n");
        threads = 1;
      }
    }
  }
    
  if(pConf->has("io")){
    
    if(pConf->has("io.input_images")){
      std::string temp = pConf->getString("io.input_images");
      input_images = Path(temp);
      
      if(input_images.isDirectory()){
        logger->fatal("Input Images: Directories not supported.");
        input_images = Path();
        return false;
      }
      
      if(input_images.getExtension() != "txt"){
        logger->fatal("Input Images: Only text files supported.");
        input_images = Path();
        return false;
      }
    }else{
      logger->fatal("Input images required.");
      return false;
    }
    
    if(pConf->has("io.output_log")){
      std::string temp = pConf->getString("io.output_log");
      Path temp_log = Path(temp);
      
      if(temp_log.isDirectory()){
        logger->warning("Output Log: Directories not supported. No registration output.");
        temp_log.clear();
      }

      if(temp_log.getExtension() != "txt"){
        logger->warning("Output Log:  No registration output.");
        temp_log.clear();
      }
      
      if(temp_log.toString() != ""){
        results_logger = &Logger::get("ResultsLogger");
        AutoPtr<FileChannel> pChannel(new FileChannel);
        pChannel->setProperty("path", temp_log.toString());
        pChannel->setProperty("rotateOnOpen", "true");
        results_logger->setChannel(pChannel);
      }
      
    }else{
      logger->warning("No output log supplied. No registration output.");
    }
    
    if(pConf->has("io.output_image")){
      std::string temp = pConf->getString("io.output_image");
      out_image = Path(temp);
      
      if(out_image.getExtension() != "png" && out_image.getExtension() != "tif"){
        logger->warning("Only PNG or TIF outputs supported. No image output.");
        out_image = Path();
      }
    }
    
  }else{
    logger->fatal("No IO info supplied.");
    return false;
  }
  
  

  if(pConf->has("registration")){
    
    if(pConf->has("registration.detector")){
      
      if(pConf->has("registration.detector.features")){
        
        if(pConf->has("registration.detector.features.type")){
          std::string temp = pConf->getString("registration.detector.features.type");
          if(temp == "SIFT"){
            feature_type = _SIFT;
          }else if (temp == "SURF"){
            feature_type = _SURF;
          }else if (temp == "AKAZE"){
            feature_type = _AKAZE;
          }else if (temp == "BRISK"){
            feature_type = _BRISK;
          }else if (temp == "ORB"){
            feature_type = _ORB;
          }else{
            logger->warning("Unknown feature type. Using default.");
          }
        }else{
            logger->warning("No feature type supplied.  Using default.");
        }
        
        if(pConf->has("registration.detector.features.params")){
        
          switch(feature_type){
            case _SIFT:
              logger->warning("Params for this feature not yet implemented"); return false;
              break;
            case _SURF:
              logger->warning("Params for this feature not yet implemented"); return false;
              break;
            case _AKAZE:
              logger->warning("Params for this feature not yet implemented"); return false;
              break;
            case _BRISK:
              logger->warning("Params for this feature not yet implemented"); return false;
              break;
            case _ORB:
              int nfeatures;
              try{
                nfeatures = pConf->getInt("registration.detector.features.params[@nfeatures]");
              }catch(std::string bad_input){
                logger->warning("Bad input for nfeatures: " + bad_input + ". Using default.");
                nfeatures = ORB_params.nfeatures;
              }
              ORB_params.nfeatures = nfeatures;
              float scaleFactor;
              try{
                scaleFactor = pConf->getDouble("registration.detector.features.params[@scaleFactor]");
              }catch(std::string bad_input){
                logger->warning("Bad input for scaleFactor: " + bad_input + ". Using default.");
                scaleFactor = ORB_params.scaleFactor;
              }
              ORB_params.scaleFactor = scaleFactor;
              int nlevels;
              try{
                nlevels = pConf->getInt("registration.detector.features.params[@nlevels]");
              }catch(std::string bad_input){
                logger->warning("Bad input for nlevels: " + bad_input + ". Using default.");
                nlevels = ORB_params.nlevels;
              }
              ORB_params.nlevels = nlevels;
              int edgeThreshold;
              try{
                edgeThreshold = pConf->getInt("registration.detector.features.params[@edgeThreshold]");
              }catch(std::string bad_input){
                logger->warning("Bad input for edgeThreshold: " + bad_input + ". Using default.");
                edgeThreshold = ORB_params.edgeThreshold;
              }
              ORB_params.edgeThreshold = edgeThreshold;
              int firstLevel;
              try{
                firstLevel = pConf->getInt("registration.detector.features.params[@firstLevel]");
              }catch(std::string bad_input){
                logger->warning("Bad input for firstLevel: " + bad_input + ". Using default.");
                firstLevel = ORB_params.firstLevel;
              }
              ORB_params.firstLevel = firstLevel;
              int WTA_K;
              try{
                WTA_K = pConf->getInt("registration.detector.features.params[@WTA_K]");
              }catch(std::string bad_input){
                logger->warning("Bad input for WTA_K: " + bad_input + ". Using default.");
                WTA_K = ORB_params.WTA_K;
              }
              ORB_params.WTA_K = WTA_K;
              ORB::ScoreType scoreType;
              std::string temp = pConf->getString("registration.detector.features.params[@scoreType]");
              if(temp == "HARRIS_SCORE"){
                scoreType = ORB::HARRIS_SCORE;
              } else if (temp == "FAST_SCORE"){
                scoreType = ORB::FAST_SCORE;
              }else{
                logger->warning("Unknown scoreType: " + temp + ". Using default.");
                scoreType = ORB_params.scoreType;
              }
              ORB_params.scoreType = scoreType;
              int patchSize;
              try{
                patchSize = pConf->getInt("registration.detector.features.params[@patchSize]");
              }catch(std::string bad_input){
                logger->warning("Bad input for patchSize: " + bad_input + ". Using default.");
                patchSize = ORB_params.patchSize;
              }
              ORB_params.patchSize = patchSize;
              int fastThreshold;
              try{
                fastThreshold = pConf->getInt("registration.detector.features.params[@fastThreshold]");
              }catch(std::string bad_input){
                logger->warning("Bad input for fastThreshold: " + bad_input + ". Using default.");
                fastThreshold = ORB_params.fastThreshold;
              }
              ORB_params.fastThreshold = fastThreshold;
              break;
          }
          
          
          
        }else{
          logger->warning("No feature params supplied.  Using defaults.");
        }
        
      }else{
        logger->warning("No feature info supplied.  Using defaults.");
      }
      
      if(pConf->has("registration.detector.FREAK")){
        bool temp = use_FREAK;
        try {
          use_FREAK = pConf->getBool("registration.detector.FREAK");
        }catch(std::string bad_input){
          logger->warning("Bad input for use_FREAK: " + bad_input + ". Using default.");
          use_FREAK = temp;
        }
        
      }else{
        logger->warning("No FREAK preference supplied.  Using defaults.");
      }
      
    }else{
          logger->warning("No detector info supplied.  Using defaults.");
    }
    
    if(pConf->has("registration.image")){
      
      if(pConf->has("registration.image.crop")){
        double temp = crop_factor;
        try {
          crop_factor = pConf->getDouble("registration.image.crop");
        }catch(std::string bad_input){
          logger->warning("Bad input for crop: " + bad_input + ".");
          crop_factor = temp;
        }
        if(crop_factor < 0.0 || crop_factor > 1.0){
          logger->warning("Bad crop factor given defaulting to 1.0");
          crop_factor = 1.0;
        }
      }
      
      if(pConf->has("registration.image.scale")){
        double temp = scale_factor;
        try {
          scale_factor = pConf->getDouble("registration.image.scale");
        }catch(std::string bad_input){
          logger->warning("Bad input for scale: " + bad_input + ".");
          scale_factor = temp;
        }
        if(scale_factor < 0.0 || scale_factor > 1.0){
          logger->warning("Bad scale factor given defaulting to 1.0");
          scale_factor = 1.0;
        }
    
      }
      
      if(pConf->has("registration.image.interpolation")){
        std::string temp = pConf->getString("registration.image.interpolation");
        
        if(temp == "NEAREST"){
          interpolation = INTER_NEAREST;
        }else if (temp == "LINEAR"){
          interpolation = INTER_LINEAR;
        }else if (temp == "CUBIC"){
          interpolation = INTER_CUBIC;
        }else if (temp == "AREA"){
          interpolation = INTER_AREA;
        }else if (temp == "LANCZOS4"){
          interpolation = INTER_LANCZOS4;
        }else if (temp == "LINEAR_EXACT"){
          interpolation = INTER_LINEAR_EXACT;
        }else if (temp == "NEAREST_EXACT"){
          interpolation = INTER_NEAREST_EXACT;
        }else if (temp == "MAX"){
          interpolation = INTER_MAX;
        }else{
          logger->warning("Improper input for interpolation. Defaulting to CUBIC");
          interpolation = INTER_CUBIC;
        }
        
      }
      
      if(pConf->has("registration.image.real")){
        bool temp = real;
        try {
          real = pConf->getBool("registration.image.real");
        }catch(std::string bad_input){
          logger->warning("Bad input for real: " + bad_input + ". Using default.");
          real = temp;
        }
      }
      
      if(pConf->has("registration.image.debayer")){
        bool temp = debayer;
        try {
          debayer = pConf->getBool("registration.image.debayer");
        }catch(std::string bad_input){
          logger->warning("Bad input for debayer: " + bad_input + ". Using default.");
          debayer = temp;
        }
      }
      
    }else{
      logger->warning("No registation image info supplied.  Using defaults.");
    }
    
    if(pConf->has("registration.matcher")){
      
    }else{
      logger->warning("No matcher info supplied.  Using defaults.");
    }
    
    if(pConf->has("registration.estimator")){
      
    }else{
      logger->warning("No estimator info supplied.  Using defaults.");
    }
    
  }else{
      logger->warning("No registration info supplied.  Using defaults.");
  }

  

  return true;

  
  
}

bool FastCam::loadFileList(){
  
  std::ifstream infile(input_images.toString().c_str());
  
  if(!infile.good()){
    logger->fatal("Unable to open file.");
    return false;
  }
  
  std::string imageFile;
  
  while (infile >>imageFile){
    if(imageFile.size() == 0){ continue;}
    pathCam::Image *image = new pathCam::Image();
    image->set_disk_file(imageFile);
    images.push_back(image);
  }
  
  if(images.size() == 0){
    logger->fatal("No images loaded.");
    return false;
  }
  
  logger->information("%u images loaded.", (unsigned int)images.size());
  infile.close();
  
  return true;
}


class FeatureQueue{
private:
  Poco::ThreadPool *pool;
  std::queue < Poco::Runnable * > jobQueue;

public:
  FeatureQueue(int min_threads, int max_threads){
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
    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);

    
    
  }

    
};

class MatchRunnable: public Poco::Runnable{
private:
  FastCam *parent;
  unsigned int image_idx;
  
public:
  
  bool successful;

  
  MatchRunnable(FastCam *parent): parent(parent), successful(false){};
  
  virtual void run(){
    
  }

    
};


bool FastCam::run(){
  bool good;
  
  good = loadFileList();
  if(!good){ logger->fatal("Exiting run."); return false; }

  logger->information(Poco::format("Performing Registration w/ %u threads\n", threads));
  auto reg_begin = std::chrono::high_resolution_clock::now();

//  std::vector < PairRegRunnable > runnable(threads, PairRegRunnable(this));
//  std::vector < Poco::Thread > thread(threads);
//
//  reg_results.resize(images.size());
//  matchM.resize(images.size());
//
//  for(unsigned int i=0; i < threads; i++){
//    thread[i].start(runnable[i]);
//  }
//
//  for(unsigned int i=0; i < threads; i++){
//    thread[i].join();
//  }
//
//  for(unsigned int i=0; i < threads; i++){
//    if(!runnable[i].successful){ logger->error("Registration failed. %d", i); return false; }
//  }
//
//
//  auto reg_end = std::chrono::high_resolution_clock::now();
//  auto reg_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(reg_end - reg_begin);
//  logger->information("Done Registering Images.");
//  logger->information(Poco::format("%f seconds including I/O", reg_elapsed.count() * 1e-9));
//
//  if(!resolve_bboxes()){ logger->error("Error resolving image bounding boxes."); }
//
//  find_overlaps();
//
//  matchM.output(images);
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
