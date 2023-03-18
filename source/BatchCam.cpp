//
//  BatchCam.cpp
//  pathCamLib
//
//  Created by Brian on 3/15/23.
//

#include "pathCam.h"

namespace pathCam{

using Poco::AutoPtr;
using Poco::Path;
using Poco::Util::XMLConfiguration;
using Poco::Environment;

BatchCam::BatchCam(Poco::Path xml_config){
  std::cout << "System OS: " << Environment::osDisplayName() << "\n";
  std::cout << "System Arch: " << Environment::osArchitecture() << "\n";
  std::cout << "Core count: " << Environment::processorCount() << "\n";

  std::cout << "Parsing " << xml_config.toString() << "\n";
  
  //Defaults
  output_log = Path("./output_log.txt");
  crop_factor = 1.0;
  double scale_factor  = 1.0;
  bool debayer = true;
  bool real = false;
  int interpolation = INTER_CUBIC;
  int feature_type = _ORB;
  bool use_FREAK = false;
  pathCam::FeatureDetector::ORBParameters ORB_params;
  cv::DescriptorMatcher::MatcherType matcher_type = cv::DescriptorMatcher::BRUTEFORCE_HAMMING;
  int estimator_type = RANSAC;
  
//  pathCam::MotionEstimator *mot =;
//  pathCam::FeatureDetector *detector;
//  pathCam::DescriptorMatcher *matcher;


  
  if(!parseXML(xml_config)){
    std::cout << "Problem parsing XML.\n Exiting.\n";
    exit(-1);
  }
    
}

bool BatchCam::parseXML(Poco::Path xml_config){
  
  AutoPtr<XMLConfiguration> pConf(new XMLConfiguration(xml_config.toString()));
  
  if(pConf->has("io")){
    
    if(pConf->has("io.input_images")){
      std::string temp = pConf->getString("io.input_images");
      input_images = Path(temp);
      
      if(input_images.isDirectory()){
        std::cout << "Input Images: Directories not supported.\n";
        input_images = Path();
        return false;
      }
      
      if(input_images.getExtension() != "txt"){
        std::cout << "Input Images: Only text files supported.\n";
        input_images = Path();
        return false;
      }
    }else{
      std::cout << "Input images required.\n";
      return false;
    }
    
    if(pConf->has("io.output_log")){
      std::string temp = pConf->getString("io.output_log");
      Path temp_log = Path(temp);
      
      if(temp_log.isDirectory()){
        std::cout << "Output Log: Directories not supported.\n";
        std::cout << "Using default.\n";
        std::cout << output_log.toString() << "\n";
        temp_log.clear();
      }

      if(temp_log.getExtension() != "txt"){
        std::cout << "Output Log: Only text files supported.\n";
        std::cout << "Using default.\n";
        std::cout << output_log.toString() << "\n";
        temp_log.clear();
      }
      
      if(temp_log.toString() != ""){ output_log = temp_log; }
      
    }else{
      std::cout << "No output log supplied.\n";
      std::cout << "Using default.\n";
      std::cout << output_log.toString() << "\n";
    }
    
    if(pConf->has("io.output_image")){
      std::string temp = pConf->getString("io.output_image");
      out_image = Path(temp);
      
      if(out_image.getExtension() != "png" && out_image.getExtension() != "tif"){
        std::cout << "Only PNG or TIF outputs supported. No image output.\n";
        out_image = Path();
      }
    }
    
  }else{
    std::cout << "No IO info supplied.\n";
    return false;
  }
  
  

  if(pConf->has("registration")){
    
    if(pConf->has("registration.detector")){
      
      if(pConf->has("registration.detector.features")){
        
        if(pConf->has("registration.detector.features.type")){
        
          
          
        }else{
          std::cout << "No feature type supplied.  Using defaults.\n";
        }
        
        if(pConf->has("registration.detector.features.params")){
        
          
          
        }else{
          std::cout << "No feature params supplied.  Using defaults.\n";
        }
        
        
      }else{
        std::cout << "No feature info supplied.  Using defaults.\n";
      }
      
      if(pConf->has("registration.detector.FREAK")){
        
        
      }else{
        std::cout << "No FREAK preference supplied.  Using defaults.\n";
      }
      
    }else{
      std::cout << "No detector info supplied.  Using defaults.\n";
    }
    
    if(pConf->has("registration.image")){
      
      if(pConf->has("registration.image.crop")){
        try {
          crop_factor = pConf->getDouble("registration.image.crop");
        }catch(std::string bad_input){
          std::cout << "Bad input for crop: " << bad_input << "\n";
        }
        if(crop_factor < 0.0 || crop_factor > 1.0){
          std::cout << "Bad crop factor given defaulting to 1.0\n";
          crop_factor = 1.0;
        }
      }
      
      if(pConf->has("registration.image.scale")){
        try {
          scale_factor = pConf->getDouble("registration.image.scale");
        }catch(std::string bad_input){
          std::cout << "Bad input for scale: " << bad_input << "\n";
        }
        if(scale_factor < 0.0 || scale_factor > 1.0){
          std::cout << "Bad scale factor given defaulting to 1.0\n";
          crop_factor = 1.0;
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
          std::cout << "Improper input for interpolation. Defaulting to CUBIC\n";
          interpolation = INTER_CUBIC;
        }
        
      }
      
      if(pConf->has("registration.image.real")){
        try {
          real = pConf->getBool("registration.image.real");
        }catch(std::string bad_input){
          std::cout << "Bad input for real: " << bad_input << "\n";
        }
      }
      
      if(pConf->has("registration.image.debayer")){
        try {
          debayer = pConf->getBool("registration.image.debayer");
        }catch(std::string bad_input){
          std::cout << "Bad input for debayer: " << bad_input << "\n";
        }
      }
      
    }else{
      std::cout << "No registation image info supplied.  Using defaults.\n";
    }
    
    if(pConf->has("registration.matcher")){
      
    }else{
      std::cout << "No matcher info supplied.  Using defaults.\n";
    }
    
    if(pConf->has("registration.estimator")){
      
    }else{
      std::cout << "No estimator info supplied.  Using defaults.\n";
    }
    
  }else{
    std::cout << "No registration info supplied.  Using defaults.\n";
  }

  

  return true;

  
  
//  std::string prop1 = pConf->getString("prop1");
//  std::cout << prop1 << "\n";
//  int prop2 = pConf->getInt("prop2");
//  std::cout << prop2 << "\n";
//  std::string prop3 = pConf->getString("prop3"); // ""
//  std::string prop4 = pConf->getString("prop3.prop4"); // ""
//  prop4 = pConf->getString("prop3.prop4[@attr]"); // "value3"
//  prop4 = pConf->getString("prop3.prop4[1][@attr]"); // "value4"
}

bool BatchCam::loadFileList(){
  
  std::ifstream infile(input_images.toString().c_str());
  
  if(!infile.good()){
    std::cout << "Unable to open file\n";
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
    std::cout << "No images loaded.\n";
    return false;
  }
  
  std::cout << images.size() << " images loaded.\n";
  infile.close();
  
  return true;
}


bool BatchCam::run(){
  bool good;
  
  good = registration();
  if(!good){ std::cout << "Registration Failed.\n"; return false; }
  
  
}

bool BatchCam::registration(){
  bool good = loadFileList();
  if(!good){ std::cout << "Exiting run.\n"; return false; }
  
  std::ofstream outfile;
  outfile.open(output_log.toString());
  
  std::cout << "Performing Registration:\n";

  auto reg_begin = std::chrono::high_resolution_clock::now();
    
  reg_results.resize(images.size());
  unsigned int last_index = 0;
  
  for(unsigned int i=0; i < images.size()-1; i++){
    
    if(i==0){
        Bbox box = Bbox(0, 0, images[0]->width, images[0]->height);
        reg_results[0] = RegInfo(true, Vec2(0, 0), box);
    }
      
    pathCam::Image * last_registered = images[last_index];
    pathCam::Image * next_image = images[i+1];

    last_registered->load_raw_from_disk();
    next_image->load_raw_from_disk();
    
    if(!last_registered->in_memory() || !next_image->in_memory()){
      std::cout << "Issue loading image.\n";
      continue;
    }
    
    outfile << last_registered->get_File() << "\t";
    outfile << next_image->get_File() << "\t";
    
    auto begin = std::chrono::high_resolution_clock::now();
    
    last_registered->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);
    next_image->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);
    
    detector->set_ORB_params(ORB_params);
    
    detector->detect_and_compute(last_registered);
    detector->detect_and_compute(next_image);
    
    if(last_registered->keypoints.size() < 200 || next_image->keypoints.size() < 200){
      std::cout << "Too little features detected.  Going back to defaults\n";
      delete detector;
      detector = new pathCam::FeatureDetector(feature_type, use_FREAK);
      detector->detect_and_compute(last_registered);
      detector->detect_and_compute(next_image);
      delete detector;
      detector = new pathCam::FeatureDetector(feature_type, use_FREAK);
    }
    
    if(last_registered->keypoints.size() < 100 || next_image->keypoints.size() < 100){
      outfile << "failed. Not enough keypoints\n";
      reg_results[i+1] = RegInfo(false, reg_results[i].vec);
      continue;
    }
    
    pathCam::Match *m = new pathCam::Match(last_registered,next_image);
    matcher->match(m);
    
    int result = mot->findHomography(m, estimator_type);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    outfile << last_registered->keypoints.size() << "\t";
    outfile << next_image->keypoints.size() << "\t";
    
    
    if(result == 1){
      outfile << m->t_x << "\t" << m->t_y << "\t";
      outfile << elapsed.count() * 1e-9 << "\n";
      double t_x = reg_results[last_index].vec.x-m->t_x;
      double t_y = reg_results[last_index].vec.y-m->t_y;
      Bbox box = Bbox(t_x, t_y, next_image->width+t_x, next_image->height+t_y);
      reg_results[i+1] = RegInfo(true, Vec2(t_x, t_y), box);
      last_index = i+1;
    }
    if(result == -1){
      outfile << "failed. Not enough matches\n";
      reg_results[i+1] = RegInfo(false, reg_results[i].vec);
    }
    if(result == -2){
      outfile << "failed. Translation not found.\n";
      reg_results[i+1] = RegInfo(false, reg_results[i].vec);
    }
    
    //Need to only unload if not using again, but doing this to make sure
    //initial program has no memory leaks
    
    last_registered->free_memory_RAW();
    next_image->free_memory_RAW();
      
    delete m;
  }

    
  auto reg_end = std::chrono::high_resolution_clock::now();
  auto reg_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(reg_end - reg_begin);
  std::cout << "Done Registering Images\n";
  std::cout << reg_elapsed.count() * 1e-9 << " seconds including I/O\n";


}

}
