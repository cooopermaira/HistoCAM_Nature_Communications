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

BatchCam::BatchCam(Poco::Path xml_config){
  std::cout << "Parsing " << xml_config.toString() << "\n";
  
  parseXML(xml_config);
    
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
      output_log = Path(temp);
      
      if(output_log.isDirectory()){
        std::cout << "Output Log: Directories not supported.\n";
        output_log = Path();
        return false;
      }

      if(output_log.getExtension() != "txt"){
        std::cout << "Output Log: Only text files supported.\n";
        output_log = Path();
        return false;
      }
      
    }else{
      std::cout << "Output Log required.\n";
      return false;
    }
    
    if(pConf->has("io.output_image")){
      std::string temp = pConf->getString("io.output_image");
      out_image = Path(temp);
      
      if(out_image.getExtension() != "png" && out_image.getExtension() != "tif"){
        std::cout << "Only PNG or TIF outputs supported.\n";
        out_image = Path();
        return false;
      }
    }
    
  }else{
    std::cout << "No IO info supplied.\n";
    return false;
  }
  
  

  if(pConf->has("registration")){
    
    if(pConf->has("registration.detector")){
      
      if(pConf->has("registration.detector.feature_type")){
        
        if(pConf->has("registration.detector.feature_type.type")){
        
          
          
        }
        
        if(pConf->has("registration.detector.feature_type.params")){
        
          
          
        }
        
        
      }
      
      if(pConf->has("registration.detector.FREAK")){
        
        
      }
      
    }
    
    if(pConf->has("registration.image")){
      
      if(pConf->has("registration.image.crop")){
       
       
        
      }
      
      if(pConf->has("registration.image.scale")){
       
      
        
      }
      
      if(pConf->has("registration.image.interpolation")){
       
      
        
      }
      
      if(pConf->has("registration.image.real")){
       
      
        
      }
      
      if(pConf->has("registration.image.debayer")){
       
      
        
      }
      
    }
    
    if(pConf->has("registration.matcher")){
      
      
      
    }
    
    if(pConf->has("registration.estimator")){
      
      
      
    }
    
    return true;
  }

  


  
  
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
