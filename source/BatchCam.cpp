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
using Poco::Util::LayeredConfiguration;
using Poco::Environment;

BatchCam::BatchCam(LayeredConfiguration::Ptr config){
  std::cout << "System OS: " << Environment::osDisplayName() << "\n";
  std::cout << "System Arch: " << Environment::osArchitecture() << "\n";
  std::cout << "Core count: " << Environment::processorCount() << "\n";

  //Defaults
  output_log = Path("./output_log.txt");
  crop_factor = 1.0;
  scale_factor  = 1.0;
  debayer = true;
  real = false;
  interpolation = INTER_CUBIC;
  feature_type = _ORB;
  use_FREAK = false;
  cv::DescriptorMatcher::MatcherType matcher_type = cv::DescriptorMatcher::BRUTEFORCE_HAMMING;
  estimator_type = RANSAC;

  if(!parseConfig(config)){
    std::cout << "Problem parsing XML.\n Exiting.\n";
    exit(-1);
  }

  mot = new pathCam::MotionEstimator();
  detector = new pathCam::FeatureDetector(feature_type, use_FREAK);
  matcher = new pathCam::DescriptorMatcher(matcher_type);

  switch(feature_type){
    case _SIFT:
      detector->set_SIFT_params(SIFT_params);
      break;
    case _SURF:
      detector->set_SURF_params(SURF_params);
      break;
    case _AKAZE:
      detector->set_AKAZE_params(AKAZE_params);
      break;
    case _BRISK:
      detector->set_BRISK_params(BRISK_params);
      break;
    case _ORB:
      detector->set_ORB_params(ORB_params);
      break;
  }


}

bool BatchCam::parseConfig(LayeredConfiguration::Ptr pConf){
    
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
            std::cout << "Unknown feature type. Using default.\n";
          }
        }else{
          std::cout << "No feature type supplied.  Using default.\n";
        }
        
        if(pConf->has("registration.detector.features.params")){
        
          switch(feature_type){
            case _SIFT:
              std::cout << "not yet implemented\n"; return false;
              break;
            case _SURF:
              std::cout << "not yet implemented\n"; return false;
              break;
            case _AKAZE:
              std::cout << "not yet implemented\n"; return false;
              break;
            case _BRISK:
              std::cout << "not yet implemented\n"; return false;
              break;
            case _ORB:
              int nfeatures;
              try{
                nfeatures = pConf->getInt("registration.detector.features.params[@nfeatures]");
              }catch(std::string bad_input){
                std::cout << "Bad input for nfeatures: " << bad_input << ". Using default.\n";
                nfeatures = ORB_params.nfeatures;
              }
              ORB_params.nfeatures = nfeatures;
              float scaleFactor;
              try{
                scaleFactor = pConf->getDouble("registration.detector.features.params[@scaleFactor]");
              }catch(std::string bad_input){
                std::cout << "Bad input for scaleFactor: " << bad_input << ". Using default.\n";
                scaleFactor = ORB_params.scaleFactor;
              }
              ORB_params.scaleFactor = scaleFactor;
              int nlevels;
              try{
                nlevels = pConf->getInt("registration.detector.features.params[@nlevels]");
              }catch(std::string bad_input){
                std::cout << "Bad input for nlevels: " << bad_input << ". Using default.\n";
                nlevels = ORB_params.nlevels;
              }
              ORB_params.nlevels = nlevels;
              int edgeThreshold;
              try{
                edgeThreshold = pConf->getInt("registration.detector.features.params[@edgeThreshold]");
              }catch(std::string bad_input){
                std::cout << "Bad input for edgeThreshold: " << bad_input << ". Using default.\n";
                edgeThreshold = ORB_params.edgeThreshold;
              }
              ORB_params.edgeThreshold = edgeThreshold;
              int firstLevel;
              try{
                firstLevel = pConf->getInt("registration.detector.features.params[@firstLevel]");
              }catch(std::string bad_input){
                std::cout << "Bad input for firstLevel: " << bad_input << ". Using default.\n";
                firstLevel = ORB_params.firstLevel;
              }
              ORB_params.firstLevel = firstLevel;
              int WTA_K;
              try{
                WTA_K = pConf->getInt("registration.detector.features.params[@WTA_K]");
              }catch(std::string bad_input){
                std::cout << "Bad input for firstLevel: " << bad_input << ". Using default.\n";
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
                std::cout << "Unknown scoreType: " << temp << ". Using Defaults.\n";
                scoreType = ORB_params.scoreType;
              }
              ORB_params.scoreType = scoreType;
              int patchSize;
              try{
                patchSize = pConf->getInt("registration.detector.features.params[@patchSize]");
              }catch(std::string bad_input){
                std::cout << "Bad input for patchSize: " << bad_input << ". Using default.\n";
                patchSize = ORB_params.patchSize;
              }
              ORB_params.patchSize = patchSize;
              int fastThreshold;
              try{
                fastThreshold = pConf->getInt("registration.detector.features.params[@fastThreshold]");
              }catch(std::string bad_input){
                std::cout << "Bad input for fastThreshold: " << bad_input << ". Using default.\n";
                fastThreshold = ORB_params.fastThreshold;
              }
              ORB_params.fastThreshold = fastThreshold;
              break;
          }
          
          
          
        }else{
          std::cout << "No feature params supplied.  Using defaults.\n";
        }
        
      }else{
        std::cout << "No feature info supplied.  Using defaults.\n";
      }
      
      if(pConf->has("registration.detector.FREAK")){
        bool temp = use_FREAK;
        try {
          use_FREAK = pConf->getBool("registration.detector.FREAK");
        }catch(std::string bad_input){
          std::cout << "Bad input for FREAK: " << bad_input << ". Using default.\n";
          use_FREAK = temp;
        }
        
      }else{
        std::cout << "No FREAK preference supplied.  Using defaults.\n";
      }
      
    }else{
      std::cout << "No detector info supplied.  Using defaults.\n";
    }
    
    if(pConf->has("registration.image")){
      
      if(pConf->has("registration.image.crop")){
        double temp = crop_factor;
        try {
          crop_factor = pConf->getDouble("registration.image.crop");
        }catch(std::string bad_input){
          std::cout << "Bad input for crop: " << bad_input << "\n";
          crop_factor = temp;
        }
        if(crop_factor < 0.0 || crop_factor > 1.0){
          std::cout << "Bad crop factor given defaulting to 1.0\n";
          crop_factor = 1.0;
        }
      }
      
      if(pConf->has("registration.image.scale")){
        double temp = scale_factor;
        try {
          scale_factor = pConf->getDouble("registration.image.scale");
        }catch(std::string bad_input){
          std::cout << "Bad input for scale: " << bad_input << "\n";
          scale_factor = temp;
        }
        if(scale_factor < 0.0 || scale_factor > 1.0){
          std::cout << "Bad scale factor given defaulting to 1.0\n";
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
          std::cout << "Improper input for interpolation. Defaulting to CUBIC\n";
          interpolation = INTER_CUBIC;
        }
        
      }
      
      if(pConf->has("registration.image.real")){
        bool temp = real;
        try {
          real = pConf->getBool("registration.image.real");
        }catch(std::string bad_input){
          std::cout << "Bad input for real: " << bad_input << ". Using default.\n";
          real = temp;
        }
      }
      
      if(pConf->has("registration.image.debayer")){
        bool temp = debayer;
        try {
          debayer = pConf->getBool("registration.image.debayer");
        }catch(std::string bad_input){
          std::cout << "Bad input for debayer: " << bad_input << ". Using default\n";
          debayer = temp;
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
  
  if(out_image.toString() != ""){
    good = compositing();
    if(!good){ std::cout << "Compositing Failed.\n"; return false; }
  }
  
  return true;
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
    
    
    detector->detect_and_compute(last_registered);
    detector->detect_and_compute(next_image);
    
    if(last_registered->keypoints.size() < 200 || next_image->keypoints.size() < 200){
      std::cout << "Too little features detected.  Going back to defaults\n";
      detector->set_ORB_params();
      detector->detect_and_compute(last_registered);
      detector->detect_and_compute(next_image);
      detector->set_ORB_params(ORB_params);
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

  return true;
}

bool BatchCam::compositing(){
  std::cout << "Compositing Images:\n";
  
  auto comp_begin = std::chrono::high_resolution_clock::now();
  
  Bbox combined_box = Bbox();
  
  for(unsigned int i=0; i < reg_results.size(); i++){
    if(reg_results[i].successful){
      if(reg_results[i].bbox.min_x <  combined_box.min_x){
        combined_box.min_x = reg_results[i].bbox.min_x;
      }
      if(reg_results[i].bbox.min_y <  combined_box.min_y){
        combined_box.min_y = reg_results[i].bbox.min_y;
      }
      if(reg_results[i].bbox.max_x >  combined_box.max_x){
        combined_box.max_x = reg_results[i].bbox.max_x;
      }
      if(reg_results[i].bbox.max_y >  combined_box.max_y){
        combined_box.max_y = reg_results[i].bbox.max_y;
      }
    }
  }
  
  
//  std::cout << "combined bbox: ";
//  std::cout << combined_box.min_x << "\t" << combined_box.min_y << "\t";
//  std::cout << combined_box.max_x << "\t" << combined_box.max_y << "\n";
  
  
  if(combined_box.min_x < 0.0){
    for(unsigned int i=0; i < reg_results.size(); i++){
      if(reg_results[i].successful){
        reg_results[i].bbox.min_x -= combined_box.min_x;
        reg_results[i].bbox.max_x -= combined_box.max_x;
      }
    }
    combined_box.max_x -= combined_box.min_x;
    combined_box.min_x -= combined_box.min_x;
  }
  
  if(combined_box.min_y < 0.0){
    for(unsigned int i=0; i < reg_results.size(); i++){
      if(reg_results[i].successful){
        reg_results[i].bbox.min_y -= combined_box.min_y;
        reg_results[i].bbox.max_y -= combined_box.max_y;
      }
    }
    combined_box.max_y -= combined_box.min_y;
    combined_box.min_y -= combined_box.min_y;
  }
  
//  std::cout << "combined bbox: ";
//  std::cout << combined_box.min_x << "\t" << combined_box.min_y << "\t";
//  std::cout << combined_box.max_x << "\t" << combined_box.max_y << "\n";
//
//  std::cout << "width height: ";
//  std::cout << "[ " << combined_box.max_x-combined_box.min_x << ", ";
//  std::cout << combined_box.max_y-combined_box.min_y << "]\n";
  
  
  Mat3b combined(combined_box.max_y-combined_box.min_y,combined_box.max_x-combined_box.min_x, Vec3b(0,0,0));
  std::cout << combined.size()  << "\n";
  
  for(unsigned int i=0; i < images.size(); i++){
    if(reg_results[i].successful){
      pathCam::Image * temp = images[i];
      temp->load_raw_from_disk();
      
      Mat image_Mat = cv::Mat(Size(temp->width,temp->height), CV_8UC1, temp->get_Raw(), Mat::AUTO_STEP);
      cvtColor(image_Mat,image_Mat,COLOR_BayerBG2BGR);
      
      image_Mat.copyTo(combined(Rect(reg_results[i].bbox.min_x, reg_results[i].bbox.min_y,                                                     image_Mat.cols, image_Mat.rows)));
      
      temp->free_memory_RAW();
    }
  }
  
  imwrite(out_image.toString(), combined);
  auto comp_end = std::chrono::high_resolution_clock::now();

  std::cout << "Done Compositing Images\n";
    
  auto comp_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - comp_begin);
  std::cout << comp_elapsed.count() * 1e-9 << " seconds including I/O\n";
  
  return true;

}

}
