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
      
      if(pConf->has("registration.detector.features")){
        
        if(pConf->has("registration.detector.features.type")){
        
          
          
        }
        
        if(pConf->has("registration.detector.features.params")){
        
          
          
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

}
