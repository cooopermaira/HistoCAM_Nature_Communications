//
//  BatchCam.cpp
//  pathCamLib
//
//  Created by Brian on 3/15/23.
//

#include "pathCam.h"

namespace pathCam{

using Poco::AutoPtr;
using Poco::Util::XMLConfiguration;

BatchCam::BatchCam(Poco::Path xml_config){
  std::cout << xml_config.toString();
  
  parseXML(xml_config);
    
}

void BatchCam::parseXML(Poco::Path xml_config){
  
  AutoPtr<XMLConfiguration> pConf(new XMLConfiguration(xml_config.toString()));
  
  
  
  
  std::string prop1 = pConf->getString("prop1");
  std::cout << prop1 << "\n";
  int prop2 = pConf->getInt("prop2");
  std::string prop3 = pConf->getString("prop3"); // ""
  std::string prop4 = pConf->getString("prop3.prop4"); // ""
  prop4 = pConf->getString("prop3.prop4[@attr]"); // "value3"
  prop4 = pConf->getString("prop3.prop4[1][@attr]"); // "value4"
}

}
