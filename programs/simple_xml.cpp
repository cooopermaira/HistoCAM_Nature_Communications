//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

#include <iostream>

using namespace pathCam;


int main(int argc, char** argv){

  using Poco::AutoPtr;
  using Poco::Util::XMLConfiguration;
  AutoPtr<XMLConfiguration> pConf(new XMLConfiguration("../../../resources/poco_test.xml"));
  std::cout <<  pConf->getString("prop1") << std::endl;
  std::cout <<   pConf->getInt("prop2") << std::endl;
  std::cout <<   pConf->getString("prop3") << std::endl; // ""
  std::cout <<   pConf->getString("prop3.prop4") << std::endl; // ""
  std::cout <<   pConf->getString("prop3.prop4[@attr]") << std::endl; // "value3"
  std::cout <<   pConf->getString("prop3.prop4[1][@attr]") << std::endl; // "value4"
  
  
  return 0;
}
