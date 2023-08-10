//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

#include <iostream>

using namespace pathCam;

using Poco::AutoPtr;
using Poco::Util::XMLConfiguration;
using Poco::Util::LayeredConfiguration;


void printSubKeysXML(XMLConfiguration::Ptr config, std::string key, std::string prefix){
  std::vector<std::string> sub_keys;
  config->keys(key, sub_keys);
  std::cout << prefix << "< " << key << " > " << sub_keys.size() << "\n";
  for(unsigned int i=0; i < sub_keys.size(); i++){
    std::string new_prefix = "\t" + prefix;
    printSubKeysXML(config, sub_keys[i], new_prefix);
  }
}

void printConfigXML(XMLConfiguration::Ptr config)
{
  std::vector<std::string> root_keys;
  config->keys(root_keys);
  
  for(unsigned int i=0; i < root_keys.size(); i++){
    printSubKeysXML(config, root_keys[i], "" );
  }

}


void printSubKeys(LayeredConfiguration::Ptr config, std::string short_key, std::string full_key, std::string print_prefix){
  std::vector<std::string> sub_keys;
  config->keys(full_key, sub_keys);
  std::cout << print_prefix << "< " << short_key << " > " << sub_keys.size() << "\n";
  for(unsigned int i=0; i < sub_keys.size(); i++){
    std::string new_prefix = "\t" + print_prefix;
    printSubKeys(config, sub_keys[i], full_key + "." + sub_keys[i], new_prefix);
  }
}

void printConfig(LayeredConfiguration::Ptr config)
{
  std::vector<std::string> root_keys;
  config->keys(root_keys);
  
  for(unsigned int i=0; i < root_keys.size(); i++){
    printSubKeys(config, root_keys[i], root_keys[i] , "");
  }

}



int main(int argc, char** argv){
  
  Path input_path = Path("../../../resources/config_example.xml");

  AutoPtr<XMLConfiguration> pConf(new XMLConfiguration(input_path.toString()));
  
  Path output_path = Path("../../../resources/output_XML.xml");

  
  pConf->save(output_path.toString());
  

  AutoPtr<LayeredConfiguration> config(new LayeredConfiguration());
  config->add(pConf);
  
  printConfig(config);

  
  return 0;
}
