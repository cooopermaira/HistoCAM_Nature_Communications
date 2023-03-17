//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

using namespace pathCam;

int main(int argc, char **argv)
{
  std::string temp = "/Users/bsumma/source/pathcam/resources/config_example.xml";
  Poco::Path temp_path = Path(temp);
  
  BatchCam *batch = new BatchCam(temp_path);
    
  
  delete batch;
  return 0;
}
