//
//  QManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include <pathCam.h>

namespace pathCam{

QManager::QManager(StreamCam *parent): parent(parent){};

void QManager::run(){
  Poco::Thread::sleep(200);
  while(parent->microscope_input || parent->diskCount > 0 ||parent->loaderCount > 0 || parent->matchableCount > 0 || parent->regCount > 0){
    
    parent->JobQ->run_jobs(true,true);
    
    
  }
  std::cout<<"jobs processed"<<std::endl;
}


}
