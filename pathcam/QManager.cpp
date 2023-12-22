//
//  QManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include <pathCam.h>

namespace pathCam{

QManager::QManager(StreamCam *parent, JobQueue *queue): parent(parent), queue(queue){};

void QManager::run(){
  while(!parent->disk_empty and !parent->disk_empty){
    
    queue->run_jobs(false);
    
    Poco::Thread::sleep(100);
  }
}


}
