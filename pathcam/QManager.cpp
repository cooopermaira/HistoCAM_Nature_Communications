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
  auto jq = parent->JobQ;
  Poco::Thread::sleep(200);
  while(parent->compositing){
    //jq->run_jobs(false);

    while(jq->pool->available() && !jq->jobQueue.empty()){
      jq->queue_mutex->lock();

      try {
        jq->pool->start(*jq->jobQueue.top());
      }
      catch(Poco::Exception &e){
        int k = 0;
      }

      jq->jobQueue.pop();
      jq->queue_mutex->unlock();
    }

  }
}


}
