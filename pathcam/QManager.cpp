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
    if (jq->pool->available()){
      if (!jq->is_empty())
      {
        jq->queue_mutex->lock();
        auto j = jq->jobQueue.top();
        jq->pool->start(*jq->jobQueue.top());
        jq->jobQueue.pop();
        jq->queue_mutex->unlock();
      }
    }

//    else {
//      jq->pool->threadAvailableEvent->wait();
//    }

  }
}


}
