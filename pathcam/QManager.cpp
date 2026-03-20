//
//  QManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include <pathCam.h>

namespace pathCam {
  void QManager::run() {
    std::cout << "q manager beginning" << std::endl;
    auto jq = parent->JobQ;
    auto jq2 = parent->jqSecondary;
    // Poco::Thread::sleep(200);
    int count = 1;

    while (parent->compositing) {
      // if (count % 10 == 0) {
      //   parent->launch_blur_metric();
      // }else if (count % 10 == 5) { //just gives a little time for it to run
      //   parent->receive_blur_metric();
      // }
      // ++count;
      Poco::Runnable *job = nullptr;
      if (jq->pool->available()) {
        {
          Poco::FastMutex::ScopedLock lock(jq->queue_mutex);
          if (!jq->jobQueue.empty()) {
            job = jq->jobQueue.top();
            jq->jobQueue.pop();
          }
        }
        if (job) jq->pool->start(*job);
        job = nullptr;
      }

      if (jq2->pool->available()) {
        {
          Poco::FastMutex::ScopedLock lock(jq2->queue_mutex);
          if (!jq2->jobQueue.empty()) {
            job = jq2->jobQueue.top();
            jq2->jobQueue.pop();
          }
        }
        if (job) jq2->pool->start(*job);
      }
    }
    jq->pool->joinAll();
    jq2->pool->joinAll();
  }
}
