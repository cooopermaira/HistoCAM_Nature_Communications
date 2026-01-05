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
    auto jq = parent->JobQ;
    auto jq2 = parent->jqSecondary;
    Poco::Thread::sleep(200);
    int count = 1;
    auto start = std::chrono::high_resolution_clock::now();

    while (parent->compositing) {

      // if (count % 10 == 0) {
      //   parent->launch_blur_metric();
      // }else if (count % 10 == 5) { //just gives a little time for it to run
      //   parent->receive_blur_metric();
      // }
      ++count;

      if (jq->pool->available() && !jq->is_empty()) {
          jq->queue_mutex->lock();
          jq->pool->start(*jq->jobQueue.top());
          jq->jobQueue.pop();
          jq->queue_mutex->unlock();
      }
      if (jq2->pool->available() && !jq2->is_empty()) {
        jq2->queue_mutex->lock();
        jq2->pool->start(*jq2->jobQueue.top());
        jq2->jobQueue.pop();
        jq2->queue_mutex->unlock();
      }

    }
  }
}
