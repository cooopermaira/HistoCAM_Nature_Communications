//
//  JobQueue.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam {
  JobQueue::JobQueue(int min_threads, int max_threads) {
    queue_mutex = new Poco::FastMutex(),
        pathCamEvent = new Poco::Event(true);
    pool = new Poco::ThreadPool(min_threads, max_threads, 60, POCO_THREAD_STACK_SIZE);
  }
  std::pair<int,unsigned long> JobQueue::getSortOrderAndJobRefs(int jobTypeFlag, unsigned long image_idx)
  {
    /*
     * Job Type Flags:
     * 0: Other
     * 1: Loader
     * 2: Match
     * 3: Registration
     */
    unsigned long sort_order;
    int jobRefNumber;
    if (jobTypeFlag == 1)
    {
      sort_order = image_idx;
      jobRefNumber = image_idx * 3 + jobTypeFlag;
    }
    else
    if (jobTypeFlag == 2)
    {
      sort_order = image_idx + 20;
      jobRefNumber = image_idx * 3 + jobTypeFlag;
    }
    else if (jobTypeFlag == 3)
    {
      sort_order = image_idx + 1;
      jobRefNumber = image_idx * 3 + jobTypeFlag;
    }
    else
    {
      sort_order = 0;
      jobRefNumber = -1;
    }
    return std::make_pair(jobRefNumber, sort_order);
  }
  void JobQueue::add_runnable(RunnableIntermediate *job, long _sortOrder) {
    if(_sortOrder == -1) {
      std::pair<int, int> jr_so_pair = getSortOrderAndJobRefs(job->jobTypeFlag, job->image_index);
      job->jobRefNumber = jr_so_pair.first;
      job->sort_order = jr_so_pair.second;
    }else{
      job->sort_order = _sortOrder;
    }
    queue_mutex->lock();
    if (job->jobRefNumber >= 0) {
      if (jobRefs.size() <= job->jobRefNumber) {
        jobRefs.resize(job->jobRefNumber + 400);
      }
      jobRefs[job->jobRefNumber] = job;
    }
    jobQueue.push(job);
    queue_mutex->unlock();
  };

  bool JobQueue::run_jobs(bool join_all) {
    queue_mutex->lock();
    int batchSize = std::min(20, (int) jobQueue.size());
    queue_mutex->unlock();

    for (int i = 0; i < batchSize; i++) {

      if (pool->available() > 0) {
        queue_mutex->lock();
        pool->start(*jobQueue.top());
        jobQueue.pop();
        queue_mutex->unlock();
      } else {
        Poco::Thread::sleep(10);
      }
    }

    if (join_all) {
      pool->joinAll();
    }
    return true;
  };

  bool JobQueue::comp_sort_order(const RunnableIntermediate *a, const RunnableIntermediate *b) {
    unsigned long a1 = a->sort_order;
    unsigned long b1 = b->sort_order;
    bool res = a1 < b1;
    return res;
  }

  bool JobQueue::CompareRunnable::operator()(const RunnableIntermediate *a, const RunnableIntermediate *b) {
    return a->sort_order > b->sort_order;
  }
}
