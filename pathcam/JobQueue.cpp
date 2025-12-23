//
//  JobQueue.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam {
  JobQueue::JobQueue(int min_threads, int max_threads, int _windowWidth) {
    windowWidth = _windowWidth;
    queue_mutex = new Poco::FastMutex(),
        pathCamEvent = new Poco::Event(true);
    pool = new Poco::ThreadPool(min_threads, max_threads, 60, POCO_THREAD_STACK_SIZE);
  }

  std::pair<long, unsigned long> JobQueue::get_job_ref_index_and_sort_order(int jobTypeFlag, unsigned long image_idx)
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
      jobRefNumber = image_idx * 3 + jobTypeFlag - 1;
    }
    else
    if (jobTypeFlag == 2)
    {
      sort_order = image_idx + 20;
      jobRefNumber = image_idx * 3 + jobTypeFlag - 1;
    }
    else if (jobTypeFlag == 3)
    {
      sort_order = image_idx + 1;
      jobRefNumber = image_idx * 3 + jobTypeFlag - 1;
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
      std::pair<int, int> jr_so_pair = get_job_ref_index_and_sort_order(job->jobTypeFlag, job->image_index);
      job->jobRefNumber = jr_so_pair.first;
      job->sort_order = jr_so_pair.second;
    }else{
      job->sort_order = _sortOrder;
    }
    queue_mutex->lock();
    if (job->jobRefNumber >= 0) {
      if (jobRefs.size() <= job->jobRefNumber + 50) {
        jobRefs.resize(job->jobRefNumber + 400);
        jobsReadiness.resize(job->jobRefNumber + 400,0);
        cancelJob.resize(job->jobRefNumber + 400,false);
      }
      jobRefs[job->jobRefNumber] = job;
    }
    if (job->jobTypeFlag == 0) {
      jobQueue.push(job);
    } else {
      update_job_readiness(job->jobTypeFlag, job->image_index);
    }
    queue_mutex->unlock();

  };

  void JobQueue::update_job_readiness(int jobTypeFlag, unsigned long image_idx) {
    if (jobTypeFlag == 2) {

      int fv = max(0,int(image_idx) - windowWidth);
      int lv = image_idx + windowWidth;
      std::vector<int> iters(lv - fv + 1);
      std::iota(iters.begin(),iters.end(),fv);

      for (auto i : iters){

        auto answer = get_job_ref_index_and_sort_order(jobTypeFlag, i);

        jobsReadiness[answer.first]++;

        unsigned long readinessRequired = 7 + min(i - windowWidth, 0);


        if (jobsReadiness[answer.first] >= readinessRequired && jobRefs[answer.first] && jobRefs[answer.first]->unprocessed) {
          //enough of this job's neighbors have processed, this job has enough information to run.
          if (cancelJob[answer.first]) {
            --parent->matchableCount;
            jobRefs[answer.first]->unprocessed = false;

            auto img = parent->get_image_ref(i);
            img->mark_too_dark();
            img->free_memory_RAW();

          }else {

            jobQueue.push(jobRefs[answer.first]);
            jobRefs[answer.first]->unprocessed = false;

          }
        }
      }
    }else {
      auto answer = get_job_ref_index_and_sort_order(jobTypeFlag, image_idx);
      if (jobRefs[answer.first]->unprocessed) {
        jobQueue.push(jobRefs[answer.first]);
        jobRefs[answer.first]->unprocessed = false;
      }
    }
  }

  void JobQueue::cancel_job(int jobTypeFlag, unsigned long image_idx) {

    auto answer = get_job_ref_index_and_sort_order(jobTypeFlag, image_idx);
    queue_mutex->lock();
    cancelJob[answer.first] = true;
    queue_mutex->unlock();
  }

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
