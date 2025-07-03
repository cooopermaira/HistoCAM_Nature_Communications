//
//  JobQueue.hpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#ifndef JobQueue_h
#define JobQueue_h

#include <stdio.h>
#include "pathCam.h"

namespace pathCam{

  class JobQueue{
    friend class QManager;
    friend class MatchRunnable;
  public:
    static bool comp_sort_order(const RunnableIntermediate *a, const RunnableIntermediate *b);
    std::vector<RunnableIntermediate*> jobRefs;
    std::vector<int> jobsReadiness;
    std::vector<bool> cancelJob;
    Poco::Event *pathCamEvent;

    StreamCam *parent;


    struct CompareRunnable {
      bool operator()(const RunnableIntermediate *a, const RunnableIntermediate *b);
    };
    Poco::FastMutex *queue_mutex;
    Poco::ThreadPool *pool;
    //std::deque < RunnableIntermediate *> jobQueue;
    std::priority_queue<RunnableIntermediate*,std::deque<RunnableIntermediate*>,CompareRunnable> jobQueue;

    int windowWidth;


  public:
    JobQueue(int min_threads, int max_threads,int windowWidth = 3);
    void add_runnable(RunnableIntermediate *job, long sortOrder = -1);
    void cancel_job(int jobTypeFlag, unsigned long image_idx);
    bool run_jobs(bool join_all);
    [[nodiscard]] bool is_empty() const {return jobQueue.empty();}

    std::pair<long, unsigned long> get_job_ref_index_and_sort_order(int jobTypeFlag, unsigned long image_idx);
    void update_job_readiness(int jobTypeFlag, unsigned long image_idx);
  };

}
#endif /* JobQueue_hpp */
