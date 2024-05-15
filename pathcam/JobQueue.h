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
public:
  static bool comp_sort_order(const RunnableIntermediate *a, const RunnableIntermediate *b);
private:

    struct CompareRunnable {
        bool operator()(const RunnableIntermediate *a, const RunnableIntermediate *b); 
    };
  Poco::FastMutex *queue_mutex;
  Poco::ThreadPool *pool;
  //std::deque < RunnableIntermediate *> jobQueue;
std::priority_queue<RunnableIntermediate*,std::deque<RunnableIntermediate*>,CompareRunnable> jobQueue;
  
  
public:
  JobQueue(int min_threads, int max_threads);
  
  void add_runnable(RunnableIntermediate *job);
  bool run_jobs(bool join_all, bool order_before_run);
  bool run_jobs(std::vector < Poco::Runnable * > jobs);
  bool is_empty(){return jobQueue.empty();}

};

}
#endif /* JobQueue_hpp */
