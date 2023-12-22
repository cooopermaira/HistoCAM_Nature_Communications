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
private:
  
  Poco::FastMutex *queue_mutex;
  Poco::ThreadPool *pool;
  std::queue < Poco::Runnable * > jobQueue;
  
  
public:
  JobQueue(int min_threads, int max_threads);
  
  void add_runnable(Poco::Runnable *job);
  bool run_jobs(bool join_all);
  bool run_jobs(std::vector < Poco::Runnable * > jobs);
  bool is_empty(){return jobQueue.empty();}
};

}
#endif /* JobQueue_hpp */
