//
//  JobQueue.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam{


JobQueue::JobQueue(int min_threads, int max_threads){
  queue_mutex = new Poco::FastMutex(),
  pool = new Poco::ThreadPool(min_threads,max_threads,60,POCO_THREAD_STACK_SIZE);
}

void JobQueue::add_runnable(Poco::Runnable *job){
  queue_mutex->lock();
  jobQueue.push(job);
  queue_mutex->unlock();
};

bool JobQueue::run_jobs(bool join_all){
  while(!jobQueue.empty()){
    if(pool->available() > 0){
      queue_mutex->lock();
      pool->start(*jobQueue.front());
      jobQueue.pop();
      queue_mutex->unlock();
    }else{
      Poco::Thread::sleep(100);
    }
  }
  if(join_all){pool->joinAll();}
  return true;
};

bool JobQueue::run_jobs(std::vector < Poco::Runnable * > jobs){
  for(unsigned int i=0; i < jobs.size(); i++){
    jobQueue.push(jobs[i]);
  }
  
  while(!jobQueue.empty()){
    if(pool->available() > 0){
      pool->start(*jobQueue.front());
      jobQueue.pop();
    }else{
      Poco::Thread::sleep(100);
    }
  }
  return true;
  
  pool->joinAll();
}

}
