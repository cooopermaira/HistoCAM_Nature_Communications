//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

#include "Poco/ThreadPool.h"
#include "Poco/Runnable.h"
#include <iostream>

class ThreadQueue{
private:
  Poco::ThreadPool *pool;
  std::queue < Poco::Runnable * > jobQueue;

public:
  ThreadQueue(int min_threads, int max_threads){
    pool = new Poco::ThreadPool(min_threads,max_threads,60,POCO_THREAD_STACK_SIZE);
  }
  
  bool run_jobs(std::vector < Poco::Runnable * > jobs){
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
    
    pool->joinAll();
  }

  
};


class HelloRunnable: public Poco::Runnable{
private:
  int temp;

public:
  
  HelloRunnable(int temp): temp(temp){};

  virtual void run(){
    Poco::Thread::sleep(5000);
    std::cout << temp << std::endl;
  }
    
};


int main(int argc, char** argv){

  std::vector < Poco::Runnable * > jobs;
  
  for (unsigned int i=0; i < 20; i++){
    jobs.push_back(new HelloRunnable(i));
  }
 
  ThreadQueue q(2,5);
  q.run_jobs(jobs);
  
  
  return 0;
}
