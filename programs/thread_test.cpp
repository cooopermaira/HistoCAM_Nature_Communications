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

using namespace pathCam;


class HelloRunnable: public Poco::Runnable{
public:
  
  HelloRunnable(){};

  virtual void run(){
    std::cout << "Hellow World" << std::endl;
  }
    
};


int main(int argc, char** argv){
  
  unsigned int threads = 4;

  std::vector < HelloRunnable > runnable(threads);
  std::vector < Poco::Thread > thread(threads);
  
  
  for(unsigned int i=0; i < threads; i++){
    thread[i].start(runnable[i]);
  }
  
  for(unsigned int i=0; i < threads; i++){
    thread[i].join();
  }
  
  return 0;
}
