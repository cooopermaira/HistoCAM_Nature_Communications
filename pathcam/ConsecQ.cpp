//
//  conseq_Q.cpp
//  pathCam
//
//  Created by cooper maira on 12/18/23.
//

#include "pathCam.h"

namespace pathCam{

ConsecQ::ConsecQ(): deque_mutex(new Poco::FastMutex()){}
  
bool ConsecQ::is_empty(){
  deque_mutex->lock();
  unsigned long int val = c_queue.size();
  long int val2 = c_queue.front();
  deque_mutex->unlock();
  return (val == 0);
}
void ConsecQ::add_index(long int idx){
  deque_mutex->lock();
  c_queue.push_back(idx);
  std::sort(c_queue.begin(),c_queue.end());
  deque_mutex->unlock();
  }
  
unsigned int ConsecQ::get_run(){
  unsigned int length = 1;
  deque_mutex->lock();
  if(c_queue.empty()){
    deque_mutex->unlock();
    return 0;
  }
  long int current_index = c_queue.front();
  for (unsigned int i = 1; i < c_queue.size(); i++){
    if (current_index + 1 == c_queue.at(i)){
      current_index++;
      length++;
      continue;
    }else{break;}
  }
  deque_mutex->unlock();
  return length;
}

std::vector<long int> ConsecQ::return_run(unsigned int length){
  std::vector<long int> k;
  deque_mutex->lock();
  for(unsigned int i = 0;i<length;i++){
    k.push_back(c_queue.front());
    c_queue.pop_front();
  }
  deque_mutex->unlock();
  return k;
  }

}
