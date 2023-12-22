//
//  conseq_Q.hpp
//  pathCam
//
//  Created by cooper maira on 12/18/23.
//

#ifndef conseq_Q_hpp
#define conseq_Q_hpp

#include <stdio.h>
#include "pathCam.h"


namespace pathCam{

 class ConsecQ{
 private:
 std::deque<long int> c_queue;
 Poco::FastMutex *deque_mutex;
 
 public:
 ConsecQ();
 ~ConsecQ(){delete deque_mutex;}
 
 void add_index(long int);
 unsigned int get_run();
 
 bool is_empty();
 std::vector<long int> return_run(unsigned int);
 };

}
#endif /* conseq_Q_hpp */
