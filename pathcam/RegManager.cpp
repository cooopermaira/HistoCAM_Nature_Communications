//
//  RegManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam{



RegManager::RegManager(StreamCam *parent, pathCam::JobQueue *queue): parent(parent), queue(queue), successful(false){};



void RegManager::run(){
  
  bool check_done = false;
  
  while(!(parent->disk_empty and parent->jobs_queued and queue->is_empty() and parent->RegistrationConsecQ.is_empty())){
    
    if(parent->RegistrationConsecQ.get_run() > 10 or check_done){
      
      check_done = false;
      std::vector < RegInfo > regvec;
      std::vector < unsigned long int > indexes = parent->RegistrationConsecQ.return_run(parent->RegistrationConsecQ.get_run());
      
      for(int i = 0; i < indexes.size(); i++){
        
        unsigned long int current_index = indexes[i];
        if(parent->reg_results[current_index].root){
          regvec.push_back(parent->reg_results[current_index]);
          continue;
        }
        
        Vec2 accum = Vec2(0.0,0.0);
        for(long int prev_index = current_index - 1; prev_index >= 0; prev_index--){
          
          if (parent->matchM.match[prev_index][current_index]){
            
            accum.x = parent->reg_results[prev_index].vec.x + parent->matchM.match[current_index][prev_index]->t_x;
            
            accum.y = parent->reg_results[prev_index].vec.y + parent->matchM.match[current_index][prev_index]->t_y;
            
            RegInfo temp_ri = RegInfo(true, accum,false,parent->reg_results[prev_index].component_membership);
            temp_ri.index = current_index;
            
            parent->reg_results[current_index] = temp_ri;
            regvec.push_back(temp_ri);
           
            break;
          }//end if
          
        }//end for
        
      }//end for
      
      parent->push_compositeQ(regvec);
      
    }else if(parent->RegistrationConsecQ.get_run() == 0){
      Poco::Thread::sleep(100);
    }else{
      check_done = true;
      Poco::Thread::sleep(100);
    }//end if
  }//end while
  parent->reg_complete = true;
}//end run()


}
