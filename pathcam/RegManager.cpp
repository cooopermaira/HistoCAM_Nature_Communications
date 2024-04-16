//
//  RegManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam{



//RegManager::RegManager(StreamCam *parent, pathCam::JobQueue *queue): parent(parent), queue(queue), successful(false){};


void RegistrationRunnable::run() {
    auto results = trace_to_root(index);
    if (!parent->reg_results[index].resolved) {
        parent->JobQ->add_runnable(this);
    }
    else {
        parent->regCount--;
        parent->push_compositeQ(parent->reg_results[index]);
    }
}



std::pair<bool, Vec2> RegistrationRunnable::trace_to_root(unsigned long index) {
    if (!parent->reg_results[index].successful) {
        return std::pair<bool, Vec2>(false, Vec2(0.0, 0.0));
    }
    else if (parent->reg_results[index].resolved) {
        return std::pair<bool, Vec2>(true, parent->reg_results[index].absoluteCoords);
    }
    else {
        auto temp = trace_to_root(parent->reg_results[index].matchedTo);
        if (temp.first) {
            parent->reg_results[index].absoluteCoords.x = parent->reg_results[index].relativeCoords.x + temp.second.x;
            parent->reg_results[index].absoluteCoords.y = parent->reg_results[index].relativeCoords.y + temp.second.y;
            parent->reg_results[index].resolved = true;
            parent->reg_results[index].component_membership = parent->reg_results[parent->reg_results[index].matchedTo].component_membership;
        }
        return std::pair<bool, Vec2>(temp.first, parent->reg_results[index].absoluteCoords);
    }
}


/*
void RegManager::run(){
  
  bool check_done = false;
  
  while( parent->microscope_input || !parent->loaderJobQ->is_empty() || !queue->is_empty() || !parent->RegistrationConsecQ.is_empty() ){
    
    //for an image to be registered, all frames between it and a root must also be registered. These frames may not have been matched in that order. ConsecQ serves to assemble consecutive runs of matched images so that they may be registered. This code waits for a run of at least ten before processing, or if it has waited long enough (check_done) proceeds with what it has. This is mostly to allow for final frames to be processed if they dont amount to a run of ten.
    if(parent->RegistrationConsecQ.get_run() > 10 || check_done){
      
      check_done = false;
      std::vector < RegInfo > regvec;
      std::vector < unsigned long int > indexes = parent->RegistrationConsecQ.return_run(parent->RegistrationConsecQ.get_run());
      
      for(int i = 0; i < indexes.size(); i++){
        
        unsigned long int current_index = indexes[i];
        //reg_results only exists at this point for an index if it was a root, otherwise this is finding an expanded spot w/ a default reginfo object
        if(parent->reg_results[current_index].root){
          regvec.push_back(parent->reg_results[current_index]);
          continue;
        }
        
        Vec2 accum = Vec2(0.0,0.0);
        for(long int prev_index = current_index - 1; prev_index >= 0; prev_index--){
          
          if (parent->matchM.match[prev_index][current_index]){
            
            accum.x = parent->reg_results[prev_index].absoluteCoords.x + parent->matchM.match[current_index][prev_index]->t_x;
            
            accum.y = parent->reg_results[prev_index].absoluteCoords.y + parent->matchM.match[current_index][prev_index]->t_y;
            
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
      //termination condition not met but nothing in the Q
      Poco::Thread::sleep(100);
    }else{
      //something in the Q but less than 10
      check_done = true;
      Poco::Thread::sleep(100);
    }//end if
    
  }//end while
  
  parent->reg_complete = true; //part of a termination condition for other processes
  
}//end run()
*/

}
