//
//  RegManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam {


//  void RegistrationRunnable::run() {
//    trace_to_root(index);
//
//    parent->reg_results_mutex->readLock();
//    auto reginfo = parent->reg_results[index];
//    parent->reg_results_mutex->unlock();
//
//    if (!reginfo->resolved) {
//      sort_order += 10;
//      parent->JobQ->add_runnable(this,sort_order);
//      //this should be changed to wait on correct jobComplete event rather than re entered into job queue.
//    } else {
//      parent->regCount--;
//      parent->push_compositeQ(reginfo);
//    }
//    jobComplete.set();
//    jobComplete.reset();
//  }


  void RegistrationRunnable::run() {

    auto them = parent->get_registration(regInfo->matchedTo);
    Vec2 theirAbCs;
    unsigned int componentMembership;

    if(them->get_abc(regInfo, theirAbCs, componentMembership)){
      Vec2 myAbCs;
      myAbCs.x = regInfo->relativeCoords.x + theirAbCs.x;
      myAbCs.y = regInfo->relativeCoords.y + theirAbCs.y;
      regInfo->set_abc(myAbCs, componentMembership);
    }

  }



  std::pair<bool, Vec2> RegistrationRunnable::trace_to_root(unsigned long index) {

    parent->reg_results_mutex->readLock();
    RegInfo* reginfo = parent->reg_results[index];
    parent->reg_results_mutex->unlock();

    if (!reginfo->successful) {
      //this image does not have relative coords yet. original job must be thrown back in Q.
      return std::pair<bool, Vec2>(false, Vec2(0.0, 0.0));

    } else if (reginfo->resolved) {
      //this image has relative coords and absolute coords
      return std::pair<bool, Vec2>(true, reginfo->absoluteCoords);

    } else {
      //this image has relative coords but not absolute coords. We attempt to get absolute coords thru recursive call
      if (reginfo->matchedTo == index){
        return std::pair<bool, Vec2>(false, Vec2(0.0, 0.0));
      }
      Vec2 returnCoords(0,0);
      auto temp = trace_to_root(reginfo->matchedTo);
      if (temp.first) {
        parent->reg_results_mutex->writeLock();
        parent->reg_results[index]->absoluteCoords.x = parent->reg_results[index]->relativeCoords.x + temp.second.x;
        parent->reg_results[index]->absoluteCoords.y = parent->reg_results[index]->relativeCoords.y + temp.second.y;
        parent->reg_results[index]->component_membership = parent->reg_results[parent->reg_results[index]->matchedTo]->component_membership;
        parent->reg_results[index]->resolved = true;
        returnCoords = parent->reg_results[index]->absoluteCoords;
        parent->reg_results_mutex->unlock();
      }

      return std::pair<bool, Vec2>(temp.first, returnCoords);
    }
  }


}
