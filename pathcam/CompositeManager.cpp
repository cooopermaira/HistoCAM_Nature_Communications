//
//  CompositeManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//

#include "pathCam.h"

namespace pathCam{

CompositeManager::CompositeManager(StreamCam *parent):parent(parent),successful(false){};

void CompositeManager::run(){
  while(!parent->reg_complete or !parent->compositeQ_empty()){
    
    if( parent->compositeQ_empty()){
      Poco::Thread::sleep(100);
      
    }else{
      std::vector < RegInfo > indexes = parent->get_Q_front();
      std::sort(indexes.begin(),indexes.end());
      
      while(parent->composites.size() <= indexes.back().component_membership){
        Poco::Thread::sleep(100);
      }
      
      unsigned int i = 0;
      unsigned int current_component = indexes[0].component_membership;
      
      while(i < indexes.size()){
        std::vector < RegInfo > new_info;
        
        while(indexes[i].component_membership == current_component and i < indexes.size()){
          new_info.push_back( indexes[i] );
          i++;
        }
        parent->composites[current_component]->update(new_info);
        if(i < indexes.size()){
          current_component = indexes[i].component_membership;

        }
      }
    }
  }
  
  for (int i = 0; i < parent->composites.size(); i++){
    imwrite("finish"+std::to_string(i)+".png", parent->composites[i]->get_composite());
  }
  
  
}

}
