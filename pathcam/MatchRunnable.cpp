//
//  Runnables.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include "pathCam.h"

namespace pathCam{


MatchRunnable::MatchRunnable(StreamCam *parent, unsigned long int image_idx): parent(parent), image_idx(image_idx), successful(false){};

void MatchRunnable::run(){
  
  pathCam::Image * image = parent->get_image_ref(image_idx);
  
  if(!image->is_good()){
    return;
  }
  
  pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);
  pathCam::MotionEstimator *mot = new pathCam::MotionEstimator();
  
  for(long int prev_idx=image_idx-1; prev_idx >=0; prev_idx--){
    
    pathCam::Image *previous = parent->get_image_ref(prev_idx);
    
    if(!previous->is_good()){ continue; }
    parent->matchM.match[prev_idx][image_idx] = new pathCam::Match(previous,image);
    pathCam::Match *m = parent->matchM.match[prev_idx][image_idx];
    matcher->match(m);
    int result = mot->findHomography(m, parent->estimator_type);
    if(result == 1){
      parent->matchM.match[image_idx][prev_idx] = new pathCam::Match(parent->matchM.match[prev_idx][image_idx]);
      
      successful = true;
      break;
    }
    else if(result == -1 or result == -2){
      parent->matchM.match[prev_idx][image_idx] = NULL;
    }
    delete m;
  }
  //beginning of new connected component
  if(!successful){parent->reg_results[image_idx] = RegInfo(false,Vec2(0.0,0.0),true);}
  parent->ConseqQ.add_index(image_idx);
  delete matcher;
  delete mot;
  
}

};


