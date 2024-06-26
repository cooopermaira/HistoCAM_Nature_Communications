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
        sort_order += 10;
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


}
