//
//  Runnables.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <stdio.h>
#include "pathCam.h"

namespace pathCam {
    MatchRunnable::MatchRunnable(StreamCam *parent, unsigned long image_idx,
                                 unsigned long sort_order) : RunnableIntermediate(sort_order), parent(parent),
                                                             image_idx(image_idx),
                                                             successful(false) {
    };


    void MatchRunnable::run() {
        pathCam::Image *image = parent->get_image_ref(image_idx);


        if (!image->is_good()) {
            return;
        }

        pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(parent->matcher_type);

        pathCam::MotionEstimator *motion_est = new pathCam::MotionEstimator();

        for (long int prev_idx = image_idx - 1; prev_idx >= 0; prev_idx--) {
            pathCam::Image *previous = parent->get_image_ref(prev_idx);
            if (previous == nullptr) {
                continue;
            }

            if (!previous->is_good()) { continue; }
            parent->matchM.match[prev_idx][image_idx] = new Match(previous, image);
            Match *m = parent->matchM.match[prev_idx][image_idx];
            matcher->match(m);
            int result = motion_est->findHomography(m, parent->estimator_type);
            if (result == 1) {
                if (std::abs(parent->matchM.match[prev_idx][image_idx]->t_x) < image->width / 2 && std::abs(
                        parent->matchM.match[prev_idx][image_idx]->t_y) < image->height / 2) {
                    //parent->matchM.match[image_idx][prev_idx] = new Match(parent->matchM.match[prev_idx][image_idx]);
                    parent->set_match(image_idx, prev_idx);
                    auto tempReg = RegInfo(true, Vec2(0.0, 0.0), false, 0);
                    tempReg.index = image_idx;
                    tempReg.resolved = false;
                    tempReg.matchedTo = prev_idx;
                    tempReg.relativeCoords.x = parent->matchM.match[image_idx][prev_idx]->t_x;
                    tempReg.relativeCoords.y = parent->matchM.match[image_idx][prev_idx]->t_y;
                    parent->reg_results[image_idx] = tempReg;
                    successful = true;
                    auto rj = new RegistrationRunnable(parent, image_idx, sort_order + 20);
                    parent->regCount++;
                    parent->JobQ->add_runnable(rj);
                    break;
                } else {
                    parent->matchM.match[prev_idx][image_idx] = nullptr;
                }
            } else {
                // if(result == -1 || result == -2){
                parent->matchM.match[prev_idx][image_idx] = nullptr;
            }
            delete m;
        }

        if (!successful) {
            parent->add_new_component(image_idx, cv::Size(image->width, image->height));
        }
        //parent->RegistrationConsecQ.add_index(image_idx);

        delete matcher;
        delete motion_est;
        parent->matchableCount--;
    } //end run
}; //end namespace
