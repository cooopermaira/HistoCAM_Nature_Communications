//
//  CompositeManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//
// This class listens for registered frames, sorts them by component membership, then passes them to the component object for compositing.

#include "pathCam.h"

namespace pathCam {

    CompositeManager::CompositeManager(StreamCam *parent) : parent(parent), successful(false) {};

    void CompositeManager::run() {

        while (parent->microscopeInput || parent->diskCount > 0 || parent->regCount > 0 || parent->loaderCount > 0 ||
               parent->matchableCount > 0 || !parent->compositeQ_empty()) {
            while(!parent->newComponentQ.empty()){
              auto res = parent->newComponentQ.front();
              parent->newComponentQ.pop();
              parent->add_new_component(res.first,res.second);
            }
            //if nothing in the Q but termination condition not met, wait
            if (parent->compositeQ_empty()) {
                Poco::Thread::sleep(100);

            } else {

                std::vector<RegInfo> indexes = parent->get_Q_front();
                std::sort(indexes.begin(), indexes.end());

                while (parent->composites.size() <= indexes.back().component_membership) {
                    //Because of the multithreading, this place in the code can be reached before a new component object has been instantiated and added to the vector. If this happens, wait.
                    Poco::Thread::sleep(100);
                }

                unsigned int current_component = indexes[0].component_membership;
                std::vector<RegInfo> new_info;

                //sort the new frames by component and pass them to their respective components for compositing.
                for (int i = 0; i < indexes.size(); i++) {

                    if (current_component == indexes[i].component_membership) {
                        new_info.push_back(indexes[i]);
                    } else {
                        parent->composites[current_component]->update(new_info);
                        current_component = indexes[i].component_membership;
                        new_info.clear();
                    }
                }

                parent->composites[current_component]->update(new_info);

            }
        }

        

/*
        for (int i = 0; i < parent->composites.size(); i++){
          std::cout << "Writing image of size: " << parent->composites[i]->get_composite().size() << "\n";
          imwrite("finish" + std::to_string(i) + ".png", parent->composites[i]->get_composite());
        }

        for(int i = 0; i < parent->MRimage->level.size(); i++) {
            cv::imwrite("test"+std::to_string(i)+".png", *parent->MRimage->level[i]->cvTiles(0, 0));
        }
        */

      perform_global_alignment();
      parent->compositing = false;
    }

  void CompositeManager::perform_global_alignment() {
    for (auto &i: parent->composites) {
      i->perform_global_alignment();
    }
  }

}
