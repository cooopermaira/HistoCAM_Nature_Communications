//
//  CompositeManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//
// This class listens for registered frames, sorts them by component membership, then passes them to the component object for compositing.
//do not change this class unless you REALLY know what youre doing. this class ultimately handles the order of frames being added even if theyre
//being added by adding a new component. its complicated and fragile.

#include "pathCam.h"
using namespace cv;
using namespace cv::detail;

int lastMC = 0, countDown = 100;
bool hasBeenNonZero = false;


namespace pathCam {
  CompositeManager::CompositeManager(StreamCam *parent) : parent(parent), successful(false),
                                                          rebuildJobsOutstanding(true) {
  };

  void CompositeManager::run() {
    unsigned long duration = 0;
    int updateCount = 0;

#ifdef HAVE_OPENCV_CUDAARITHM
    cuda::setDevice(parent->compositorCudaDevice);
#endif

    rebuildJobsOutstanding = 0;

    std::cout << "composite manager beginning" << std::endl;
    while (parent->microscopeInput || parent->diskCount > 0 || parent->regCount > 0 || parent->loaderCount > 0 ||
           parent->matchableCount > 0 || !parent->compositeQ_empty() || !parent->newComponentQ.empty()) {
      //debug_termination_check();

      //pull new components that might need to be processed
      std::tuple<unsigned long, Size, unsigned int> newComp;
      bool isNewComp = false;
      if (!parent->newComponentQ.empty()) {
        newComp = parent->newComponentQ.front();
        isNewComp = true;
      }

      //if nothing in the Q but termination condition not met, wait
      if (parent->compositeQ_empty()) {
        if (isNewComp) {
          if (parent->inferencing) {
            push_remaining_tiles_for_inference();

            if (!parent->composites.empty()) {
              //parent->composites[parent->composites.size() - 1]->save_pyramid_as_image();
              parent->compositeWait.wait();
            }
          }
          parent->add_new_component(std::get<0>(newComp), std::get<1>(newComp), std::get<2>(newComp));
          parent->newComponentQ.pop();
        } else {
          Poco::Thread::sleep(100);
        }
      } else {
        auto regInfos = parent->get_Q_front(false);
        if (isNewComp) {
          if (regInfos.front()->index > std::get<0>(newComp)) {
            parent->add_new_component(std::get<0>(newComp), std::get<1>(newComp), std::get<2>(newComp));
            parent->newComponentQ.pop();
          }
        }
        regInfos = parent->get_Q_front(true);
        std::sort(regInfos.begin(), regInfos.end());

        // while (parent->composites.size() <= indexes.back()->component_membership) {
        //   //Because of the multithreading, this place in the code can be reached before a new component object has been instantiated and added to the vector. If this happens, wait.
        //   Poco::Thread::sleep(100);
        // }

        if (!regInfos.empty()) {
          Poco::RWLock::ScopedWriteLock lock(parent->component_mutex);
          for (auto ri: regInfos) {
            stage(ri);
          }

          auto start = std::chrono::high_resolution_clock::now();

          for (auto &comp: parent->composites) {
            comp->update();
          }

          auto stop = std::chrono::high_resolution_clock::now();
          duration += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

          parent->notify_observers();
        }
      }

      if (!parent->microscopeInput && parent->loaderCount == 0 && !outstandingSubmitted) {
        outstandingSubmitted = true;
        submit_outstanding_jobs();
      }
    }
    std::cout << "composite loop time: " + std::to_string(duration) << std::endl;

    bool align = true;

    // *******************************************************
    //    ******************** BEGIN POST PROCESSING  ********************
    // *******************************************************


    //process delayed frames
    std::vector<std::thread> threads;
    for (auto &comp: parent->composites) {
      if (comp->suspended) { continue; }

      threads.emplace_back([comp]() {
        for (int i = 0; i < comp->get_exit_rep_count(); ++i) {
          comp->xcMatchShouldContinue = false;
          comp->update();
        }
      });
    }

    for (auto &t: threads) {
      t.join();
    }
    parent->notify_observers();
    threads.clear();

    debug_print_component_status();
    combine_components();
    debug_print_component_status();

    if (align) {
      auto tAlign = std::chrono::high_resolution_clock::now();

      for (auto &comp: parent->composites) {
        if (comp->suspended) {
          comp->imagePyramid->suspended = true;
          continue;
        }

        threads.emplace_back([comp]() {
          auto t1 = std::chrono::high_resolution_clock::now();
          // while (comp->outstandingCMS_jobs > 0 || comp->xcInProgress) {
          //   Poco::Thread::sleep(50);
          // }
          comp->alignmentShouldProceed = false;

          comp->realTimeAlignmentEvent.set();

          if (comp->realtimeAlignmentThread.joinable()) {
            comp->realtimeAlignmentThread.join();
          }
          for (auto &subComp: comp->absorbedComponents) {
            while (subComp->outstandingCMS_jobs > 0) {
              Poco::Thread::sleep(50);
            }
          }
          comp->alignmentHasBegun = true;
          auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t1).
              count();
          std::cout << "component " << comp->componentIndex << " wait time " << t2 << std::endl;
          comp->align_and_rebuild();
        });
      }
      for (auto &t: threads) {
        t.join();
      }
      parent->notify_observers();

      auto tAlignEnd = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - tAlign).count();
      std::cout << std::endl << "total align time " << tAlignEnd << std::endl;
    } else {
      for (auto comp: parent->composites) {
        comp->xcMatchShouldContinue = false;
      }
    }

    for (auto img: parent->images) {
      if (!img) { continue; }
      if (img->get_Raw() || img->get_raw_cuda()) {
        std::cout << img->index << " not freed" << std::endl;
      }
    }


    parent->compositing = false;
  }

  void CompositeManager::push_remaining_tiles_for_inference() {
    // for (auto i: parent->composites) {
    //   parent->push_tile_embed_Q(i->queuedTiles, i->componentIndex);
    //   i->queuedTiles.clear();
    // }
    // parent->inferenceWait.set();
  }

  void CompositeManager::submit_outstanding_jobs() {
    if (parent->maxIndex < 0) { return; }
    for (int i = parent->maxIndex + 1; i <= parent->maxIndex + parent->windowWidth; ++i) {
      parent->JobQ->update_job_readiness(2, i);
    }
  }

  void CompositeManager::perform_global_alignment() {
    // for (auto i: parent->composites) {
    //   //i->perform_global_alignment(0, 0.2);
    //   i->rebuild(CompositeVoronoi::ORB_CPU);
    //
    // }
    // parent->align_and_rebuild();
  }

  void CompositeManager::save_components_to_disk() {
    for (auto i: parent->composites) {
      //i->imagePyramid->level[0]->saveBaseTilesToDisk();

      i->save_pyramid_as_image("/home/cm/Desktop/10x.png", false, false);
    }
  }

  void CompositeManager::combine_components() const {
    for (auto &comp: parent->composites) {
      // Only process roots
      if (comp->joinedTo != comp || comp->suspended) {
        continue;
      }

      auto rootA = comp;

      for (auto cIdx: comp->relatedComponents) {
        auto rootB = parent->joined_to_root(parent->composites[cIdx]);

        // Skip if already unified
        if (rootA->componentIndex == rootB->componentIndex || rootB->suspended) {
          continue;
        }

        std::shared_ptr<Composite> big;
        std::shared_ptr<Composite> small;

        // Special Rule: component 0 always wins
        if (rootA->componentIndex == 0) {
          big = rootA;
          small = rootB;
        } else if (rootB->componentIndex == 0) {
          big = rootB;
          small = rootA;
        } else {
          // Normal union-by-size
          if (rootA->memberCount >= rootB->memberCount) {
            big = rootA;
            small = rootB;
          } else {
            big = rootB;
            small = rootA;
          }
        }

        // Merge data into winner
        big->absorbedComponents.push_back(small);
        big->absorbedComponents.insert(big->absorbedComponents.end(),
                                       small->absorbedComponents.begin(), small->absorbedComponents.end());

        big->newContributingFrames.insert(
          small->contributingFrames.begin(),
          small->contributingFrames.end());

        big->newContributingFrames.insert(
          small->newContributingFrames.begin(),
          small->newContributingFrames.end());

        big->memberCount += small->memberCount;

        // Perform union
        small->joinedTo = big;

        // Suspend loser
        small->suspended = true;

        // Ensure winner is not suspended
        big->suspended = false;

        // Continue with updated root
        rootA = big;
      }
    }
  }

  void CompositeManager::stage(RegInfo *_regInfo) const {
    //reminder - this function is on the composite thread. no need to lock mutexes against composite activity
    _regInfo->accessMutex.lock();
    assert(_regInfo->inCompositeQ);
    _regInfo->inCompositeQ = false;
    if (!_regInfo->image->labelObserved) {
      _regInfo->image->label = parent->composites[_regInfo->component_membership]->componentMagLabel;
    }
    parent->composites[_regInfo->component_membership]->stage(_regInfo);
    _regInfo->staged = true;
    _regInfo->accessMutex.unlock();
  }

  void CompositeManager::debug_print_component_status() const {
    for (auto &comp: parent->composites) {
      if (comp->suspended) {
        std::cout << "component " << comp->componentIndex << " SUSPENDED" << std::endl;
      } else {
        std::cout << "component " << comp->componentIndex << " ACTIVE" << std::endl;
      }
    }
    std::cout << std::endl;
  }
}
