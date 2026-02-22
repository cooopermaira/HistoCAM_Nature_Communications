//
//  CompositeManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//
// This class listens for registered frames, sorts them by component membership, then passes them to the component object for compositing.
//do not change this class unless you REALLY now what youre doing. this class ultimately handles the order of frames being added even if theyre
//being added by adding a new component. its complicated and fragile.

#include "pathCam.h"
using namespace cv;
using namespace cv::detail;

int lastMC = 0, countDown = 100;
bool hasBeenNonZero = false;



namespace pathCam {
  CompositeManager::CompositeManager(StreamCam *parent) : parent(parent), successful(false),
                                                          rebuildJobsOutstanding(true) {};

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
        auto indexes = parent->get_Q_front(false);
        if (isNewComp) {
          if (indexes.front()->index > std::get<0>(newComp)) {
            parent->add_new_component(std::get<0>(newComp), std::get<1>(newComp), std::get<2>(newComp));
            parent->newComponentQ.pop();
          }
        }
        indexes = parent->get_Q_front(true);
        std::sort(indexes.begin(), indexes.end());

        while (parent->composites.size() <= indexes.back()->component_membership) {
          //Because of the multithreading, this place in the code can be reached before a new component object has been instantiated and added to the vector. If this happens, wait.
          Poco::Thread::sleep(100);
        }

        if (!indexes.empty()) {
          Image *lastViewedFrame = nullptr;
          for (auto index: indexes) {
            stage(index);
            lastViewedFrame = index->image;
          }

          auto start = std::chrono::high_resolution_clock::now();
          for (auto &comp: parent->composites) {
            comp->update();
          }
          auto stop = std::chrono::high_resolution_clock::now();
          duration += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

          parent->notify_observers();
          parent->lastViewedFrame = lastViewedFrame;
        }
      }

      if (!parent->microscopeInput && parent->loaderCount == 0 && !outstandingSubmitted) {
        outstandingSubmitted = true;
        submit_outstanding_jobs();
      }
    }
    std::cout << "composite loop time: " + std::to_string(duration) << std::endl;


    //process delayed frames
    std::vector<std::thread> threads;
    for (auto &comp: parent->composites) {
      if (comp->suspended) { continue; }
      auto mc = std::dynamic_pointer_cast<MetricComposite>(comp);

      threads.emplace_back([mc]() {
        for (int i = 0; i < mc->frameDelay; ++i) {
          mc->xcMatchShouldContinue = false;
          mc->update();
        }
      });
    }

    for (auto &t: threads) {
      t.join();
    }
    parent->notify_observers();
    threads.clear();

    auto tAlign = std::chrono::high_resolution_clock::now();


    for (auto &comp: parent->composites) {
      if (comp->suspended) { continue; }
      auto mc = std::dynamic_pointer_cast<MetricComposite>(comp);

      threads.emplace_back([mc]() {
        auto t1 = std::chrono::high_resolution_clock::now();
        while (mc->outstandingCMS_jobs > 0 || mc->xcInProgress) {
          Poco::Thread::sleep(50);
        }
        mc->alignmentHasBegun = true;
        auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).
            count();
        std::cout << "wait time " << t2 << std::endl;
        mc->align_and_rebuild();
      });
    }
    for (auto &t: threads) {
      t.join();
    }
    parent->notify_observers();


    for (auto img: parent->images) {
      if (!img) { continue; }
      if (img->get_Raw() || img->get_raw_cuda()) {
        std::cout << img->index << std::endl;
      }
    }

    auto tAlignEnd = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - tAlign).count();
    std::cout << "total align time " << tAlignEnd << std::endl;
    //
    // tAlign = std::chrono::high_resolution_clock::now();
    // MatchSiftData(reinterpret_cast<MetricComposite *>(parent->composites[0])->compSiftData,reinterpret_cast<MetricComposite *>(parent->composites[1])->compSiftData);
    // std::vector<float> homography(9);
    // int numMatches;
    // FindHomography(reinterpret_cast<MetricComposite *>(parent->composites[0])->compSiftData,homography.data(),&numMatches,10000,0.8,0.9,5);
    // tAlignEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tAlign).count();
    // std::cout << "xc homography time " << tAlignEnd << std::endl;

    push_remaining_tiles_for_inference();



    //save_components_to_disk();

    // if (parent->recordingMode) {
    //   long totalTime = 0;
    //   int totalImages = 0;
    //   for (auto &img:parent->images) {
    //     if (!img){continue;}
    //     totalTime += img->blurTime;
    //     ++totalImages;
    //   }
    //   std::cout<<"write time: "<<totalTime<<"   total images: "<<totalImages<<std::endl;
    // }


    parent->compositing = false;
    parent->inferenceWait.set();
  }

  void CompositeManager::push_remaining_tiles_for_inference() {
    // for (auto i: parent->composites) {
    //   parent->push_tile_embed_Q(i->queuedTiles, i->componentIndex);
    //   i->queuedTiles.clear();
    // }
    // parent->inferenceWait.set();
  }

  void CompositeManager::submit_outstanding_jobs() {
    if (parent->maxIndex < 0){return;}
    for (int i = parent->maxIndex + 1; i <= parent->maxIndex + parent->windowWidth; ++i) {
      parent->JobQ->update_job_readiness(2,i);
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

      i->save_pyramid_as_image("/home/pathcam/pyr.png", true, true);
    }
  }

  void CompositeManager::debug_termination_check() {
    if (parent->matchableCount > 0) {
      hasBeenNonZero = true;
    }
    if (!hasBeenNonZero) { return; }

    if (parent->microscopeInput || parent->diskCount > 0 || parent->regCount > 0 || parent->loaderCount > 0 || parent->matchableCount == 0
      || !parent->compositeQ_empty() || !parent->newComponentQ.empty()) {
      return;
    }

    if (parent->matchableCount == lastMC) {//matchableCount has changed in how long now??
      --countDown;
      if (countDown > 0) {
        return;
      }
      countDown = 100;

    } else {//ok, matchableCount changed so things are still going on
      lastMC = parent->matchableCount;
      countDown = 100;
      return;
    }

    std::vector<long> emptyList;
    auto ans = parent->get_image_ref(emptyList);

    std::vector<RunnableIntermediate *> somehowOutstandingMatchables;
    for (auto &img: ans) {
      auto sojr = parent->JobQ->get_job_ref_index_and_sort_order(2, img->index);
      if (parent->JobQ->jobRefs.size() <= sojr.first) {
        return;
      }
      auto job = parent->JobQ->jobRefs[sojr.first];
      if (job && !job->successful) {
        somehowOutstandingMatchables.push_back(job);
      }
    }
    std::vector<RunnableIntermediate *> unprocessedJobs, canceledJobs;
    for (auto &job: somehowOutstandingMatchables) {
      if (job->unprocessed) {
        unprocessedJobs.push_back(job);
      } else {
        canceledJobs.push_back(job);
      }
    }
    for (int i = 0; i <= parent->maxIndex; ++i) {
      if (parent->matchablesIncremented[i] != 1) {
        std::cout<<i<<" incremented "<<parent->matchablesIncremented[i]<<" times"<<std::endl;
      }
      if (parent->matchablesDecremented[i] != 1) {
        std::cout<<i<<" decremented "<<parent->matchablesDecremented[i]<<" times"<<std::endl;
      }
    }
    //parent->compositing = false;
    int k = 0;
  }


  void CompositeManager::stage(RegInfo *_regInfo) const {
    _regInfo->accessMutex.lock();
    assert(_regInfo->inCompositeQ);
    _regInfo->inCompositeQ = false;
    if (!_regInfo->image->labelObserved) {
      _regInfo->image->label = parent->composites[_regInfo->component_membership]->componentMagLabel;
    }
    parent->composites[_regInfo->component_membership]->stage(_regInfo);
    _regInfo->accessMutex.unlock();
  }
}
