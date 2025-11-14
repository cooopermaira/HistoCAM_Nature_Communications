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
          Image* lastViewedFrame = nullptr;
          for (auto index : indexes) {
            parent->composites[index->component_membership]->stage(index);
            lastViewedFrame = index->image;
          }

          auto start = std::chrono::high_resolution_clock::now();
          for (auto &comp : parent->composites) {
            comp->update();
          }
          auto stop = std::chrono::high_resolution_clock::now();
          duration += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

          parent->notify_observers();
          parent->lastViewedFrame = lastViewedFrame;
        }


      }

      //check_render_info();

      if (!parent->microscopeInput && parent->loaderCount == 0) {
        submit_outstanding_jobs();
      }
    }


    //process delayed frames
    for (auto &comp: parent->composites) {
      auto mc = reinterpret_cast<MetricComposite*>(comp);
      for (int i = 0; i < mc->frameDelay; ++i) {
        mc->update();
      }
      mc->mostRecentFrame->free_memory_RAW();
    }

    std::cout << "CM duration: " + std::to_string(duration) << std::endl;



    push_remaining_tiles_for_inference();

    std::cout<<"here: "<<reinterpret_cast<MetricComposite*>(parent->composites[0])->debugFrameCount<<std::endl;
    std::cout<<"immediate: "<<reinterpret_cast<MetricComposite*>(parent->composites[0])->debugTileCount1<<std::endl;
    std::cout<<"later: "<<reinterpret_cast<MetricComposite*>(parent->composites[0])->debugTileCount2<<std::endl;



    //save_components_to_disk();

    if (parent->recordingMode) {
      long totalTime = 0;
      int totalImages = 0;
      for (auto &img:parent->images) {
        if (!img){continue;}
        totalTime += img->writeTime;
        ++totalImages;
      }
      std::cout<<"write time: "<<totalTime<<"   total images: "<<totalImages<<std::endl;
    }

    if (parent->segmentWithSAM) {
      auto start = std::chrono::high_resolution_clock::now();
      perform_global_alignment();
      //parent->as->initialize();
      //parent->as->embed_SAM_tiles(false);
      parent->as->load_model();


      auto stop = std::chrono::high_resolution_clock::now();
      std::cout<<"SAM initialization runtime: "+std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count())<<std::endl;;
    }

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
    for (int i = parent->maxIndex - parent->windowWidth; i <= parent->maxIndex; i++) {
      parent->JobQ->update_job_readiness(2, i);
    }
  }

  void CompositeManager::perform_global_alignment() {
    // for (auto i: parent->composites) {
    //   //i->perform_global_alignment(0, 0.2);
    //   i->rebuild(CompositeVoronoi::ORB_CPU);
    //
    // }
    parent->align_and_rebuild();
  }

  void CompositeManager::save_components_to_disk() {
    for (auto i: parent->composites) {
      //i->imagePyramid->level[0]->saveBaseTilesToDisk();

      i->save_pyramid_as_image("/home/pathcam/pyr.png",true,true);
    }
  }

  void CompositeManager::debug_termination_check() {
    if (parent->matchableCount > 11) {
      return;
    }
    std::vector<unsigned long> emptyList;
    auto ans = parent->get_image_ref(emptyList);

    std::vector<RunnableIntermediate*> somehowOutstandingMatchables;
    for (auto &img : ans) {
      auto sojr = parent->JobQ->get_job_ref_index_and_sort_order(2,img->index);
      if (parent->JobQ->jobRefs.size() <= sojr.first) {
        return;
      }
      auto job = parent->JobQ->jobRefs[sojr.first];
      if (job && !job->successful) {
        somehowOutstandingMatchables.push_back(job);
      }
    }
    std::vector<RunnableIntermediate*> unprocessedJobs,canceledJobs;
    for (auto &job : somehowOutstandingMatchables) {
      if (job->unprocessed) {
        unprocessedJobs.push_back(job);
      }else {
        canceledJobs.push_back(job);
      }
    }
    int k = 0;
  }







}
