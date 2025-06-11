//
//  CompositeManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//
// This class listens for registered frames, sorts them by component membership, then passes them to the component object for compositing.

#include "pathCam.h"
using namespace cv;
using namespace cv::detail;

namespace pathCam {
  CompositeManager::CompositeManager(StreamCam *parent) : parent(parent), successful(false),
                                                          rebuildJobsOutstanding(true) {
  };

  void CompositeManager::run() {
    unsigned long duration = 0;

#ifdef HAVE_OPENCV_CUDAARITHM
    cuda::setDevice(parent->compositorCudaDevice);
#endif

    rebuildJobsOutstanding = 0;
    while (parent->microscopeInput || parent->diskCount > 0 || parent->regCount > 0 || parent->loaderCount > 0 ||
           parent->matchableCount > 0 || !parent->compositeQ_empty() || !parent->newComponentQ.empty()) {
      //pull new components that might need to be processed
      std::tuple<unsigned long, cv::Size, unsigned int> newComp;
      bool isNewComp = false;
      if (!parent->newComponentQ.empty()) {
        newComp = parent->newComponentQ.front();
        isNewComp = true;
      }

      //if nothing in the Q but termination condition not met, wait
      if (parent->compositeQ_empty()) {
        if (isNewComp) {
          //          perform_global_alignment();
          //          if (rebuildJobsOutstanding > 0) {
          //            rebuildJobsComplete.wait();
          //          }
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
          check_render_info();
          Poco::Thread::sleep(100);
        }
      } else {
        auto indexes = parent->get_Q_front();
        if (isNewComp) {
          if (indexes.front()->index > std::get<0>(newComp)) {
            //perform_global_alignment();
            if (rebuildJobsOutstanding > 0) {
              rebuildJobsComplete.wait();
            }
            parent->add_new_component(std::get<0>(newComp), std::get<1>(newComp), std::get<2>(newComp));
            parent->newComponentQ.pop();
          }
        }
        std::sort(indexes.begin(), indexes.end());

        while (parent->composites.size() <= indexes.back()->component_membership) {
          //Because of the multithreading, this place in the code can be reached before a new component object has been instantiated and added to the vector. If this happens, wait.
          Poco::Thread::sleep(100);
        }

        unsigned int current_component = indexes[0]->component_membership;
        std::vector<RegInfo *> new_info;

        //sort the new frames by component and pass them to their respective components for compositing.
        for (int i = 0; i < indexes.size(); i++) {
          if (current_component == indexes[i]->component_membership) {
            new_info.push_back(indexes[i]);
          } else {
            auto start = std::chrono::high_resolution_clock::now();
            parent->composites[current_component]->update(new_info);
            auto stop = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
            current_component = indexes[i]->component_membership;
            new_info.clear();
          }
        }
        auto start = std::chrono::high_resolution_clock::now();
        parent->composites[current_component]->update(new_info);
        auto stop = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
      }

      check_render_info();

      if (!parent->microscopeInput && parent->loaderCount == 0) {
        submit_outstanding_jobs();
      }
    }


    push_remaining_tiles_for_inference();

    //perform_SIFT_multires_bundle_adjustment();
    perform_global_alignment();
    //rebuildJobsComplete.wait();
    //save_components_to_disk();

    std::cout << "CM duration: " + std::to_string(duration) << std::endl;
    parent->compositing = false;
    parent->inferenceWait.set();
  }

  void CompositeManager::push_remaining_tiles_for_inference() {
    for (auto i: parent->composites) {
      parent->push_tile_embed_Q(i->queuedTiles, i->componentIndex);
      i->queuedTiles.clear();
    }
    parent->inferenceWait.set();
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

      i->save_pyramid_as_image("/Users/coopermaira/Desktop/image_dump/full4x_green.png", false, false, false, false);
    }
  }

  void CompositeManager::align_new_comp() {
    parent->newComponentQ.pop();
    perform_global_alignment();
    if (rebuildJobsOutstanding > 0) {
      rebuildJobsComplete.wait();
    }
  }

  void CompositeManager::check_render_info() {
    for (int i = 1; i < parent->composites.size(); i++) {
      parent->composites[i]->check_set_render_info();
    }
  }

  void CompositeManager::decrement_rebuild_jobs_outstanding() {
    rebuildJobsOutstanding--;
    parent->notify_observers();
    if (rebuildJobsOutstanding == 0) {
      rebuildJobsComplete.set();
    }
  }

  void CompositeManager::perform_SIFT_multires_bundle_adjustment() {
    std::vector<std::pair<Image*,Image*> > imagePairsToMatch;
    std::vector<RegInfo*> regs;
    build_match_pairs(imagePairsToMatch,regs);

    std::vector<MatchesInfo> matchesInfo(imagePairsToMatch.size());
    for (int i = 0; i < matchesInfo.size(); ++i) {
      //call params
      auto sd1 = imagePairsToMatch[i].first->siftData;
      auto sd2 = imagePairsToMatch[i].second->siftData;
      auto idx1 = imagePairsToMatch[i].first->index;
      auto idx2 = imagePairsToMatch[i].second->index;

      matchesInfo[i] = compute_matches_info(sd1, sd2, idx1, idx2);
    }
    int k = 0;
  }


  MatchesInfo CompositeManager::compute_matches_info(SiftData &_sift1, SiftData &_sift2, unsigned long _img1_idx, unsigned long _img2_idx) {
    MatchesInfo matches_info;
    matches_info.src_img_idx = _img1_idx;
    matches_info.dst_img_idx = _img2_idx;

    // Run matching in both directions
    MatchSiftData(_sift1, _sift2);
    MatchSiftData(_sift2, _sift1);

    // Ensure host-side SiftPoints are synchronized from GPU. need xpos, ypos later for bundle adjustment
    cudaMemcpy(_sift1.h_data, _sift1.d_data, sizeof(SiftPoint) * _sift1.numPts, cudaMemcpyDeviceToHost);
    cudaMemcpy(_sift2.h_data, _sift2.d_data, sizeof(SiftPoint) * _sift2.numPts, cudaMemcpyDeviceToHost);


    // Track mutual matches
    for (int i = 0; i < _sift1.numPts; ++i) {
      int match_idx = _sift1.h_data[i].match;
      if (match_idx < 0 || match_idx >= _sift2.numPts) continue;

      // Confirm mutual match
      if (_sift2.h_data[match_idx].match == i) {
        DMatch m;
        m.queryIdx = i;
        m.trainIdx = match_idx;
        m.distance = _sift1.h_data[i].match_error;
        matches_info.matches.push_back(m);
      }
    }

    return matches_info;
  }


  void CompositeManager::build_match_pairs(std::vector<std::pair<Image*, Image*> > &_imagePairsToMatch, std::vector<RegInfo*> &_regs) {

    auto composites = parent->composites;
    double radSq = pow(0.9 * parent->scope_radius,2);

    for (auto cmp: composites) {

      //get image refs for all delaunay members
      std::vector<unsigned long> indexes;
      for (auto item: cmp->delaunayMembers) {
        indexes.push_back(item.second);
      }
      _regs = parent->get_reg_ref(indexes);
      auto imgs = parent->get_image_ref(indexes);

      //for each composite, loop thru delaunay members to make pairs within component.
      //Choose pairs that have a chance of matching
      for (int i = 0; i < _regs.size(); ++i) {
        for (int j = i + 1; j < _regs.size(); ++j) {
          if (cmp->componentMagLabel == Image::_2X) {
            if (pow(_regs[i]->absoluteCoords.x - _regs[j]->absoluteCoords.x,2) +
                pow(_regs[i]->absoluteCoords.y - _regs[j]->absoluteCoords.y,2) < radSq) {
              _imagePairsToMatch.push_back({imgs[i], imgs[j]});
            }
          }else {
            if (abs(_regs[i]->absoluteCoords.x - _regs[j]->absoluteCoords.x) < 0.9 * parent->image_width &&
                abs(_regs[i]->absoluteCoords.y - _regs[j]->absoluteCoords.y) < 0.9 * parent->image_height) {
              _imagePairsToMatch.push_back({imgs[i], imgs[j]});
                }
          }
        }
      }
      //TODO: create matching pairs across components. this works for initial tests.
    }
  }

}
