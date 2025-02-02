//
//  CompositeManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//
// This class listens for registered frames, sorts them by component membership, then passes them to the component object for compositing.

#include "pathCam.h"

namespace pathCam {

  CompositeManager::CompositeManager(StreamCam *parent) : parent(parent), successful(false),rebuildJobsOutstanding(true) {};

  void CompositeManager::run() {
    unsigned long duration = 0;


    rebuildJobsOutstanding = 0;
    while (parent->microscopeInput || parent->diskCount > 0 || parent->regCount > 0 || parent->loaderCount > 0 ||
           parent->matchableCount > 0 || !parent->compositeQ_empty() || !parent->newComponentQ.empty()) {

//      if(duration.count() < 100){
//        Poco::Thread::sleep(100 - duration.count());
//      }
      //start = timeCheck;

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
          //perform_global_alignment();
          if(rebuildJobsOutstanding > 0) {
            rebuildJobsComplete.wait();
          }
          parent->add_new_component(std::get<0>(newComp), std::get<1>(newComp), std::get<2>(newComp));
          parent->newComponentQ.pop();

        }else{

        check_render_info();
        Poco::Thread::sleep(100);
        }

      } else {

        auto indexes = parent->get_Q_front();
        if (isNewComp) {
          if (indexes.front()->index > std::get<0>(newComp)) {
            //perform_global_alignment();
            if(rebuildJobsOutstanding > 0) {
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
        std::vector<RegInfo*> new_info;

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
    }


    push_remaining_tiles_for_inference();

    //perform_global_alignment();
    //rebuildJobsComplete.wait();
    //save_components_to_disk();
    std::cout<<"CM duration: "+std::to_string(duration)<<std::endl;
    parent->compositing = false;
    parent->inferenceWait.set();
  }

  void CompositeManager::push_remaining_tiles_for_inference() {
    for (auto i:parent->composites) {
      for (auto tilePoint: i->queuedTiles) {
        parent->push_tile_embed_Q({tilePoint, i->componentIndex});
      }
    }
  }


  void CompositeManager::perform_global_alignment() {
    for (auto i: parent->composites) {
      i->perform_global_alignment(0, 0.2);
    }
  }

  void CompositeManager::save_components_to_disk() {
    for (auto i: parent->composites) {
      //i->imagePyramid->level[0]->saveBaseTilesToDisk();

      i->save_pyramid_as_image("/Users/coopermaira/desktop/tiledImage.png");

    }
  }

  void CompositeManager::align_new_comp() {
    parent->newComponentQ.pop();
    perform_global_alignment();
    if(rebuildJobsOutstanding > 0) {
      rebuildJobsComplete.wait();
    }
  }

  void CompositeManager::check_render_info() {
    for (auto cmp : parent->composites){
      cmp->check_set_render_info();
    }
  }

  void CompositeManager::decrement_rebuild_jobs_outstanding() {
    rebuildJobsOutstanding--;
    parent->notify_observers();
    if(rebuildJobsOutstanding == 0){
      rebuildJobsComplete.set();
    }
  }

  RebuildRunnable::RebuildRunnable(pathCam::CompositeVoronoi *_composite, int _dtVertex, unsigned long _imageIndex, std::vector<Point2i> _rebuildTiles, Mat _polyMaskOutput) :
  dtVertex(_dtVertex),
  imageIndex(_imageIndex),
  composite(_composite),
  rebuildTiles(_rebuildTiles),
  polyMaskOutput(_polyMaskOutput),
  RunnableIntermediate(0,0){
    cm = composite->parent->cm;
  }

  void RebuildRunnable::run() {
    auto image = composite->parent->get_image_ref(imageIndex);
    image->load_raw_from_disk();
    //image->build_whitebalance_Mat(composite->parent);

    std::vector<Mat> channels(2);
    Mat3b threeChannelPreallocated;
    Mat4b fourChannelPreallocated;

    if(image->readyImage.data){
      channels[0] = image->readyImage;
    }else{

    Mat image_Mat = cv::Mat(composite->image_size, CV_8U, image->get_Raw(), Mat::AUTO_STEP);


    cvtColor(image_Mat, threeChannelPreallocated, COLOR_BayerBG2BGR);




    channels[0] = threeChannelPreallocated; //3 channel
    }

    image->free_memory_RAW();

    if (image->label == Image::_2X) {//flat field correction if needed
      if(!image->readyImage.data){
        auto center = Point2i(composite->image_size.width / 2, composite->image_size.height / 2);
        auto bb = Rect(center.x - composite->parent->scope_radius - 10, center.y - composite->parent->scope_radius - 10,
    2 * composite->parent->scope_radius + 20, 2 * composite->parent->scope_radius + 20);
        cv::divide(threeChannelPreallocated(bb), composite->flat_field(bb), threeChannelPreallocated(bb), 1.0, CV_8U);
      }
      channels[1] = composite->circleMask * 255;           //alpha channel

    }else if(image->label == Image::_4X){
      if(!image->readyImage.data){
        divide(threeChannelPreallocated, composite->parent->flat_field4X, threeChannelPreallocated, 1, CV_8U);
      }
      channels[1] = Mat(image->height,image->width,CV_8U,Scalar(255));

    }else{
      //divide(threeChannelPreallocated, composite->parent->flat_field4X, threeChannelPreallocated, 1, CV_8U);
      channels[1] = Mat(image->height,image->width,CV_8U,Scalar(255));

    }

    merge(channels, fourChannelPreallocated);

    auto imageBox = cv::Rect_<float>(image->absoluteCoords.x, image->absoluteCoords.y, image->width, image->height);
    composite->imagePyramid->level[0]->insertMatAtBase(fourChannelPreallocated,imageBox,rebuildTiles);

    if(image->readyImage.data){
      image->readyImage.release();
    }

    cm->decrement_rebuild_jobs_outstanding();
  }

}
