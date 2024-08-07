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
    auto start = std::chrono::high_resolution_clock::now();
    rebuildJobsOutstanding = 0;
    while (parent->microscopeInput || parent->diskCount > 0 || parent->regCount > 0 || parent->loaderCount > 0 ||
           parent->matchableCount > 0 || !parent->compositeQ_empty() || !parent->newComponentQ.empty()) {

      auto timeCheck = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timeCheck - start);
      if(duration.count() < 50){
        Poco::Thread::sleep(50 - duration.count());
      }
      start = timeCheck;

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
          parent->newComponentQ.pop();
          //perform_global_alignment();
          if(rebuildJobsOutstanding > 0) {
            rebuildJobsComplete.wait();
          }
          parent->add_new_component(std::get<0>(newComp), std::get<1>(newComp), std::get<2>(newComp));

        }else{
        Poco::Thread::sleep(100);
        }
      } else {

        std::vector<RegInfo> indexes = parent->get_Q_front();
        if (isNewComp) {
          if (indexes.front().index > std::get<0>(newComp)) {
            parent->newComponentQ.pop();
            //perform_global_alignment();
            if(rebuildJobsOutstanding > 0) {
              rebuildJobsComplete.wait();
            }
            parent->add_new_component(std::get<0>(newComp), std::get<1>(newComp), std::get<2>(newComp));
          }
        }
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
//    rebuildJobsComplete.wait();
    //save_components_to_disk();
    parent->compositing = false;
  }

  void CompositeManager::perform_global_alignment() {
    for (auto i: parent->composites) {
      i->perform_global_alignment(0, 0.2);
    }
  }

  void CompositeManager::save_components_to_disk() {
    for (auto i: parent->composites) {
      i->save_pyramid_as_image();
    }
  }

  void CompositeManager::decrement_rebuild_jobs_outstanding() {
    rebuildJobsOutstanding--;
    if(rebuildJobsOutstanding == 0){
      parent->update_observers();
      rebuildJobsComplete.set();
    }
  }

  RebuildRunnable::RebuildRunnable(pathCam::CompositeVoronoi *_composite, int _dtVertex, unsigned long _imageIndex) : dtVertex(_dtVertex),imageIndex(_imageIndex),composite(_composite),
  RunnableIntermediate(0,0){
    cm = composite->parent->cm;
  }

  void RebuildRunnable::run() {
    auto dt = composite->subdiv;
    auto image = composite->parent->get_image_ref(imageIndex);
    Mat polyMaskOutput = Mat::zeros(composite->image_size,CV_8U);
    std::vector<Point2i> face;
    std::vector<std::vector<Point2f>> facets;
    std::vector<Point2f> centers;
    dt.getVoronoiFacetList({dtVertex}, facets, centers);

    //shift and recast
    for (auto &ii: facets[0]) {//we have pulled only one face so facets has only 1 element
      ii.x -= centers[0].x;
      ii.x += composite->image_size.width / 2;
      ii.y -= centers[0].y;
      ii.y += composite->image_size.height / 2;
      face.push_back((Point2i) ii);
    }
    //build polygon mask for new point
    cv::fillConvexPoly(polyMaskOutput, face, cv::Scalar(255));
    image->load_raw_from_disk();
    Mat image_Mat = cv::Mat(composite->image_size, CV_8U, image->get_Raw(), Mat::AUTO_STEP);
    Mat3b threeChannelPreallocated;
    Mat4b fourChannelPreallocated;

    cvtColor(image_Mat, threeChannelPreallocated, COLOR_BayerBG2BGR);

    image->free_memory_RAW();

    if (image->label == Image::_2X) {//flat field correction if needed
      polyMaskOutput = polyMaskOutput.mul(composite->circleMask);
      auto center = Point2i(composite->image_size.width / 2, composite->image_size.height / 2);
      auto bb = Rect(center.x - composite->parent->scope_radius - 10, center.y - composite->parent->scope_radius - 10,
                     2 * composite->parent->scope_radius + 20, 2 * composite->parent->scope_radius + 20);
      cv::divide(threeChannelPreallocated(bb), composite->flat_field(bb), threeChannelPreallocated(bb), 1.0, CV_8U);
    }
    std::vector<Point2i> effectedTiles;
    composite->calculate_effected_tiles(face, effectedTiles, image->absoluteCoords);
    Mat tile_mask = Mat::zeros(composite->image_size.height,composite->image_size.width,CV_8U);
    //Vector containing the tiles where their center
    std::vector<Point2i> center_in_poly_mask;
    for (auto tile : effectedTiles){
      Point2i adjustedPoint;
      auto ul = Point2i((tile.x + 0.5) * composite->imagePyramid->tile_size,(tile.y + 0.5)* composite->imagePyramid->tile_size);
      adjustedPoint.x = ul.x - image->absoluteCoords.x;
      adjustedPoint.y = ul.y - image->absoluteCoords.y;
      if(adjustedPoint.x < 0 || adjustedPoint.x > image->width || adjustedPoint.y < 0 || adjustedPoint.y > image->height){
        continue;
      }
      uint8_t pixelVal;
      try {
        pixelVal = polyMaskOutput.at<uint8_t>(adjustedPoint);
      }
      catch(cv::Exception e){
        int k = 0;
      }
      if(pixelVal > 0) {
        auto tileSize = composite->imagePyramid->level[0]->getTileSize();
        auto tileBox = cv::Rect_<float>(tileSize * tile.x - image->absoluteCoords.x, tileSize * tile.y - image->absoluteCoords.y, tileSize, tileSize);
        tile_mask(tileBox).setTo(255);
        center_in_poly_mask.push_back(tile);
      }
    }
    std::vector<Mat> channels(2);
    channels[0] = threeChannelPreallocated; //3 channel
    channels[1] = tile_mask;           //alpha channel
    merge(channels, fourChannelPreallocated);
    Mat imageMat;
    fourChannelPreallocated.copyTo(imageMat);
    auto imageBox = cv::Rect_<float>(image->absoluteCoords.x, image->absoluteCoords.y, image->width, image->height);
    composite->imagePyramid->level[0]->inserTileAtBase(imageMat, tile_mask, imageBox, center_in_poly_mask);
    cm->decrement_rebuild_jobs_outstanding();
    int k = 0;
  }

}
