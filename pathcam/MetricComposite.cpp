//
// Created by cooper maira on 11/11/25.
//

#include "pathCam.h"

namespace pathCam {
  MetricComposite::MetricComposite(StreamCam *parent, Size image_size, int _componentIndex) : Composite(
    parent, image_size, _componentIndex) {
    frameDelay = 10;
    waitingFrames.resize(frameDelay, {nullptr, {}});
    compositeImage = imagePyramid->level[0];

    rectMask = Mat(image_size, CV_8U, cv::Scalar(255));
    rectMaskGPU = cuda::GpuMat(image_size,CV_8UC1, rectMask.data);
    threeChannelPreallocated = Mat(image_size, CV_8UC3);
    threeChannelPrealGPU = cuda::GpuMat(image_size, CV_8UC3, threeChannelPreallocated.data);
    fourChannelPreallocated = Mat::zeros(image_size, CV_8UC4);
    fourChannelPrealGPU = cuda::GpuMat(image_size,CV_8UC4, fourChannelPreallocated.data);

    circleMask = Mat::zeros(image_size, CV_8U);
    circle(circleMask, cv::Point(image_size.width / 2, image_size.height / 2), parent->scope_radius,
           Scalar(255),
           -1);
  }

  /* This function is pretty confusing but the jist is that when a new frame comes in, we find what pyramid tiles it
   * could cover. For each of those tiles, if it improves the coverage type (no coverage to partial coverage, partial
   * coverage to full coverage) its data is copied to that tile immediately so that the user sees updates whenever they
   * cover a new area. Otherwise, we see if the image has less motion blur than whatever frame filled that tile. If
   * it does, we put it in a queue and wait 10 iterations before putting that frame's data into the tile. In that time,
   * new frames coming in have the chance to suplant frames in the queue. This process prevents tiles from being updated
   * over and over again by a series of consequtive frames and substantially lowers computational cost
   */
  void MetricComposite::update() {
    //make sure component is placed in MR image
    if (imagePyramid->scale == 0) {
      assert(!staging.empty());
      establish_scale_at_root(staging.front()->image);
      assert(imagePyramid->scale > 0);
    }


    if (!staging.empty()) {
      auto ri = staging.front();
      auto img = ri->image;
      staging.pop();
      update_Bbox_no_composite({ri});
      mostRecentFrame = img;

      //grab affected tiles with their category of coverage
      auto affectedPyramidTilesWithStatus = calculate_affected_tiles_with_status(
        Point2f(ri->absoluteCoords.x, ri->absoluteCoords.y));
      std::vector<Point2i> immediateProcessingTiles;

      waitingFrames[positionForNextWaitngFrame % frameDelay] = {img, {}};

      //find what tiles raise status category of pyramid tiles
      for (auto &el: affectedPyramidTilesWithStatus) {
        auto &pyrTileObj = compositeImage->getTile(el.first.x, el.first.y);
        if (pyrTileObj.status < el.second) {
          //this frame improves status of this pyramid tile and should fill the tile without delay
          if (el.second == TileObj::singleFrameCoverage) {
            pyrTileObj.owner = img;
            pyrTileObj.motionBlur = img->motionBlur;
          }
          pyrTileObj.status = el.second;
          immediateProcessingTiles.push_back(el.first);
        } else if (el.second == TileObj::singleFrameCoverage && pyrTileObj.owner->motionBlur > img->motionBlur
                   || !pyrTileObj.owner) {
          waitingFrames[positionForNextWaitngFrame % frameDelay].second.push_back(el.first);
        }
      }
      ++positionForNextWaitngFrame;

      if (!immediateProcessingTiles.empty()) {
        process_tiles(img, immediateProcessingTiles);
        ++debugFrameCount;
        debugTileCount1 += immediateProcessingTiles.size();
      }

      if (imagePyramid->scale > 0) {
        float x = (imagePyramid->offset.x + img->absoluteCoords.x) * imagePyramid->scale;
        float y = (imagePyramid->offset.y + img->absoluteCoords.y) * imagePyramid->scale;
        float w = parent->image_width * imagePyramid->scale;
        float h = parent->image_height * imagePyramid->scale;
        bool showAsCircle = (componentMagLabel == Image::_2X);

        parent->update_last_frame(Rect_<float>(x, y, w, h), showAsCircle, componentIndex,
                                  Image::get_label(componentMagLabel));
      }
    } else {
      waitingFrames[positionForNextWaitngFrame % frameDelay] = {nullptr, {}};
      ++positionForNextWaitngFrame;
    }


    //process delayed frames, allowing them to blur correct if necessary
    //this is an erase-remove_if implementation with a lambda function inside that updates tileObj if img should be owner,
    //otherwise it removes the tile from the img's list
    for (auto &[img,tiles]: waitingFrames) {
      if (!img) { continue; }
      tiles.erase(
        std::remove_if(tiles.begin(),
                       tiles.end(),
                       [&](Point2i &tile) {
                         auto &tileObj = compositeImage->getTile(tile.x, tile.y);
                         if (tileObj.owner == img) {
                           return false;
                         }
                         if (tileObj.motionBlur > img->motionBlur) {
                           tileObj.motionBlur = img->motionBlur;
                           tileObj.owner = img;
                           return false;
                         }
                         return true;
                       }), tiles.end()
      );
      if (tiles.empty() && img != mostRecentFrame) {
        img->free_memory_RAW();
        img = nullptr;
      }
    }

    //once delay is met, process frame
    auto &[img,tiles] = waitingFrames[positionForNextWaitngFrame % frameDelay];
    if (positionForNextWaitngFrame > frameDelay && !tiles.empty()) {
      contributingImages.push_back(img);
      needsAlignment = true;

      assert(img && img->get_Raw());
      process_tiles(img, tiles);

      debugTileCount2 += tiles.size();
      ++debugFrameCount;

      tiles.clear();
      if (img != mostRecentFrame) {
        img->free_memory_RAW();
      }
      img = nullptr;
    }
  }

  void MetricComposite::align_and_rebuild() {
    auto start = std::chrono::high_resolution_clock::now();
    auto members = find_contributing_images();
    auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto overlaps = calculate_member_overlaps(members);
    auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto matcher = pathCam::DescriptorMatcher(parent->matcher_type);
    auto motion_est =  pathCam::MotionEstimator();
    int res1 = 0,good = 0;
    for (auto &[img1,img2] : overlaps) {
      Match m(img1,img2);
      matcher.match(&m,0);
      int result = motion_est.findHomography(&m, parent->estimator_type, 10, 0);
      if (result == 1) {
        ++res1;
        if (abs(m.scale - 1) < .05) {
          ++good;
        }else {
          std::cout<<abs(m.scale - 1)<<std::endl;
        }
      }

    }
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    int k = 0;

  }

  void MetricComposite::process_tiles(Image *img, std::vector<Point2i> &tiles) {
    assert(parent->unifiedMemory); //change this to a fix later
    if (!img->in_memory()) {
      img->load_raw_from_disk();
    }

    //put raw data into fourChannelPreallocated
    prepare_4CPA(img, tiles);

    //calculate region of pyramid for data placement
    auto imageBox = cv::Rect_<float>(img->absoluteCoords.x, img->absoluteCoords.y, img->width, img->height);
    Mat mask = componentMagLabel == Image::_2X ? circleMask : rectMask;

    imagePyramid->insertTilesAtBase(fourChannelPreallocated, mask, imageBox, tiles);
  }


  std::vector<std::pair<Point2i, int> > MetricComposite::calculate_affected_tiles_with_status(Point2f AbC) {
    std::vector<std::pair<Point2i, int> > results;

    if (componentMagLabel == Image::_2X) {
      Point2i center(imageSize.width / 2 + AbC.x, imageSize.height / 2 + AbC.y);

      int distance = parent->scope_radius + parent->tileSize;
      int distSq = pow(parent->scope_radius, 2);

      Point2f ulP(center.x - distance, center.y - distance);
      Point2f lrP(center.x + distance, center.y + distance);

      auto ulTileInd = imagePyramid->level[0]->getIJ(ulP);
      auto lrTileInd = imagePyramid->level[0]->getIJ(lrP);

      int inlierCorners;
      for (int x = ulTileInd.x; x <= lrTileInd.x; ++x) {
        for (int y = ulTileInd.y; y <= lrTileInd.y; ++y) {
          inlierCorners = 0;
          for (int i = 0; i < 4; ++i) {
            int locX = parent->tileSize * (x + i % 2);
            int locy = parent->tileSize * (y + i / 2);
            if (pow(locX - center.x, 2) + pow(locy - center.y, 2) < distSq) {
              ++inlierCorners;
            }
          }
          if (inlierCorners == 4) {
            results.emplace_back(Point2i(x, y), TileObj::singleFrameCoverage);
          } else if (inlierCorners > 0) {
            results.emplace_back(Point2i(x, y), TileObj::partialCoverage);
          }
        }
      }
    } else {
      auto lrP = AbC + Point2f(float(imageSize.width), float(imageSize.height));

      auto ulTileInd = imagePyramid->level[0]->getIJ(AbC);
      auto lrTileInd = imagePyramid->level[0]->getIJ(lrP);

      for (int x = ulTileInd.x; x <= lrTileInd.x; ++x) {
        for (int y = ulTileInd.y; y <= lrTileInd.y; ++y) {
          if (parent->tileSize * x >= AbC.x && parent->tileSize * x + parent->tileSize <= AbC.x + imageSize.
              width &&
              parent->tileSize * y >= AbC.y && parent->tileSize * y + parent->tileSize <= AbC.y + imageSize.
              height) {
            results.emplace_back(Point2i(x, y), TileObj::singleFrameCoverage);
          } else {
            results.emplace_back(Point2i(x, y), TileObj::partialCoverage);
          }
        }
      }
    }
    return results;
  }

  std::vector<std::pair<Image *, Image *> > MetricComposite::calculate_member_overlaps(std::vector<Image *> images) {
    if (images.empty()) {
      images = contributingImages;
    }
    std::vector<std::pair<Image *, Image *> > results;

    Point2i mDistance;
    int sqScopeRad = parent->scope_radius * parent->scope_radius * 0.7;
    for (int i = 0; i < images.size() - 1; ++i) {
      for (int j = i + 1; j < images.size(); ++j) {

        mDistance = images[i]->absoluteCoords - images[j]->absoluteCoords;

        if (componentMagLabel == Image::_2X) {
          if (pow(mDistance.x,2) + pow(mDistance.y,2) < sqScopeRad) {
            results.emplace_back(images[i],images[j]);
          }

        }else{
          if (abs(mDistance.x) < (1 - parent->crop_factor) * 0.9 * imageSize.width &&
            abs(mDistance.y) < (1 - parent->crop_factor) * 0.9 * imageSize.height) {
            results.emplace_back(images[i],images[j]);
          }
        }
      }
    }
    return results;
  }

  std::vector<Image *> MetricComposite::find_contributing_images() const {
    std::vector<Image*> members;

    auto ulInd = parent->composites[0]->imagePyramid->level[0]->getIJ(parent->composites[0]->root_offset);
    auto lrInd = parent->composites[0]->imagePyramid->level[0]->getIJ(parent->composites[0]->max_offset);
    bool found;
    for (auto &img : contributingImages) {
      found = false;
      for (int x = ulInd.x; x<=lrInd.x; ++x) {
        for (int y = ulInd.y; y <= lrInd.y; ++y) {

          if (auto tileObj = compositeImage->tiles(x,y); tileObj && tileObj->owner == img) {
            members.push_back(img);
            found = true;
            break;
          }
        }
        if (found){break;}
      }
    }
    return members;
  }

}
