//
// Created by cooper maira on 11/11/25.
//

#include "pathCam.h"

namespace pathCam {
  MetricComposite::MetricComposite(StreamCam *parent, Size image_size, int _componentIndex) : Composite(
      parent, image_size, _componentIndex), ftg(new FeatureTrackGenerator) {
    frameDelay = 10;
    waitingFrames.resize(frameDelay, {nullptr, {}});
    //compositeImage = imagePyramid->level[0];

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

  /* This function is pretty confusing but the gist is that when a new frame comes in, we find what pyramid tiles it
   * could cover. For each of those tiles, if it improves the coverage type (no coverage to partial coverage, partial
   * coverage to full coverage) its data is copied to that tile immediately so that the user sees updates whenever they
   * cover a new area. Otherwise, we see if the image has less motion blur than whatever frame filled that tile. If
   * it does, we put it in a queue and wait 10 iterations before putting that frame's data into the tile. In that time,
   * new frames coming in have the chance to supplant frames in the queue. This process prevents tiles from being updated
   * over and over again by a series of consecutive frames and substantially lowers computational cost
   */
  void MetricComposite::update() {
    if (suspended) { return; }

    //place component in MR image
    if (imagePyramid->scale == 0 && !xcMatchInitiated) {
      if (staging.empty()) { return; }
      xcMatchInitiated = true;

      std::thread t([this, img = staging.front()->image]() {
        std::lock_guard lock(EstRoot_mutex);
        std::cout << "component " << componentIndex << " establishing scale on separate thread" << std::endl;
        establish_scale_at_root(img);
      });
      t.detach();
    }


    if (!staging.empty()) {
      auto ri = staging.front();
      auto img = ri->image;
      staging.pop();

      update_Bbox_no_composite({ri});

      waitingFrames[positionForNextWaitngFrame % frameDelay] = {img, {}};
      std::vector<Point2i> immediateProcessingTiles;


      //grab affected tiles with their category of coverage
      auto affectedPyramidTilesWithStatus = calculate_affected_tiles_with_status(
        Point2f(ri->absoluteCoords.x, ri->absoluteCoords.y));


      //calculate: for which of the affected tiles is this frame an improvement?
      for (auto &el: affectedPyramidTilesWithStatus) {
        auto pyrTileObj = imagePyramid->get_base_tile(el.first);

        //check if frame improves status of tile, if so process immediately
        if (pyrTileObj->status < el.second) {
          //if the tile is promoting to singleFrameCoverage, set owner and motionBlur from this frame
          if (el.second == TileObj::singleFrameCoverage) {
            pyrTileObj->owner = img;
            pyrTileObj->status = el.second;
            immediateProcessingTiles.push_back(el.first);
          }
          // pyrTileObj->status = el.second;
          // immediateProcessingTiles.push_back(el.first);
        }

        // check if frame is less blurry than current source for tile (pyrTileObj)
        else if (el.second == TileObj::singleFrameCoverage && image_improves_tile(pyrTileObj, img)) {
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
    //this is an erase-remove_if implementation with a lambda function inside that updates tileObj
    //if img should be owner, otherwise it removes the tile from the img's list

    for (auto &[img,tiles]: waitingFrames) {
      if (!img) { continue; }

      tiles.erase(
        std::remove_if(tiles.begin(),
                       tiles.end(),
                       [&](Point2i &tileIdx) {
                         auto tileObj = imagePyramid->get_base_tile(tileIdx);
                         if (tileObj->owner == img) {
                           return false;
                         }
                         if (image_improves_tile(tileObj, img)) {
                           tileObj->owner = img;
                           return false;
                         }
                         return true;
                       }), tiles.end()
      );

      if (tiles.empty()) {
        img->free_memory_RAW();
        img = nullptr;
      }
    }

    //once delay is met, process frame
    auto &[img,tiles] = waitingFrames[positionForNextWaitngFrame % frameDelay];
    if (positionForNextWaitngFrame > frameDelay && !tiles.empty()) {
      contributingImages.insert(img);
      needsAlignment = true;

      assert(img && img->get_Raw());

      process_tiles(img, tiles);

      debugTileCount2 += tiles.size();
      ++debugFrameCount;

      tiles.clear();
      img->free_memory_RAW();
      img = nullptr;
    }
  }

  void MetricComposite::align_and_rebuild() {
    auto start = std::chrono::high_resolution_clock::now();

    auto members = find_contributing_images();
    members.insert(root);

    for (auto &img: members) {
      img->keypointsImageSpace.resize(img->keypoints.size());
      for (int i = 0; i < img->keypoints.size(); ++i) {
        auto pt = img->keypoints[i].pt;
        img->keypointsImageSpace[i].pt.x = pt.x / parent->scale_factor;
        img->keypointsImageSpace[i].pt.y = pt.y / parent->scale_factor;
      }
    }

    auto matches = ftg->matches;
    std::vector<Match *> memberMatches;

    for (auto m: matches) {
      if (members.find(m->image_1) != members.end() && members.find(m->image_2) != members.end()) {
        memberMatches.push_back(m);
        for (int i = 0; i < m->good_matches.size(); ++i) {
          if (m->inliers[i]) {
            ftg->process_match(m->image_1->index, m->image_2->index, m->good_matches[i]);
          }
        }
      }
    }

    const std::vector memberImages(members.begin(), members.end());
    auto tracks = ftg->generateCurrentTracks(memberImages);
    parent->bai->run_bundle_adjustment(tracks, memberImages);

    float maxDev = 0, avgDev = 0;
    for (auto img: memberImages) {
      if (!img->regInfo->stayFixedDuringBundleAdjustment) {
        auto pv = parent->bai->optimizer->poseVertex(img->index);
        Point2i coords(-pv->t[0], -pv->t[1]);
        Point2i diff = img->regInfo->absoluteCoords - coords;

        img->regInfo->absoluteCoords = coords;

        //assert(abs(pv->t[2]/10000 - 1) < 0.03);
        if (abs(diff.x) > maxDev) {
          maxDev = abs(diff.x);
        }
        if (abs(diff.y) > maxDev) {
          maxDev = abs(diff.y);
        }
        avgDev += abs(diff.x) + abs(diff.y);
      }
    }
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::high_resolution_clock::now() - start).count();
    avgDev /= (2 * memberImages.size());
    int k = 0;

    rebuild(memberImages);
  }

  void MetricComposite::rebuild(std::vector<Image *> members) {
    auto start = std::chrono::high_resolution_clock::now();

    Mat blank(parent->tileSize, parent->tileSize,CV_8UC4, Scalar(0, 0, 0, 0));
    Mat mask(parent->tileSize, parent->tileSize,CV_8UC1, Scalar(255));
    for (auto &tileIdx: imagePyramid->liveTiles) {
      auto to = imagePyramid->get_base_tile(tileIdx);
      to->owner = nullptr;
      auto box = Rect_<float>(tileIdx.x * parent->tileSize, tileIdx.y * parent->tileSize, parent->tileSize,
                              parent->tileSize);
      std::vector v = {tileIdx};
      imagePyramid->insertTilesAtBase(blank, mask, box, v);
    }
    imagePyramid->liveTiles.clear();

    for (auto &img: members) {
      auto affectedTilesWithStatus = calculate_affected_tiles_with_status(img->regInfo->absoluteCoords);
      for (auto &p: affectedTilesWithStatus) {
        if (p.second == TileObj::singleFrameCoverage) {
          auto tileObj = imagePyramid->get_base_tile(p.first);
          if (image_improves_tile(tileObj, img)) {
            tileObj->owner = img;
            imagePyramid->liveTiles.insert(p.first);
          }
        }
      }
    }

    for (auto &img: members) {
      std::vector<Point2i> tiles;
      for (auto &tileIdx: imagePyramid->liveTiles) {
        auto tileObj = imagePyramid->get_base_tile(tileIdx);
        if (tileObj->owner == img) {
          tiles.push_back(tileIdx);
        }
      }
      process_tiles(img, tiles);
    }
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "cuda bundle adjustment solve time " << t3 << std::endl;

    /*
        //visualize differences
        int failCount = 0;
        parent->ORB_params.nfeatures = 10000;
        std::vector inds = {-1,0,1,0};
        for (auto &tileIdx : imagePyramid->liveTiles) {
          auto myTileObj = imagePyramid->get_base_tile(tileIdx);

          for (int i = 0; i < 4; ++i) {
            int x = inds[i];
            int y = inds[(i + 1) % 4];
            Point2i neighborPt = tileIdx + Point2i(x,y);

            if (imagePyramid->liveTiles.find(neighborPt) != imagePyramid->liveTiles.end()) {
              auto neighborObj = imagePyramid->get_base_tile(neighborPt);

              if (neighborObj->owner != myTileObj->owner) {
                //find this match
                bool found = false;
                Match* thisMatch = nullptr;
                for (auto m : ftg->matches) {
                  if ((m->image_1 == neighborObj->owner && m->image_2 == myTileObj->owner) ||
                    (m->image_2 == neighborObj->owner && m->image_1 == myTileObj->owner)) {
                    found = true;
                    thisMatch = m;
                    break;
                  }
                }
                if (!found) {

                  auto myImg = myTileObj->owner;
                  if (!myImg->fullKeyPoints) {
                    myImg->load_raw_from_disk();
                    cuda::GpuMat rawMat(imageSize, CV_8U, myImg->get_raw_cuda());
                    cuda::cvtColor(rawMat, threeChannelPrealGPU, COLOR_BayerBG2BGR, 0, parent->cvCompositeStream);
                    threeChannelPrealGPU.convertTo(convertHoldingGPU, CV_32F, parent->cvCompositeStream);
                    cuda::divide(convertHoldingGPU, ffGPU,
                       convertHoldingGPU, 1, CV_32F, parent->cvCompositeStream);
                    convertHoldingGPU.convertTo(threeChannelPrealGPU, CV_8UC3, parent->cvCompositeStream);
                    cvtColor(threeChannelPreallocated,myImg->reg_image,COLOR_BGR2GRAY);

                    auto detector = FeatureDetector(parent->feature_type, parent->use_FREAK);
                    detector.set_ORB_params(parent->ORB_params);
                    detector.detect_and_compute(myImg);
                    myImg->fullKeyPoints = true;
                  }

                  myImg = neighborObj->owner;
                  if (!myImg->fullKeyPoints) {
                    myImg->load_raw_from_disk();
                    cuda::GpuMat rawMat(imageSize, CV_8U, myImg->get_raw_cuda());
                    cuda::cvtColor(rawMat, threeChannelPrealGPU, COLOR_BayerBG2BGR, 0, parent->cvCompositeStream);
                    threeChannelPrealGPU.convertTo(convertHoldingGPU, CV_32F, parent->cvCompositeStream);
                    cuda::divide(convertHoldingGPU, ffGPU,
                       convertHoldingGPU, 1, CV_32F, parent->cvCompositeStream);
                    convertHoldingGPU.convertTo(threeChannelPrealGPU, CV_8UC3, parent->cvCompositeStream);
                    cvtColor(threeChannelPreallocated,myImg->reg_image,COLOR_BGR2GRAY);

                    auto detector = FeatureDetector(parent->feature_type, parent->use_FREAK);
                    detector.set_ORB_params();
                    detector.detect_and_compute(myImg);
                    myImg->fullKeyPoints = true;
                  }


                  auto matcher = DescriptorMatcher(parent->matcher_type);
                  Match *m = new Match(neighborObj->owner, myTileObj->owner);
                  matcher.match(m);

                  if (1 == MotionEstimator::findHomography(m, parent->estimator_type, 4)) {

                  } else {
                    MotionEstimator::findHomography(m, parent->estimator_type, 4);
                    ++failCount;
                  }
                }
              }
            }
          }
        }
        std::cout<<"fail count "<<failCount<<std::endl;
        */
  }


  void MetricComposite::process_tiles(Image *img, std::vector<Point2i> &tiles, bool forceFullImage) {
    assert(parent->unifiedMemory); //change this to a fix later
    if (!img->in_memory()) {
      img->load_raw_from_disk();
    }

    if (!img->subsequentMatchLaunched) {
      ++outstandingCMS_jobs;
      auto cms = new ComponentMatchSearch(parent, img);
      parent->jqSecondary->add_runnable(cms);
    }
    //lock mutex against component wide flatfielding
    update_mutex.lock();

    //put raw data into fourChannelPreallocated
    prepare_4CPA(img, tiles, forceFullImage);

    //calculate region of pyramid for data placement
    auto imageBox = cv::Rect_<float>(img->regInfo->absoluteCoords.x, img->regInfo->absoluteCoords.y, img->width,
                                     img->height);
    Mat mask = componentMagLabel == Image::_2X ? circleMask : rectMask;

    imagePyramid->insertTilesAtBase(fourChannelPreallocated, mask, imageBox, tiles);
    update_mutex.unlock();
  }


  std::vector<std::pair<Point2i, int> > MetricComposite::calculate_affected_tiles_with_status(Point2f AbC) const {
    std::vector<std::pair<Point2i, int> > results;

    if (componentMagLabel == Image::_2X) {
      Point2i center(imageSize.width / 2 + AbC.x, imageSize.height / 2 + AbC.y);

      int distance = parent->scope_radius + parent->tileSize;
      int distSq = pow(parent->scope_radius, 2);

      Point2f ulP(center.x - distance, center.y - distance);
      Point2f lrP(center.x + distance, center.y + distance);

      const auto ulTileInd = imagePyramid->level[0]->getIJ(ulP);
      const auto lrTileInd = imagePyramid->level[0]->getIJ(lrP);

      for (int x = ulTileInd.x; x <= lrTileInd.x; ++x) {
        for (int y = ulTileInd.y; y <= lrTileInd.y; ++y) {
          int inlierCorners = 0;
          for (int i = 0; i < 4; ++i) {
            int locX = parent->tileSize * (x + i % 2);
            int locY = parent->tileSize * (y + i / 2);

            int dx = locX - center.x;
            int dy = locY - center.y;

            if (dx * dx + dy * dy < distSq) {
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

  int MetricComposite::get_sqrd_center_distance_tile_to_img(Point2i _imgAbC, Point2i _tileCoord) {
    auto p1 = _imgAbC + Point2i(imageSize.width / 2, imageSize.height / 2);
    auto p2 = _tileCoord * parent->tileSize + Point2i(parent->tileSize / 2, parent->tileSize / 2);

    return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
  }

  std::vector<std::pair<Image *, Image *> > MetricComposite::calculate_member_overlaps(std::vector<Image *> images) {
    if (images.empty()) {
      images = std::vector(contributingImages.begin(), contributingImages.end());
    }
    std::vector<std::pair<Image *, Image *> > results;

    Point2i mDistance;
    int sqScopeRad = parent->scope_radius * parent->scope_radius * 0.7;
    for (int i = 0; i < images.size() - 1; ++i) {
      for (int j = i + 1; j < images.size(); ++j) {
        mDistance = images[i]->absoluteCoords - images[j]->absoluteCoords;

        if (componentMagLabel == Image::_2X) {
          if (pow(mDistance.x, 2) + pow(mDistance.y, 2) < sqScopeRad) {
            results.emplace_back(images[i], images[j]);
          }
        } else {
          if (abs(mDistance.x) < (1 - parent->crop_factor) * 0.9 * imageSize.width &&
              abs(mDistance.y) < (1 - parent->crop_factor) * 0.9 * imageSize.height) {
            results.emplace_back(images[i], images[j]);
          }
        }
      }
    }
    return results;
  }

  bool MetricComposite::image_improves_tile(std::shared_ptr<TileObj> _to, Image *_img) {
    //tile has no owner, candidate frame wins by default
    if (!_to->owner) {
      return true;
    }

    //frames have about the same blur, prioritize closeness to center of frame instead
    if (std::abs(_to->owner->motionBlur - _img->motionBlur) < 0.1f) {
      return get_sqrd_center_distance_tile_to_img(_to->owner->regInfo->absoluteCoords, _to->index) >
             get_sqrd_center_distance_tile_to_img(_img->regInfo->absoluteCoords, _to->index);
    }

    //amount of motion blur is significantly different, choose clearest image
    return _to->owner->motionBlur > _img->motionBlur;
  }

  std::unordered_set<Image *> MetricComposite::find_contributing_images() const {
    std::unordered_set<Image *> members;

    for (auto &tileIdx: imagePyramid->liveTiles) {
      auto to = imagePyramid->get_base_tile(tileIdx);
      members.insert(to->owner);
    }
    return members;
  }
}
