//
// Created by cooper maira on 11/11/25.
//

#include "pathCam.h"

namespace pathCam {
  inline SiftData collect_SiftData(const std::vector<std::pair<Image *, std::vector<const SiftPoint *> > > &_inVec,
                                   int _subsample) {
    SiftData siftData;
    int totalPoints = 0;
    for (auto &[img,vec]: _inVec) {
      totalPoints += vec.size();
    }
    totalPoints /= _subsample;

    InitSiftData(siftData, totalPoints, true, true);


    int startPos = 0;
    for (auto &[img,vec]: _inVec) {
      Point2f shift((img->width - img->parent->siftWindow) / 2 + img->regInfo->absoluteCoords.x,
                    (img->height - img->parent->siftWindow) / 2 + img->regInfo->absoluteCoords.y);
      for (int i = 0; i < static_cast<int>(vec.size()) / _subsample; ++i) {
        const SiftPoint *src = vec[i * _subsample]; // take every 10th input
        siftData.h_data[startPos + i] = *src; // pack output contiguously
        siftData.h_data[startPos + i].xpos += shift.x;
        siftData.h_data[startPos + i].ypos += shift.y;
      }
      startPos += static_cast<int>(vec.size()) / _subsample;
    }

    cudaMemcpy(siftData.d_data, siftData.h_data,
               static_cast<size_t>(startPos) * sizeof(SiftPoint),
               cudaMemcpyHostToDevice);

    siftData.numPts = startPos;
    return siftData;
  }

  inline void querySiftRect_sortedByX(
    const SiftData &sd,
    float xmin, float xmax,
    float ymin, float ymax,
    std::vector<const SiftPoint *> &out) {
    assert(sd.h_data);
    assert(sd.numPts >= 0 && sd.numPts <= sd.maxPts);
    if (sd.numPts == 0 || xmin > xmax || ymin > ymax) {
      out.clear();
      return;
    }

    const SiftPoint *begin = sd.h_data;
    const SiftPoint *end = sd.h_data + sd.numPts;

    // First point with xpos >= xmin
    auto lo = std::lower_bound(begin, end, xmin,
                               [](const SiftPoint &p, float v) {
                                 return p.xpos < v;
                               });

    // First point with xpos > xmax
    auto hi = std::upper_bound(lo, end, xmax,
                               [](float v, const SiftPoint &p) {
                                 return v < p.xpos;
                               });

    out.clear();
    out.reserve(static_cast<size_t>(hi - lo)); // upper bound on hits

    for (auto it = lo; it != hi; ++it) {
      if (it->ypos >= ymin && it->ypos <= ymax) {
        out.push_back(it); // pointers into sd.h_data
      }
    }
  }

  MetricComposite::MetricComposite(StreamCam *parent, Size image_size, int _componentIndex) : Composite(
      parent, image_size, _componentIndex), ftg(new FeatureTrackGenerator) {
    frameDelay = 10;
    waitingFrames.resize(frameDelay, {nullptr, {}});
    //compositeImage = imagePyramid->level[0];

    cudaMallocManaged(&rectMaskBuf, imageSize.area());
    cudaMemset(rectMaskBuf, 255, imageSize.area());
    rectMask = Mat(image_size, CV_8UC1, rectMaskBuf);
    rectMaskGPU = cuda::GpuMat(imageSize,CV_8UC1, rectMaskBuf);

    cudaMallocManaged(&threeChnBuf, 3 * imageSize.area());
    threeChannelPreallocated = Mat(imageSize,CV_8UC3, threeChnBuf);
    threeChannelPrealGPU = cuda::GpuMat(imageSize,CV_8UC3, threeChnBuf);

    cudaMallocManaged(&fourChnBuf, 4 * imageSize.area());
    fourChannelPreallocated = Mat(imageSize,CV_8UC4, fourChnBuf);
    fourChannelPrealGPU = cuda::GpuMat(imageSize,CV_8UC4, fourChnBuf);

    if (parent->circleMask.empty()) {
      circleMask = Mat::zeros(image_size, CV_8U);
      circle(circleMask, Point(image_size.width / 2, image_size.height / 2), parent->scope_radius,
             Scalar(255),
             -1);
    }
  }

  MetricComposite::~MetricComposite() {
    cudaFree(threeChnBuf);
    cudaFree(fourChnBuf);
    cudaFree(rectMaskBuf);
    for (int i = 0; i < ftg->storedMatches.size(); ++i) {
      if (ftg->storedMatches[i]) {
        delete ftg->storedMatches[i];
      }
      ftg->storedMatches.clear();
    }
  }

  /* This function is pretty confusing but the gist is that when a new frame comes in we find what pyramid tiles it
   * could cover. For each of those tiles, if the frame improves the coverage type (no coverage to partial coverage, partial
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

    // PROCESS NEW FRAMES BEGIN
    if (!staging.empty()) {
      auto ri = staging.front();
      auto img = ri->image;
      memberFrames.push_back(img);
      assert(img->regInfo);

      staging.pop();


      update_Bbox_no_composite({ri});

      waitingFrames[positionForNextWaitngFrame % frameDelay] = {img, {}};
      std::vector<Point2i> immediateProcessingTiles;


      //grab affected tiles with their category of coverage
      auto affectedPyramidTilesWithStatus = calculate_affected_tiles_with_status(
        Point2f(ri->absoluteCoords));


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
          // //this if you want edge tiles. not very functional
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

      //if (imagePyramid->scale > 0) {
        float x = (imagePyramid->offset.x + img->absoluteCoords.x) * imagePyramid->scale;
        float y = (imagePyramid->offset.y + img->absoluteCoords.y) * imagePyramid->scale;
        float w = parent->image_width * imagePyramid->scale;
        float h = parent->image_height * imagePyramid->scale;
        bool showAsCircle = (componentMagLabel == Image::_2X);

        parent->update_last_frame(Rect_<float>(x, y, w, h), showAsCircle, componentIndex,
                                  Image::get_label(componentMagLabel));
      //}

    } else {
      waitingFrames[positionForNextWaitngFrame % frameDelay] = {nullptr, {}};
      ++positionForNextWaitngFrame;
    }
    // PROCESS NEW FRAMES END


    //PROCESS OLD FRAMES BEGIN

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

      //assert(img && img->get_Raw());
      process_tiles(img, tiles);

      debugTileCount2 += tiles.size();
      ++debugFrameCount;


      tiles.clear();
      img->free_memory_RAW();
      img = nullptr;
    }
    // PROCESS OLD FRAMES END
  }

  void MetricComposite::align_and_rebuild() {
    auto start = std::chrono::high_resolution_clock::now();

    ig = new ImageGraph();
    bai = new BundleAdjustmentIntegrator();

    std::unordered_set<Image *> members = find_contributing_images();
    members.insert(root);


    auto matches = ftg->storedMatches; //matches are just stored here before being processed all at once.
    std::vector<Match *> memberMatches;


    for (auto m: matches) {
      ig->addEdge(m->image_1->index, m->image_2->index, ImageGraph::EdgeKind::ORB);
    }
    for (int ii = 0; ii < extraMatches.size(); ++ii) {
      auto [img1,img2,kp1,kp2] = extraMatches[ii];
      ig->addEdge(img1->index, img2->index, ImageGraph::EdgeKind::SIFT);
    }

    for (auto img: members) {
      ig->setMember(img->index, true);
    }

    auto graphConnectivityResult = ig->computeMinPromotionsToConnectMembersPreferORB();
    if (!graphConnectivityResult.success){return;}

    if (!graphConnectivityResult.promoted_nodes.empty()) {
      std::cout << "Component " << componentIndex << " promoting additional " << graphConnectivityResult.promoted_nodes.size() <<
          " frames in BA" << std::endl;
      //important to check if empty or get_image_ref returns every image known to StreamCam
      for (auto img: parent->get_image_ref(graphConnectivityResult.promoted_nodes)) {
        ig->setMember(img->index,true);
        members.insert(img);
      }
    }


    for (auto m: matches) {
      if (m->image_1->index == 197 || m->image_2->index == 196) {
        int k = 0;
      }
      if (members.find(m->image_1) != members.end() && members.find(m->image_2) != members.end()) {
        // members.insert(m->image_1);
        // members.insert(m->image_2);

        for (int i = 0; i < m->good_matches.size(); ++i) {
          if (m->inliers[i]) {
            ftg->process_match(m->image_1->index, m->image_2->index, m->good_matches[i]);
          }
        }
      }
    }
    for (auto &img: members) {
      img->keypointsImageSpace.resize(img->keypoints.size());
      for (int i = 0; i < img->keypoints.size(); ++i) {
        img->keypointsImageSpace[i].pt = img->keypoints[i].pt / parent->scale_factor;
      }
    }

    if (graphConnectivityResult.used_sift) {
      std::cout << "Component " << componentIndex << " using sift in BA" << std::endl;
      return;
      for (auto [img1,img2,kp1,kp2]: extraMatches) {
        if (members.find(img1) != members.end() && members.find(img2) != members.end()) {
          assert(kp1.size() == kp2.size());
          img1->keypointsImageSpace.reserve(img1->keypointsImageSpace.size() + kp1.size());
          img2->keypointsImageSpace.reserve(img2->keypointsImageSpace.size() + kp2.size());

          for (int i = 0; i < min(int(kp1.size()), 300); ++i) {
            DMatch dm;
            dm.queryIdx = img1->keypointsImageSpace.size();
            dm.trainIdx = img2->keypointsImageSpace.size();

            img1->keypointsImageSpace.push_back(kp1[i]);
            img2->keypointsImageSpace.push_back(kp2[i]);

            ftg->process_match(img1->index, img2->index, dm);
          }
        }
      }
    }

    //GENERATE TRACKS AND RUN
    std::vector memberImages(members.begin(), members.end());
    auto tracks = ftg->generateCurrentTracks(memberImages);
    bai->run_bundle_adjustment(tracks, memberImages);
    // for (auto &[id,pv]: bai->poseVertices) {
    //   std::cout << id << " " << pv->edges.size() << std::endl;
    // }

    //decide if youre going to accept the bundle adjustment answer, at least in part.
    auto start1 = std::chrono::high_resolution_clock::now();

    auto rootID = root->index;
    std::unordered_map<ImageGraph::ImgId, ImageGraph::ImgId> parentGraph;
    bool ok = ig->computePreferredParentsToRootAfterPromotions(rootID, parentGraph);
    //ok will always be true if we've gotten this far in the function

    std::unordered_map<ImageGraph::ImgId,int> idDistanceToRoot;
    idDistanceToRoot.reserve(memberImages.size() * 2);
    std::sort(memberImages.begin(), memberImages.end(),
    [&](Image* a, Image* b) {
        int da = ImageGraph::hopDistanceToRoot(a->index, rootID, parentGraph, idDistanceToRoot);
        int db = ImageGraph::hopDistanceToRoot(b->index, rootID, parentGraph, idDistanceToRoot);

        if (da != db) return da < db;
        return a->index < b->index; // tie-breaker: lower id first (or keep stable_sort if you prefer)
    });

    for (auto img: memberImages) {
      if (!img->regInfo->stayFixedDuringBundleAdjustment) {
        auto myVertex = bai->optimizer->poseVertex(img->index);
        auto theirID = parentGraph[img->index];
        auto theirVertex = bai->optimizer->poseVertex(theirID);

        bool found = false;
        Match* ourMatch;
        Image* them;
        int multiplier;
        for (auto m : img->matches) {
          if (m->image_1->index == theirID || m->image_2->index == theirID) {
            them = m->image_1->index == theirID ? m->image_1 : m->image_2;
            multiplier = m->image_1->index == theirID ? -1 : 1;
            ourMatch = m;
            found = true;
            break;
          }
        }
        if (!found) {
          throw std::runtime_error("failed to find match while testing BA solution");
        }

        Point2i ourDistance(multiplier * ourMatch->t_x,multiplier * ourMatch->t_y);
        Point2i myCoords(-myVertex->t[0], -myVertex->t[1]);
        Point2i theirCoords(-theirVertex->t[0], -theirVertex->t[1]);
        auto vDist = myCoords - theirCoords;

        Point2i acceptedDistance = vDist;
        if ((ourDistance.x - vDist.x) * (ourDistance.x - vDist.x) + (ourDistance.y - vDist.y) * (ourDistance.y - vDist.y) > 900) {
          acceptedDistance = ourDistance;
        }

        img->regInfo->absoluteCoords = them->regInfo->absoluteCoords + acceptedDistance;
      }
    }
    auto t4 = std::chrono::duration_cast<std::chrono::milliseconds>
           (std::chrono::high_resolution_clock::now() - start1).count();
    rebuild(memberImages);


    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "total align time comp " << componentIndex << ": " << t3 << std::endl;
  }

  void MetricComposite::rebuild(std::vector<Image *> members) {
    for (auto &tileIdx: imagePyramid->liveTiles) {
      auto to = imagePyramid->get_base_tile(tileIdx);
      to->owner = nullptr;
    }
    auto liveTilesCopy = imagePyramid->liveTiles;
    imagePyramid->liveTiles.clear();

    for (auto &img: members) {
      auto affectedTilesWithStatus = calculate_affected_tiles_with_status(img->regInfo->absoluteCoords);
      for (auto &p: affectedTilesWithStatus) {
        if (p.second == TileObj::singleFrameCoverage) {
          auto ul = p.first * parent->tileSize;
          assert(img->regInfo->absoluteCoords.x <= ul.x && img->regInfo->absoluteCoords.y <= ul.y &&
            img->regInfo->absoluteCoords.x + imageSize.width >= ul.x + parent->tileSize &&
            img->regInfo->absoluteCoords.y + imageSize.height >= ul.y + parent->tileSize);
          auto tileObj = imagePyramid->get_base_tile(p.first);
          if (image_improves_tile(tileObj, img)) {
            tileObj->owner = img;
            imagePyramid->liveTiles.insert(p.first);
          }
        }
      }
    }

    std::vector<std::pair<Image *, std::vector<const SiftPoint *> > > componentFeatures;
    Point2i siftWindowCorner((imageSize.width - parent->siftWindow) / 2, (imageSize.height - parent->siftWindow) / 2);

    for (auto &img: members) {
      std::vector<Point2i> tileIndexes;
      for (auto &tileIdx: imagePyramid->liveTiles) {
        auto tileObj = imagePyramid->get_base_tile(tileIdx);
        if (tileObj->owner == img) {
          tileIndexes.push_back(tileIdx);
          auto ul = tileIdx * parent->tileSize;
          assert(img->regInfo->absoluteCoords.x <= ul.x && img->regInfo->absoluteCoords.y <= ul.y &&
            img->regInfo->absoluteCoords.x + imageSize.width >= ul.x + parent->tileSize &&
            img->regInfo->absoluteCoords.y + imageSize.height >= ul.y + parent->tileSize);
        }
      }
      process_tiles(img, tileIndexes,false);


      // for (auto &tileInd: tileIndexes) {
      //   auto lowerQueryPt = tileInd * parent->tileSize - img->regInfo->absoluteCoords - siftWindowCorner;
      //   auto upperQueryPt = lowerQueryPt + Point2i(parent->tileSize, parent->tileSize);
      //   assert(img->siftInitialized);
      //   std::vector<const SiftPoint *> fts;
      //   querySiftRect_sortedByX(img->siftData,
      //                           lowerQueryPt.x, upperQueryPt.x, lowerQueryPt.y, upperQueryPt.y, fts);
      //
      //   if (!fts.empty()) {
      //     componentFeatures.emplace_back(img, fts);
      //   }
      // }
    }


    auto tBox = Rect(0, 0, parent->tileSize, parent->tileSize);
    for (auto &tileIdx: liveTilesCopy) {
      auto tileObj = imagePyramid->get_base_tile(tileIdx);
      if (!tileObj->owner) {
        //kill tile
        tileObj->image.setTo(Scalar(0, 0, 0, 0));
        Rect tileReg(tileIdx * parent->tileSize, Size(parent->tileSize, parent->tileSize));
        imagePyramid->level[0]->tileUpwards(tileIdx, tileReg, tileObj, tBox);
        tileObj.reset();
      }
    }

    compSiftData = collect_SiftData(componentFeatures, 5);
  }

  void MetricComposite::search_and_absorb_other_components() {
    for (auto &m: componentJoinMatches) {
      Image *myMember, *theirMember;
      if (m->image_1->regInfo->component_membership == componentIndex) {
        myMember = m->image_1;
        theirMember = m->image_2;
      } else {
        myMember = m->image_2;
        theirMember = m->image_1;
      }
      if (theirMember->regInfo->component_membership != componentIndex) {
        //absorb it, which means change all the members' registrations
      }
    }
  }


  void MetricComposite::process_tiles(Image *img, std::vector<Point2i> &tiles, bool alertDoubleLoad, const bool forceFullImage) {
    assert(parent->unifiedMemory); //change this to a fix later

    img->load_raw_from_disk(alertDoubleLoad);

    //not a mistake. we have two process that need the raw, second call increments the counter

    if (!img->subsequentMatchLaunched) {
      img->load_raw_from_disk(alertDoubleLoad);
      img->subsequentMatchLaunched = true;
      ++outstandingCMS_jobs;
      auto cms = new ComponentMatchSearch(parent, img, this);
      parent->jqSecondary->add_runnable(cms);
    }
    //lock mutex against component wide flatfielding
    update_mutex.lock();

    //put raw data into fourChannelPreallocated
    prepare_4CPA(img, tiles, forceFullImage);
    img->free_memory_RAW();

    //calculate region of pyramid for data placement
    auto imageBox = cv::Rect_<float>(img->regInfo->absoluteCoords.x, img->regInfo->absoluteCoords.y, img->width,
                                     img->height);
    Mat mask = componentMagLabel == Image::_2X ? circleMask : rectMask;

    imagePyramid->insertTilesAtBase(fourChannelPreallocated, mask, imageBox, tiles);
    update_mutex.unlock();
  }


  std::vector<std::pair<Point2i, int> > MetricComposite::calculate_affected_tiles_with_status(const Point2f AbC) const {
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

  int MetricComposite::get_sqrd_center_distance_tile_to_img(Point2i _imgAbC, Point2i _tileCoord) const {
    auto p1 = _imgAbC + Point2i(imageSize.width / 2, imageSize.height / 2);
    auto p2 = _tileCoord * parent->tileSize + Point2i(parent->tileSize / 2, parent->tileSize / 2);

    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
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

  bool MetricComposite::image_improves_tile(const std::shared_ptr<TileObj> &_to, const Image *_img) const {
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
