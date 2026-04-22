//
// Created by cooper maira on 11/11/25.
//

#include "pathCam.h"

namespace pathCam {
  MetricComposite::MetricComposite(StreamCam *parent, Size image_size, int _componentIndex) : Composite(
    parent, image_size, _componentIndex) {
    waitingFrames.resize(frameDelay, {nullptr, {}});
  }

  MetricComposite::~MetricComposite() {
    // if (!suspended && successfullyAligned) {
    //   FreeSiftData(compSiftData);
    // }
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
      xcInProgress = true;
      xcMatchInitiated = true;

      // std::thread t([this, img = staging.front()->image]() {
      std::lock_guard lock(EstRoot_mutex);
      // std::cout << "component " << componentIndex << " establishing scale" << std::endl;
      auto start = std::chrono::high_resolution_clock::now();
      establish_scale_at_root_cpu(staging.front()->image);
      std::cout << "XC REGISTERED COMPONENT " << componentIndex << " in " << std::chrono::duration_cast<
        std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count() << std::endl;
      xcInProgress = false;
      // });
      // t.detach();
    }

    Poco::FastMutex::ScopedLock lock(update_mutex);

    // PROCESS NEW FRAMES BEGIN
    if (!staging.empty()) {
      auto ri = staging.front();
      auto img = ri->image;
      memberFrames.push_back(img);
      assert(img->regInfo);
      ri->compPopped = true;


      staging.pop();
      ++frameCount;
      maxIndex = max(maxIndex, img->index);

      launch_component_match_search_with_XC(img);

      if (img->labelObserved) {
        ++observedLabels[img->label];
        int maxObservations = 0;
        unsigned winner = componentMagLabel;
        for (auto &[label,observationCount]: observedLabels) {
          if (observationCount > maxObservations) {
            maxObservations = observationCount;
            winner = label;
          }
        }
        if (winner != componentMagLabel) {
          componentMagLabel = winner;

          Poco::FastMutex::ScopedLock lock(update_mutex);
          get_flatfield();
          imagePyramid->set_mag_label(componentMagLabel);
          parent->MRImageSet->sort_by_scale();
        }
      }

      update_Bbox_no_composite({ri});

      waitingFrames[positionForNextWaitngFrame % frameDelay] = {img, {}};
      std::vector<Point2i> immediateProcessingTiles;


      //grab affected tiles with their category of coverage
      auto affectedPyramidTilesWithStatus = calculate_affected_tiles_with_status(
        Point2f(ri->absoluteCoords));


      //calculate: for which of the affected tiles is this frame an improvement?
      for (auto &ptStat: affectedPyramidTilesWithStatus) {
        auto pyrTileObj = imagePyramid->get_base_tile(ptStat.first);

        //check if frame improves status of tile, if so process immediately
        if (pyrTileObj->status < ptStat.second) {
          //if the tile is promoting to singleFrameCoverage, set owner and motionBlur from this frame
          if (ptStat.second == TileObj::singleFrameCoverage) {
            img->ownedTiles.insert(ptStat.first);
            pyrTileObj->owner = img;
            pyrTileObj->status = ptStat.second;
            immediateProcessingTiles.push_back(ptStat.first);
          }
          // //this if you want edge tiles. not very functional
          // pyrTileObj->status = el.second;
          // immediateProcessingTiles.push_back(el.first);
        }

        // check later if frame is less blurry than current source for tile (pyrTileObj)
        else if (ptStat.second == TileObj::singleFrameCoverage) {
          waitingFrames[positionForNextWaitngFrame % frameDelay].second.push_back(ptStat.first);
          // img->ownedTiles.insert(ptStat.first);
        }
      }
      ++positionForNextWaitngFrame;

      if (!immediateProcessingTiles.empty()) {
        process_tiles(img, immediateProcessingTiles);
      }

      //if (imagePyramid->scale > 0) {
      float x = (imagePyramid->offset.x + img->regInfo->absoluteCoords.x) * imagePyramid->scale;
      float y = (imagePyramid->offset.y + img->regInfo->absoluteCoords.y) * imagePyramid->scale;
      float w = parent->image_width * imagePyramid->scale;
      float h = parent->image_height * imagePyramid->scale;
      bool showAsCircle = (componentMagLabel == Image::_2X);

      parent->update_last_frame(Rect_<float>(x, y, w, h), showAsCircle, componentIndex,
                                Image::get_label(componentMagLabel), get_scale());
      //}
    } else {
      waitingFrames[positionForNextWaitngFrame % frameDelay] = {nullptr, {}};
      ++positionForNextWaitngFrame;
    }
    // PROCESS NEW FRAMES END


    //PROCESS OLD FRAMES BEGIN
    /*
        here we process delayed frames, allowing the least blurry frames to win out before processing.
        this is an erase-remove_if implementation with a lambda function that updates the tileObj
        if img should be owner, otherwise it removes the tile from the img's list as another img can fill that tile with
        better data
    */
    for (auto &[img,tiles]: waitingFrames) {
      if (!img) { continue; }

      tiles.erase(
        std::remove_if(tiles.begin(),
                       tiles.end(),
                       [&](Point2i &tileIdx) {
                         auto tileObj = imagePyramid->get_base_tile(tileIdx);
                         if (tileObj->owner == img) {
                           //im marked as the owner, so i keep it
                           return false;
                         }
                         if (image_improves_tile(tileObj, img)) {
                           //I can fill this tile and I'm now recognized as the best image to do it, so i keep it
                           tileObj->owner->ownedTiles.erase(tileIdx); //tileObj->owner is guaranteed to be non null here
                           img->ownedTiles.insert(tileObj->index);

                           tileObj->owner = img;
                           return false;
                         }
                         //I could fill it, but i never became acknowledged as the best frame to do so, so i give it up
                         img->ownedTiles.erase(tileIdx);
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
      needsAlignment = true;

      process_tiles(img, tiles);

      tiles.clear();
      img->free_memory_RAW();
      img = nullptr;
    }
    // PROCESS OLD FRAMES END
  }



  void MetricComposite::align_and_rebuild() {
    // combining components - should only be relevant if CompositeManager::combine_components() ran prior to alignment
    for (auto &loser: absorbedComponents) {
      ftg->storedMatches.insert(loser->ftg->storedMatches.begin(), loser->ftg->storedMatches.end());
      for (auto &img: loser->landmarkFrames) {
        landmarkFrames.push_back(img);
      }
      for (auto &match: ftg->interComponentMatches[loser->componentIndex]) {
        ftg->storedMatches.insert(match);
      }
      loser->root->regInfo->root = false;
    }


    auto start = std::chrono::high_resolution_clock::now();

    auto ig = ImageGraph();

    std::unordered_set<Image *> members = find_contributing_images();
    members.insert(root);
    for (auto img: landmarkFrames) {
      members.insert(img);
    }
    auto matches = ftg->storedMatches; //matches are just stored here before being processed all at once.
    std::unordered_map<Image *, std::vector<std::shared_ptr<Match> > > adjacency;

    for (auto &m: matches) {
      adjacency[m->image_1].push_back(m);
      adjacency[m->image_2].push_back(m);
    }
    std::queue<Image *> q;

    // Seed with confirmed members
    for (auto img: members) {
      img->regInfo->component_membership = componentIndex;
      q.push(img);
    }

    while (!q.empty()) {
      Image *img = q.front();
      q.pop();

      for (auto m: adjacency[img]) {
        Image *other;
        Point2i offset;

        if (m->image_1 == img) {
          other = m->image_2;
          offset = Point2i(m->t_x, m->t_y);
        } else {
          other = m->image_1;
          offset = Point2i(-m->t_x, -m->t_y);
        }

        // If not yet assigned
        if (other->regInfo->component_membership != componentIndex) {
          other->regInfo->component_membership = componentIndex;
          other->regInfo->absoluteCoords = img->regInfo->absoluteCoords - offset;
          other->regInfo->matchedTo = img->index;
          other->regInfo->relativeCoords = offset;

          members.insert(other);
          if (other->regInfo->component_membership != componentIndex) {
            int k = 0;
          }
          q.push(other);
        }
      }
    }

    members = reduce_members_through_competition(members);
    members.insert(root);

    std::vector membersForRebuild(members.begin(), members.end());

    for (auto m: matches) {
      ig.addEdge(m->image_1->index, m->image_2->index, ImageGraph::EdgeKind::ORB);
    }
    for (int ii = 0; ii < extraMatches.size(); ++ii) {
      auto [img1,img2,kp1,kp2] = extraMatches[ii];
      ig.addEdge(img1->index, img2->index, ImageGraph::EdgeKind::SIFT);
    }


    for (auto img: members) {
      ig.setMember(img->index, true);
    }

    std::vector<size_t> discardedIslands;
    auto graphConnectivityResult = ig.computeMinPromotionsToConnectMembersPreferORB();
    if (!graphConnectivityResult.success) {
      std::sort(graphConnectivityResult.member_islands.begin(), graphConnectivityResult.member_islands.end(),
                [](const std::vector<long> &a, const std::vector<long> &b) {
                  return a.size() > b.size();
                });

      members.clear();
      assert(!graphConnectivityResult.member_islands[0].empty());
      auto imgRefs = parent->get_image_ref(graphConnectivityResult.member_islands[0]);
      members.insert(imgRefs.begin(), imgRefs.end());

      Image* closestIndexToRoot = nullptr;
      for (auto el : members) {
        if (el->index >= root->index) {
          if (closestIndexToRoot && closestIndexToRoot->index - root->index > el->index - root->index) {
            closestIndexToRoot = el;
          }else {
            closestIndexToRoot = el;
          }
        }
        if (el == root){break;}
      }

      if (closestIndexToRoot != root) {
        closestIndexToRoot->regInfo->root = true;
        imagePyramid->offset += Point2f(closestIndexToRoot->regInfo->absoluteCoords);
      }

      for (int i = 1; i < graphConnectivityResult.member_islands.size(); ++i) {
        discardedIslands.push_back(graphConnectivityResult.member_islands[i].size());
      }
    }

    if (!graphConnectivityResult.promoted_nodes.empty()) {
      //important to check if empty or get_image_ref returns every image known to StreamCam
      for (auto img: parent->get_image_ref(graphConnectivityResult.promoted_nodes)) {
        ig.setMember(img->index, true);
        members.insert(img);
        if (img->regInfo->component_membership != componentIndex) {
          int k = 0;
        }
      }
    }
    auto memberOverlaps = calculate_member_overlaps(std::vector(members.begin(), members.end()));
    auto start1 = std::chrono::high_resolution_clock::now();
    ImageGraph::PromoteMembersForOverlapConnectivityShortestHop(members, memberOverlaps,
                                                                std::vector(
                                                                  ftg->storedMatches.begin(),
                                                                  ftg->storedMatches.end()));
    auto t4 = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::high_resolution_clock::now() - start1).count();


    for (auto m: matches) {
      if (members.find(m->image_1) != members.end() && members.find(m->image_2) != members.end()) {
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
      img->regInfo->wasAligned = true;
    }

    if (graphConnectivityResult.used_sift) {
      std::cout << "Component " << componentIndex << " using sift in BA" << std::endl;
      return;
    }

    //GENERATE TRACKS AND RUN
    std::vector memberImages(members.begin(), members.end());

    auto tracks = ftg->generateCurrentTracks(memberImages);
    auto iters = BundleAdjustmentIntegrator::run_coopers_planar_ba_edge_list(
      tracks, memberImages, 2 * memberImages.size() + 5000);

    rebuild(membersForRebuild);

    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::high_resolution_clock::now() - start).count();

    Poco::FastMutex::ScopedLock lock(parent->printToScreenMutex);
    std::cout << "ALIGNMENT OF COMPONENT " << componentIndex << " MAGLABEL " << Image::get_label(componentMagLabel) <<
        std::endl;

    if (!discardedIslands.empty()) {
      std::cout << "failed to connect graph, discarded " << discardedIslands.size() << " islands, "
          << std::accumulate(discardedIslands.begin(), discardedIslands.end(), size_t{0}) << " frames" << std::endl;
    }

    if (!graphConnectivityResult.promoted_nodes.empty()) {
      std::cout << "promoted " << graphConnectivityResult.promoted_nodes.size() << " frames" << std::endl;
    }

    std::cout << memberImages.size() << " frames aligned in " << iters.first << " and " << iters.second << " iterations"
        << std::endl;
    std::cout << "total align time comp " << componentIndex << ": " << t3 << std::endl;
  }


  void MetricComposite::rebuild(const std::vector<Image *> &members) {
    for (auto &tileIdx: imagePyramid->liveTiles) {
      auto to = imagePyramid->get_base_tile(tileIdx);
      to->owner = nullptr;
    }
    auto liveTilesCopy = imagePyramid->liveTiles;
    imagePyramid->liveTiles.clear();

    //debug/assert lambda funciton for checking tiles owned by an image fully reside within that image
    auto check_tile_img = [](cv::Point2i abc_, cv::Point2i tileIdx_, int tileSize_, cv::Size imageSize_) {
      auto ul = tileIdx_ * tileSize_;
      bool crit1 = abc_.x <= ul.x && abc_.y <= ul.y;
      return crit1 &&
             abc_.x + imageSize_.width >= ul.x + tileSize_ &&
             abc_.y + imageSize_.height >= ul.y + tileSize_;
    };

    for (auto &img: members) {
      auto affectedTilesWithStatus = calculate_affected_tiles_with_status(img->regInfo->absoluteCoords);

      for (auto &p: affectedTilesWithStatus) {
        if (p.second == TileObj::singleFrameCoverage) {
          assert(check_tile_img(img->regInfo->absoluteCoords,p.first,parent->tileSize,imageSize));
          auto tileObj = imagePyramid->get_base_tile(p.first);
          if (image_improves_tile(tileObj, img)) {
            tileObj->owner = img;
            imagePyramid->liveTiles.insert(p.first);
          }
        }
      }
    }

    // std::vector<std::pair<Image *, std::vector<const SiftPoint *> > > componentFeatures;
    // Point2i siftWindowCorner((imageSize.width - parent->siftWindow) / 2, (imageSize.height - parent->siftWindow) / 2);

    for (auto &img: members) {
      std::vector<Point2i> tileIndexes;
      for (auto &tileIdx: imagePyramid->liveTiles) {
        auto tileObj = imagePyramid->get_base_tile(tileIdx);
        if (tileObj->owner == img) {
          tileIndexes.push_back(tileIdx);
          assert(check_tile_img(img->regInfo->absoluteCoords,tileIdx,parent->tileSize,imageSize));
        }
      }
      img->subsequentMatchLaunched = true;
      process_tiles(img, tileIndexes, false);


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


    Rect tileROI(0, 0, parent->tileSize, parent->tileSize);
    for (auto &tileIdx: liveTilesCopy) {
      auto tileObj = imagePyramid->get_base_tile(tileIdx);
      if (!tileObj->owner) {
        //kill tile
        tileObj->image.setTo(Scalar(0, 0, 0, 0));
        Rect tileRegion(tileIdx * parent->tileSize, Size(parent->tileSize, parent->tileSize));
        imagePyramid->level[0]->tileUpwards(tileIdx, tileRegion, tileObj, tileROI);
        tileObj.reset();
      }
    }

    // compSiftData = collect_SiftData(componentFeatures, 5);
    successfullyAligned = true;
  }

  std::unordered_set<Image *> MetricComposite::reduce_members_through_competition(
    std::unordered_set<Image *> _members) const {
    for (auto &tileIdx: imagePyramid->liveTiles) {
      auto to = imagePyramid->get_base_tile(tileIdx);
      to->owner = nullptr;
    }

    for (auto &img: _members) {
      img->ownedTiles.clear();
    }
    imagePyramid->liveTiles.clear();

    for (auto &img: _members) {
      auto affectedTilesWithStatus = calculate_affected_tiles_with_status(img->regInfo->absoluteCoords);

      for (auto &p: affectedTilesWithStatus) {
        if (p.second == TileObj::singleFrameCoverage) {
          auto tileObj = imagePyramid->get_base_tile(p.first);

          if (image_improves_tile(tileObj, img)) {
            if (tileObj->owner) {
              tileObj->owner->ownedTiles.erase(tileObj->index);
            }
            img->ownedTiles.insert(tileObj->index);

            tileObj->owner = img;
            imagePyramid->liveTiles.insert(p.first);
          }
        }
      }
    }
    return find_contributing_images();
  }


  void MetricComposite::process_tiles(Image *img, std::vector<Point2i> &tiles, bool alertDoubleLoad,
                                      const bool forceFullImage) {
    img->load_raw_from_disk(alertDoubleLoad);

    //put raw data into fourChannelPreallocated
    prepare_4CPA_cpu(img, tiles, forceFullImage);
    img->free_memory_RAW();

    //calculate region of pyramid for data placement
    auto imageBox = cv::Rect_<float>(img->regInfo->absoluteCoords.x, img->regInfo->absoluteCoords.y, img->width,
                                     img->height);
    Mat mask = componentMagLabel == Image::_2X ? circleMask : rectMask;

    // cv::Mat randomcolor(imageSize.height, imageSize.width, CV_8UC4,
    //                     cv::Scalar(rand() & 255, rand() & 255, rand() & 255, 255));
    // imagePyramid->insertTilesAtBase(randomcolor, mask, imageBox, tiles);

    imagePyramid->insertTilesAtBase(fourChannelPreallocated, mask, imageBox, tiles);
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


  size_t MetricComposite::consolidate_tile_ownership() {
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    constexpr int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    std::vector<std::pair<Point2i, Image *> > updates;

    for (auto &tileIdx: imagePyramid->liveTiles) {
      std::unordered_map<Image *, int> neighborCount;
      for (int i = 0; i < 8; ++i) {
        //this method of grabbing tiles returns null if it doesnt exist rather than creating it
        auto tileObj = imagePyramid->level[0]->tiles(tileIdx.x + dx[i], tileIdx.y + dy[i]);
        if (tileObj && tileObj->owner) {
          ++neighborCount[tileObj->owner];
        }
      }

      //this method of grabbing tiles will create it if it doesnt exist, but we know it exists if its in liveTiles
      auto myTile = imagePyramid->get_base_tile(tileIdx);
      auto myOwnerPresence = neighborCount[myTile->owner];

      int maxPresence = myOwnerPresence + 1;
      std::vector<Image *> candidateOwners;
      for (auto &[competingOwner,presence]: neighborCount) {
        if (presence > maxPresence) {
          candidateOwners.clear();
          candidateOwners.push_back(competingOwner);
        } else if (presence == maxPresence) {
          candidateOwners.push_back(competingOwner);
        }
      }

      //max distance a tile could be from an img
      int bestDistance = (imageSize.width / 2 * imageSize.width / 2) + (imageSize.height / 2 * imageSize.height / 2);
      Image *winner = myTile->owner;
      for (auto &img: candidateOwners) {
        auto dist = get_sqrd_center_distance_tile_to_img(img->regInfo->absoluteCoords, tileIdx);
        if (dist < bestDistance) {
          bestDistance = dist;
          winner = img;
        }
      }
      if (winner != myTile->owner) {
        updates.emplace_back(tileIdx, winner);
      }
    }
    for (auto &[tileIdx,img]: updates) {
      imagePyramid->get_base_tile(tileIdx)->owner = img;
    }
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::high_resolution_clock::now() - start).count();

    return updates.size();
  }

  bool MetricComposite::image_improves_tile(const std::shared_ptr<TileObj> &_to, Image *_img) const {
    //tile has no owner, candidate frame wins by default
    if (!_to->owner) {
      return true;
    }

    float myBlur, theirBlur;
    {
      std::lock_guard lock(_img->blurMutex);
      myBlur = _img->motionBlur;
    }
    {
      std::lock_guard lock(_to->owner->blurMutex);
      theirBlur = _to->owner->motionBlur;
    }
    //frames have about the same blur, prioritize closeness to center of frame instead unless the tile is already
    //pretty close to the center of the frame
    if (std::abs(theirBlur - myBlur) < /*0.01f*/50) {
      if (_to->owner->ownedTiles.size() < 12 && _img->ownedTiles.size() > 12) {
        //owner does not have sufficient presence and should be removed to reduce member image count
        return true;
      }

      auto v1 = get_sqrd_center_distance_tile_to_img(_to->owner->regInfo->absoluteCoords, _to->index);
      return v1 > 25 * parent->tileSize * parent->tileSize + get_sqrd_center_distance_tile_to_img(
               _img->regInfo->absoluteCoords, _to->index);
    }

    //amount of motion blur is significantly different, choose clearest image
    return theirBlur > myBlur;
  }


  std::unordered_set<Image *> MetricComposite::find_contributing_images() const {
    std::unordered_set<Image *> members;

    for (auto &tileIdx: imagePyramid->liveTiles) {
      auto to = imagePyramid->get_base_tile(tileIdx);
      members.insert(to->owner);
    }

    for (auto &[img,tileIdx]: waitingFrames) {
      if (img) {
        members.insert(img);
      }
    }

    return members;
  }

  std::vector<std::pair<Image *, Image *> > MetricComposite::calculate_member_neighbors() {
    std::unordered_set<std::pair<Image *, Image *>, ImagePairHash, ImagePairEqual> neighborPairs;
    for (auto &tileIdx: imagePyramid->liveTiles) {
      auto tileObj = imagePyramid->get_base_tile(tileIdx);

      //up down left right of tileIdx
      for (int i = 0; i < 4; ++i) {
        int x = -1 + 2 * (i % 2) + tileIdx.x;
        int y = -1 + 2 * (i / 2) + tileIdx.y;
        if (auto toNeighbor = imagePyramid->level[0]->tiles(x, y); toNeighbor) {
          if (toNeighbor->owner != tileObj->owner) {
            neighborPairs.insert({toNeighbor->owner, tileObj->owner});
          }
        }
      }
    }
    return {neighborPairs.begin(), neighborPairs.end()};
  }

  void MetricComposite::add_landmark_frame(Image *img) {
    landmarkFrames.push_back(img);
    if (!img->subsequentMatchLaunched) {
      img->load_raw_from_disk(false); //freed in ComponentMatchSearch::run()
      img->subsequentMatchLaunched = true;
      ++outstandingCMS_jobs;
      auto members = find_contributing_images();
      auto cms = new ComponentMatchSearch(parent, img, this, {members.begin(), members.end()});
      parent->jqSecondary->add_runnable(cms);
    }
  }

  void MetricComposite::launch_component_match_search_with_XC(Image *img_) {
    auto members = find_contributing_images();
    members.insert(root);

    Poco::RWLock::ScopedReadLock lock(parent->component_mutex);

    for (auto &component: parent->composites) {
      if (component->componentIndex == componentIndex) { continue; }
      if (component->componentMagLabel == componentMagLabel) {
        auto ans = component->find_contributing_images();
        members.insert(ans.begin(), ans.end());
        members.insert(component->root);
      }
    }

    launch_component_match_search(img_, {members.begin(), members.end()});
  }
}
