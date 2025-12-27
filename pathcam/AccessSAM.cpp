//
// Created by cooper maira on 7/25/25.
//

#include "pathCam.h"





namespace pathCam {
  using namespace nvinfer1;

  nvLogger nvloger;
  
  void AccessSAM::initialize() {
    cudaSetDevice(parent->compositorCudaDevice);
    int interval = (parent->SAMTileSize / parent->tileSize);
    assert(interval % 2 == 0);
    auto comp = parent->composites[0];

    auto ul = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto lr = comp->imagePyramid->level[0]->getIJ(Point2f(comp->max_offset.x, comp->max_offset.y));

    int id = 0;
    int yTileCount = 0;

    for (int y = ul.y; y <= lr.y; ++y) {
      if ((yTileCount - 1) % (interval - 1) == 0) {
        int xTileCount = 0;

        for (int x = ul.x; x <= lr.x; ++x) {
          if ((xTileCount - 1) % (interval - 1) == 0) {
            auto st = new SAMTile(id, {x - 1, y - 1}, this, 0, parent->SAMTileSize);
            st->valid = true;
            tiles.push_back(st);
            ++id;

            //add links to neighbors
            if (xTileCount > 0) {
              auto brotherX = tiles[get_tile_id({x - interval + 1, y}, comp->componentIndex)];
              std::vector<Point2i> temp;
              for (int yy = 0; yy < interval; ++yy) {
                temp.emplace_back(x - 1, y - 1 + yy);
              }
              tiles.back()->neighbors.emplace_back(brotherX, temp);
              brotherX->neighbors.emplace_back(tiles.back(), temp);
            }
            if (yTileCount > 0) {
              auto brotherY = tiles[get_tile_id({x, y - interval + 1}, comp->componentIndex)];
              std::vector<Point2i> temp;
              for (int xx = 0; xx < interval; ++xx) {
                temp.emplace_back(x - 1 + xx, y - 1);
              }
              tiles.back()->neighbors.emplace_back(brotherY, temp);
              brotherY->neighbors.emplace_back(tiles.back(), temp);
            }
          }
          ++xTileCount;
        }
      }
      ++yTileCount;
    }
  }

  void AccessSAM::load_model() {

    auto dBlob = readFile(parent->SAM_decoder_path.toString());

    IRuntime *dRuntime = createInferRuntime(nvloger);
    decoderEngine = dRuntime->deserializeCudaEngine(dBlob.data(), dBlob.size());
    delete dRuntime;

    assert(decoderEngine);
    decoderCtx = decoderEngine->createExecutionContext();
    assert(decoderCtx);

    cudaStreamCreate(&decoderStream);

    auto eBlob = readFile(parent->SAM_encoder_path.toString());

    IRuntime *eRuntime = createInferRuntime(nvloger);
    encoderEngine = eRuntime->deserializeCudaEngine(eBlob.data(), eBlob.size());
    delete eRuntime;

    assert(encoderEngine);
    encoderCtx = encoderEngine->createExecutionContext();
    assert(encoderCtx);

    cudaStreamCreate(&encoderStream);
  }


  void AccessSAM::embed_SAM_tiles(bool _buildBuffer) {
    load_model();

    std::deque<SAMTile *> embedQueue;
    for (auto &st: tiles) {
      if (st->valid) {
        embedQueue.push_back(st);
      }
    }
    size_t nElementsPerChannel = parent->SAMTileSize * parent->SAMTileSize;
    auto comp = parent->composites[0];

    SAMTile *tileToFree = nullptr;

    while (!embedQueue.empty()) {
      std::sort(embedQueue.begin(), embedQueue.end(), tile_compare);
      auto tile = embedQueue.front();
      embedQueue.pop_front();

      if (_buildBuffer) {
        tile->get_tile_data(comp, parent->SAMTileSize / parent->tileSize);
        CHECK_CUDA(cudaMalloc(&tile->rawBuffer, nElementsPerChannel * 3 * sizeof(float)));
        tile->make_raw_buffer(tile->rawBuffer);
      }
      CHECK_CUDA(cudaMalloc(&tile->high_res_feats_0, 32 * 256 * 256 * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&tile->high_res_feats_1, 64 * 128 * 128 * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&tile->image_embed, 256 * 64 * 64 * sizeof(float)));

      /*the more intuitive way to do this is to use an 'input consumed' event and then free the raw buffer as soon as that
       * turns true rather than wait for the next iteration, sync the whole stream and then free the input. The reason
       * for doing it this way is to control what gets run next after each run is completed while allowing as much parallelism
       * between runs as possible. EnqueuV2 launches the kernel on its own thread, meanwhile the nex tile can prep its data
       * but if the priority of the tiles changes, theres still a chance between iterations for that to take effect. If we
       * enqueued them all at once, this wouldnt be the case. If we block for input consumed, we dont allow parallelism.
       */
      CHECK_CUDA(cudaStreamSynchronize(encoderStream));
      if (tileToFree) {
        tile->embeddingComplete = true;
        CHECK_CUDA(cudaFree(tileToFree->rawBuffer));
      }

      encoderCtx->setInputTensorAddress("image", tile->rawBuffer);
      encoderCtx->setOutputTensorAddress("high_res_feats_0", tile->high_res_feats_0);
      encoderCtx->setOutputTensorAddress("high_res_feats_1", tile->high_res_feats_1);
      encoderCtx->setOutputTensorAddress("image_embed", tile->image_embed);

      encoderCtx->enqueueV3(encoderStream);


      //std::vector buffer{tile->rawBuffer, tile->high_res_feats_1};
      //speedSam->mImageEncoder->mContext->enqueueV2(buffer.data(), speedSam->mImageEncoder->mCudaStream, nullptr);
      //cudaEventRecord(tile->embeddingCompleteCudaEvent,speedSam->mImageEncoder->mCudaStream);


      tileToFree = tile;
    }

    //free last tile
    if (tileToFree) {
      cudaFree(tileToFree->rawBuffer);
    }

    //delete encoderCtx;
    //delete encoderEngine;

    std::cout << "embedded tiles: " << tiles.size() << std::endl;
  }


  void AccessSAM::create_segmentation_coarse_to_fine(std::vector<Point3f> &_clicks, int _segID, const std::vector<Point2f> &_fov) {
    cudaSetDevice(parent->compositorCudaDevice);
    for (auto &p : _clicks) {
      assert(p.x >= _fov[0].x && p.y >= _fov[0].y && p.x <= _fov[1].x && p.y <= _fov[1].y);
    }
    assert(_fov.size() == 2);

    //get base component (for now)
    auto ipBase = parent->composites[0]->imagePyramid->level[0];

    //create mat from tiles
    auto ul = ipBase->getIJ(_fov[0]);
    auto lr = ipBase->getIJ(_fov[1]);

    if ((lr.x - ul.x) * (lr.y - ul.y) > 10000) {
      std::cout <<"I don't think so. That fild of view is too big. Delete this annotation and try again. Don't zoom out so far when you're clicking"<<std::endl;
      return;
    }

    Rect2i fovRect(_fov[0].x,_fov[0].y,_fov[1].x - _fov[0].x,_fov[1].y - _fov[0].y);
    cuda::GpuMat fovMat(fovRect.height,fovRect.width,CV_8UC4);

    for (int y = ul.y; y <= lr.y; ++y) {
      for (int x = ul.x; x <= lr.x; ++x) {
        Rect2i tileRect(x * parent->tileSize, y * parent->tileSize, parent->tileSize, parent->tileSize);
        auto worldspaceROI = tileRect & fovRect;

        auto tileROI = worldspaceROI;
        tileROI.x -= tileRect.x;
        tileROI.y -= tileRect.y;

        auto fovMatROI = worldspaceROI;
        fovMatROI.x -= fovRect.x;
        fovMatROI.y -= fovRect.y;

        auto tileObj = ipBase->getTile(x,y);
        tileObj->image(tileROI).copyTo(fovMat(fovMatROI));
      }
    }

    //resize field of view to SAM tile size so all clicks can be processed together
    cuda::resize(fovMat,fovMat,Size(parent->SAMTileSize,parent->SAMTileSize));

    //make dummy SAMTile
    auto fovTile = SAMTile(-1,{0,0},this,0);
    fovTile.noncontiguousWrapper = fovMat;
    fovTile.noncontiguousWrapper.download(fovTile.ncwStoreLocal);

    fovTile.make_raw_buffer();
    fovTile.embed_tile_with_engine(encoderCtx);

    //reshape clicks and determine if mask should shrink
    bool unionWithExistingMask = true;
    for (auto &p : _clicks) {
      p.x -= _fov[0].x;
      p.y -= _fov[0].y;

      p.x *= (float)parent->SAMTileSize / (_fov[1].x - _fov[0].x);
      p.y *= (float)parent->SAMTileSize / (_fov[1].y - _fov[0].y);
      unionWithExistingMask = unionWithExistingMask && p.z;
    }
    fovTile.clicksVec = _clicks;

    //generate segmentation for FOV
    auto output = fovTile.run_segmentation(_segID, false);

    cuda::resize(output,output,fovRect.size());
    cuda::threshold(output,output,0,255,THRESH_BINARY);
    output.convertTo(output,CV_8U);

    cuda::GpuMat tileMaskPreal(parent->tileSize,parent->tileSize,CV_8U);

    for (int y = ul.y; y <= lr.y; ++y) {
      for (int x = ul.x; x <= lr.x; ++x) {

        Point2i locationPoint(x * parent->tileSize,y * parent->tileSize);
        Rect tileRect(locationPoint ,Size(parent->tileSize,parent->tileSize));
        Rect worldspaceROI = tileRect & fovRect;

        Rect fovROI = worldspaceROI;
        fovROI.x -= fovRect.x;
        fovROI.y -= fovRect.y;

        if (cuda::countNonZero(output(fovROI))) {

          Rect tileROI = worldspaceROI;
          tileROI.x -= tileRect.x;
          tileROI.y -= tileRect.y;

          tileMaskPreal.setTo(Scalar(0));
          output(fovROI).copyTo(tileMaskPreal(tileROI));

          push_mask_for_display({x,y},0,tileMaskPreal,_segID,unionWithExistingMask);
          // maskMatHolding_32F.setTo(Scalar(0));
          // maskMatHolding_8U.setTo(Scalar(0));
          // output(fovROI).copyTo(maskMatHolding_32F(samTileROI));
          // cuda::resize(maskMatHolding_32F,samTile->inputMaskMat,{256,256});
          //
          //
          // Mat mask;
          // cuda::threshold(maskMatHolding_32F, maskMatHolding_32F, 0, 255, THRESH_BINARY);
          // maskMatHolding_32F.convertTo(maskMatHolding_8U,CV_8U);
          // maskMatHolding_8U.download(mask);
          // auto test2 = samTile->debug_draw_tile_with_clicks_and_mask(samTile->ncwStoreLocal,mask,Scalar(0,150,0,255),0.7,_segID);
          // imwrite("/media/max/Data/pathcam_SAM/test_images/"+std::to_string(samTile->location.x)+"_"+std::to_string(samTile->location.y)+"_inputmask.png",test2);

          // auto tileOutput = samTile->run_segmentation(_segID,false);
          // cuda::resize(tileOutput,maskMatHolding_32F,{1024,1024});
          // cuda::threshold(maskMatHolding_32F,maskMatHolding_32F, 0, 255, THRESH_BINARY);
          // maskMatHolding_32F.convertTo(maskMatHolding_8U,CV_8U);
          // maskMatHolding_8U.download(mask);
          // auto test3 = samTile->debug_draw_tile_with_clicks_and_mask(samTile->ncwStoreLocal,mask,Scalar(0,150,0,255),0.7,_segID);
          // imwrite("/media/max/Data/pathcam_SAM/test_images/"+std::to_string(samTile->location.x)+"_"+std::to_string(samTile->location.y)+"_refined.png",test3);

        }
      }
    }

    int k = 0;

  }



  void AccessSAM::create_segmentation(std::vector<Point3f> &_clicks, int _segID) {
    cudaSetDevice(parent->compositorCudaDevice);


    // Set the optimization profile
    decoderCtx->setOptimizationProfileAsync(0, decoderStream);

    while (!segmentProcessQ.empty()) {
      //clear the queue. the only thing that could be in here at this point is from debug
      segmentProcessQ.pop_front();
    }

    //a tile, its list of clicks, and if each click appears in other tiles as well
    std::map<int, std::vector<std::pair<Point3f, bool> > > tilesAndTheirClicks;

    for (auto &click: _clicks) {
      //get list of tiles this click falls in
      auto ans = get_tiles_covering_point(Point2f(click.x, click.y), parent->SAMTileSize - parent->tileSize);

      //for each tile, add this click to its list
      for (auto &tile: ans) {
        tilesAndTheirClicks[tile].push_back({click, ans.size() > 1});
      }
    }

    //choose which tiles should run with which clicks
    auto ans = choose_clicks_for_each_tile(tilesAndTheirClicks);

    //set the points and push tile to process Q
    for (auto &kv: ans) {
      auto tile = tiles[kv.first];
      tile->increase_embed_priority();
      if (tile->segmentations.find(_segID) != tile->segmentations.end()) {
        tile->runIsRepeat = true;
      }

      for (auto &point: kv.second) {
        Point3f pointInTileSpace = point - Point3f((float) parent->tileSize * tile->location.x,
                                                   (float) parent->tileSize * tile->location.y, 0);

        //debug test
        if (pointInTileSpace.x < 0 || pointInTileSpace.y < 0 || pointInTileSpace.x > 1024 || pointInTileSpace.y >
            1024) {
          int k = 0;
        }

        tile->clicksVec.push_back(pointInTileSpace);
      }


      //tile ready to be enqueued
      tile->canRun = true;
      segmentProcessQ.push_back(tile);
    }

    while (!segmentProcessQ.empty()) {
      //sort by clicksVec size
      std::sort(segmentProcessQ.begin(), segmentProcessQ.end(),
          [](SAMTile* a, SAMTile* b) {
              return a->clicksVec.size() > b->clicksVec.size();
          });

      //get front tile
      auto tile = segmentProcessQ.front();
      segmentProcessQ.pop_front();

      //skip tile if its already been processed. This happens because it gets added once if it has clicks
      //and can then be subsequently added by neighbors if there is mask > 0 in overlapping regions
      if (tile->canRun) {
        //ensure the tiles embedding has completed
        cudaEventSynchronize(tile->embeddingCompleteCudaEvent);

        //process the segmentation
        tile->run_segmentation(_segID);
      }
    }
  }


  std::vector<int> AccessSAM::get_tiles_covering_point(const Point2f &_point, int _stride) {
    std::vector<int> out;

    auto comp = parent->composites[0];
    auto rootTile = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto myTile = comp->imagePyramid->level[0]->getIJ(_point);
    auto locInTile4 = _point - Point2f((float) parent->tileSize * myTile.x, (float) parent->tileSize * myTile.y);
    if (locInTile4.x < 0 || locInTile4.y < 0 || locInTile4.y > 256 || locInTile4.x > 256) {
      throw std::runtime_error("ACCESS SAM ERROR: RETURNING INCORRECT BASE TILE FOR CLICK POINT");
    }

    auto primarySAMTileID = get_tile_id(myTile, 0);
    auto primarySAMTile = tiles[primarySAMTileID];
    auto locInTile12 = _point - Point2f((float) parent->tileSize * primarySAMTile->location.x,
                                        (float) parent->tileSize * primarySAMTile->location.y);
    if (locInTile12.x > parent->SAMTileSize || locInTile12.x < 0 || locInTile12.y > parent->SAMTileSize || locInTile12.y < 0) {
      // std::vector<SAMTile *> correctTiles;
      // for (auto &tile: tiles) {
      //   if (_point.x - tile->location.x >= 0 && _point.x - tile->location.x <= 1024 && _point.y - tile->location.y >= 0
      //       && _point.y - tile->location.y <= 1024) {
      //     correctTiles.push_back(tile);
      //   }
      // }
      // get_tile_id(myTile, 0);
      // throw std::exception();
      std::cout << "ACCESS SAM WARNING: INVALID CLICK LOCATION"<<std::endl;
    }

    out.push_back(primarySAMTileID);

    auto diff = myTile - rootTile;
    Point2i baseTileInSAMTile(diff.x % 3, diff.y % 3);


    if (baseTileInSAMTile.x == 0) {
      //pushback SAM tile (-1,0) from primary tile
      auto pointThatWillFallInTile = myTile - Point2i(_stride / (int) parent->tileSize, 0);
      auto id = get_tile_id(pointThatWillFallInTile, 0);
      auto tile = tiles[id];
      auto locInTile1 = _point - Point2f((float) parent->tileSize * tile->location.x,
                                         (float) parent->tileSize * tile->location.y);
      if (locInTile1.x > 1024 || locInTile1.x < 0 || locInTile1.y > 1024 || locInTile1.y < 0) {
        int k = 0;
      }
      out.push_back(id);
    }
    if (baseTileInSAMTile.y == 0) {
      //pushback SAM tile (0,-1) from primary tile
      auto pointThatWillFallInTile = myTile - Point2i(0, _stride / (int) parent->tileSize);
      auto id = get_tile_id(pointThatWillFallInTile, 0);
      auto tile = tiles[id];
      auto locInTile1 = _point - Point2f((float) parent->tileSize * tile->location.x,
                                         (float) parent->tileSize * tile->location.y);
      if (locInTile1.x > 1024 || locInTile1.x < 0 || locInTile1.y > 1024 || locInTile1.y < 0) {
        int k = 0;
      }
      out.push_back(id);
    }
    if (baseTileInSAMTile.x == 0 && baseTileInSAMTile.y == 0) {
      //pushback SAM tile (-1,-1) from primary tile
      auto pointThatWillFallInTile = myTile - Point2i(_stride / (int) parent->tileSize,
                                                      _stride / (int) parent->tileSize);
      auto id = get_tile_id(pointThatWillFallInTile, 0);
      auto tile = tiles[id];
      auto locInTile1 = _point - Point2f((float) parent->tileSize * tile->location.x,
                                         (float) parent->tileSize * tile->location.y);
      if (locInTile1.x > 1024 || locInTile1.x < 0 || locInTile1.y > 1024 || locInTile1.y < 0) {
        int k = 0;
      }
      out.push_back(id);
    }

    return out;
  }

  void AccessSAM::on_click(Point3f _click) {
    auto effectedTiles = get_tiles_covering_point(Point2f(_click.x, _click.y), parent->SAMTileSize - parent->tileSize);
    for (auto &ind: effectedTiles) {
      tiles[ind]->increase_embed_priority();
    }
  }


  std::map<int, std::vector<Point3f> > AccessSAM::choose_clicks_for_each_tile(
    const std::map<int, std::vector<std::pair<Point3f, bool> > > &clicksByTile, int cap) {
    std::map<int, std::vector<Point3f> > result;

    for (const auto &[tileId, clicks]: clicksByTile) {
      // Split by uniqueness (non-overlap vs overlap) and label (yes/no)
      std::vector<cv::Point3f> Uyes, Uno, Oyes, Ono;
      Uyes.reserve(clicks.size());
      Uno.reserve(clicks.size());
      Oyes.reserve(clicks.size());
      Ono.reserve(clicks.size());

      for (const auto &c: clicks) {
        const Point3f &p = c.first;
        const bool isOverlap = c.second;
        const bool isYes = (p.z >= 0.5f);

        if (!isOverlap) (isYes ? Uyes : Uno).push_back(p);
        else (isYes ? Oyes : Ono).push_back(p);
      }

      auto pick = [](std::vector<cv::Point3f> &src, int k, std::vector<cv::Point3f> &out) -> int {
        const int n = std::min<int>(k, static_cast<int>(src.size()));
        out.insert(out.end(), src.begin(), src.begin() + n);
        if (n > 0) src.erase(src.begin(), src.begin() + n);
        return n;
      };

      std::vector<Point3f> sel;
      sel.reserve(std::min<int>(cap, clicks.size()));

      if ((int) clicks.size() <= cap) {
        // If fewer than cap total, just take all (max possible per tile).
        for (const auto &c: clicks) sel.push_back(c.first);
        result.emplace(tileId, std::move(sel));
        continue;
      }

      const int targetYes = cap / 2; // 5
      const int targetNo = cap - targetYes; // 5
      int Ayes = 0, Ano = 0;

      // Step 1a: take uniques toward targets
      Ayes += pick(Uyes, targetYes, sel);
      Ano += pick(Uno, targetNo, sel);
      int rem = cap - (Ayes + Ano);

      // Step 1b: still uniques left? fill while reducing imbalance
      while (rem > 0 && (!Uyes.empty() || !Uno.empty())) {
        const bool takeYes = (Ayes < targetYes && !Uyes.empty()) || Uno.empty();
        if (takeYes) { Ayes += pick(Uyes, 1, sel); } else { Ano += pick(Uno, 1, sel); }
        rem = cap - (Ayes + Ano);
      }

      // Step 2a: pull from overlaps to hit targets
      if (rem > 0) {
        const int needYes = std::max(0, targetYes - Ayes);
        const int needNo = std::max(0, targetNo - Ano);
        Ayes += pick(Oyes, needYes, sel);
        Ano += pick(Ono, needNo, sel);
        rem = cap - (Ayes + Ano);
      }

      // Step 2b: fill remaining from whatever is left (still bias toward balance)
      while (rem > 0 && (!Oyes.empty() || !Ono.empty())) {
        const bool takeYes = (Ayes < Ano && !Oyes.empty()) || Ono.empty();
        if (takeYes) { Ayes += pick(Oyes, 1, sel); } else { Ano += pick(Ono, 1, sel); }
        rem = cap - (Ayes + Ano);
      }

      result.emplace(tileId, std::move(sel));
    }

    return result;
  }


  int AccessSAM::get_tile_id(Point2i _tileIndexPoint, unsigned int _componentIndex) const {
    auto comp = parent->composites[_componentIndex];
    auto rootOffsetTile = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto maxOffsetTile = comp->imagePyramid->level[0]->getIJ(Point2f(comp->max_offset.x, comp->max_offset.y));
    int interval = parent->SAMTileSize / (int) parent->tileSize - 1;
    auto pt1 = (_tileIndexPoint.y - rootOffsetTile.y) / interval;
    auto pt2 = (maxOffsetTile.x - rootOffsetTile.x + 1) / interval;
    auto pt3 = (_tileIndexPoint.x - rootOffsetTile.x) / interval;
    return pt1 * pt2 + pt3;
  }


  void AccessSAM::get_clicks_embedding(std::vector<Point3f> &_clicks, void *&_clicksGPU, void *&_clickLabelsGPU) {
    cudaMalloc(&_clicksGPU, sizeof(float) * 2 * _clicks.size());
    cudaMalloc(&_clickLabelsGPU, sizeof(float) * _clicks.size());

    float clicks[2 * _clicks.size()];
    float clickLabels[_clicks.size()];

    for (int i = 0; i < _clicks.size(); ++i) {
      clicks[2 * i] = _clicks[i].x;
      clicks[2 * i + 1] = _clicks[i].y;
      clickLabels[i] = _clicks[i].z;
    }

    cudaMemcpy(_clicksGPU, clicks, _clicks.size() * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_clickLabelsGPU, clickLabels, _clicks.size() * sizeof(float), cudaMemcpyHostToDevice);
  }


  void AccessSAM::push_mask_for_display(Point2i _tileCoord, unsigned int _componentIndex, const cuda::GpuMat &_mask,
                                        int _segID, bool _unionWithExistingMask) {
    auto pyrBase = parent->composites[_componentIndex]->imagePyramid->level[0];
    auto tileObj = pyrBase->getTile(_tileCoord.x, _tileCoord.y);

    //if no display object exists, initialize one
    if (tileObj->SAMMasks.find(_segID) == tileObj->SAMMasks.end()) {
      tileObj->SAMMasks[_segID] = {cuda::GpuMat(parent->tileSize, parent->tileSize,CV_8U, Scalar(0)), nullptr};
    }

    //combine with current mask by taking max at each pixel
    //parent->composites[0]->imagePyramid->imgPyramidMutex->lock();
    if (_unionWithExistingMask) {
      cuda::max(_mask, tileObj->SAMMasks[_segID].first, tileObj->SAMMasks[_segID].first);
    }else {
      tileObj->SAMMasks[_segID].first = _mask.clone();
    }
    tileObj->newAnnoData = true;

    //pick level region as own bounds and roi as entire tile
    Rect levelRegion(_tileCoord.x * parent->tileSize, _tileCoord.y * parent->tileSize, parent->tileSize,
                     parent->tileSize);
    Rect tileRegion(0, 0, parent->tileSize, parent->tileSize);

    auto level = parent->composites[0]->imagePyramid->level[0];
    level->tileUpwards(_tileCoord, levelRegion, tileObj, tileRegion, _segID);
    //parent->composites[0]->imagePyramid->imgPyramidMutex->unlock();

    parent->notify_observers();
  }

  Point2i AccessSAM::get_transformation_to_display(SAMTile *_samTile, Point2i _subtile) {
    auto myImage = _samTile->img;

    auto theirTileId = get_tile_id(_subtile + _samTile->location, 0);
    auto theirSAMTile = tiles[theirTileId];
    auto theirImage = theirSAMTile->img;

    if (theirImage->index == myImage->index) {
      return {0,0};
    }

    parent->resize_mmatch_mutex->readLock();
    auto m1 = parent->matchM.match[myImage->index][theirImage->index];
    if (m1 == nullptr) {
      parent->resize_mmatch_mutex->unlock();
      return {0,0};
    }
    Point2d actualDistance(myImage->absoluteCoords.x - theirImage->absoluteCoords.x,myImage->absoluteCoords.y - theirImage->absoluteCoords.y);
    Point2d matchedDistance = {m1->t_x,m1->t_y};
    auto res = actualDistance - matchedDistance;
/*
    //visual proof
    Rect myROI(_subtile.x * 256, _subtile.y * 256,256,256);
    auto diff = _samTile->location - theirSAMTile->location;
    auto theirSubTile = _subtile + diff;
    theirSubTile.x = theirSubTile.x %4;
    theirSubTile.y = theirSubTile.y %4;
    Rect theirROI(theirSubTile.x * 256, theirSubTile.y * 256, 256, 256);
    imwrite("/media/max/Data/pathcam_SAM/myTile.png",_samTile->ncwStoreLocal(myROI));
    imwrite("/media/max/Data/pathcam_SAM/theirTile.png",theirSAMTile->ncwStoreLocal(theirROI));
*/
    return res;
  }


  Seeds AccessSAM::pickSeedsFromLogits_v2(const cuda::GpuMat &logits,
                                          const cuda::GpuMat &mask8u,
                                          int k,
                                          bool outsidePrefersHigh,
                                          cuda::Stream stream) {
    CV_Assert(logits.type() == CV_32F && logits.channels() == 1);
    CV_Assert(mask8u.type() == CV_8U && mask8u.size() == logits.size());
    CV_Assert(k > 0 && (k & 1) == 1);

    // 1) Mean logits via normalized box filter (CUDA)
    cuda::GpuMat meanLogits; {
      // Normalized box = local average. CV_32F -> CV_32F.
      // Some builds have createBoxFilter(srcType, dstType, ksize)
      // (normalized by default). If yours requires an anchor, pass Point(-1,-1).
      auto box = cuda::createBoxFilter(CV_32F, CV_32F, Size(k, k));
      box->apply(logits, meanLogits, stream);
    }

    // 2) Build interior (eroded) and outer ring (dilate - original) on GPU
    cuda::GpuMat interior, dilated, outerRing; {
      auto se3 = getStructuringElement(MORPH_RECT, Size(3, 3));
      auto erodeF = cuda::createMorphologyFilter(MORPH_ERODE, CV_8U, se3);
      auto dilateF = cuda::createMorphologyFilter(MORPH_DILATE, CV_8U, se3);
      erodeF->apply(mask8u, interior, stream);
      dilateF->apply(mask8u, dilated, stream);
      cuda::subtract(dilated, mask8u, outerRing, noArray(), CV_8U, stream);
    }

    // Ensure the above GPU ops are done before minMaxLoc (which syncs anyway)
    stream.waitForCompletion();

    // 3) Inlier: argmax of meanLogits inside the interior
    double inMin = 0.0, inMax = 0.0;
    cv::Point inMinLoc, inMaxLoc;
    // Note: cuda::minMaxLoc has signature without Stream; pass a mask for ROI
    cv::cuda::minMaxLoc(meanLogits, &inMin, &inMax, &inMinLoc, &inMaxLoc, interior);

    // 4) Outlier: argmax or argmin in the outer ring (choose based on your logit convention)
    double outMin = 0.0, outMax = 0.0;
    cv::Point outMinLoc, outMaxLoc;
    cv::cuda::minMaxLoc(meanLogits, &outMin, &outMax, &outMinLoc, &outMaxLoc, outerRing);

    Seeds s;
    s.inlier = inMaxLoc;
    s.inlierScore = static_cast<float>(inMax);

    if (outsidePrefersHigh) {
      s.outlier = outMaxLoc;
      s.outlierScore = static_cast<float>(outMax);
    } else {
      s.outlier = outMinLoc;
      s.outlierScore = static_cast<float>(outMin);
    }
    return s;
  }
}
