//
// Created by cooper maira on 7/25/25.
//

#include "pathCam.h"

//#include "utils.hpp"


namespace pathCam {
  SAMTile::SAMTile(int _ID, Point2i _location, AccessSAM *_as, unsigned _componentIndex, unsigned _size) : ID(_ID),
    location(_location),
    componentIndex(_componentIndex),
    size(_size),
    as(_as) {
    noncontiguousWrapper = cuda::GpuMat(size, size,CV_8UC4, Scalar(0, 0, 0, 0));
    inputMask = nullptr;
    outputMask = nullptr;
    confidence = nullptr;
    hasMaskInputGPU = nullptr;
    inputMaskMat = cuda::GpuMat(256, 256,CV_32FC1, Scalar(0));
    cudaEventCreate(&embeddingCompleteCudaEvent);
  }

  void SAMTile::run_segmentation(int _segmentationID) {
    if (clicksVec.empty() && !hasMaskInput) { return; }
    if (clicksVec.empty()) {
      //give dummy input to avoid input requirements, -1 means ignore
      clicksVec.push_back({0,0,-1});
    }

    cuda::GpuMat temp;
    cuda::compare(inputMaskMat,Scalar(0),temp,CMP_GT);
    if (cuda::countNonZero(temp)) {
      hasMaskInput = true;
    }

    //get embeddings
    AccessSAM::get_clicks_embedding(clicksVec, clicksGPU, clickLabelsGPU);

    if (!hasMaskInputGPU) {
      cudaMalloc(&hasMaskInputGPU, sizeof(float));
    }
    auto maskInputVal = static_cast<float>(hasMaskInput); // 1.0f or 0.0f
    cudaMemcpy(hasMaskInputGPU, &maskInputVal, sizeof(float), cudaMemcpyHostToDevice);

    //allocate if not already allocated
    if (!outputMask) {
      cudaMalloc(&outputMask, sizeof(float) * 256 * 256 * 4);
    }
    if (!confidence) {
      cudaMalloc(&confidence, 4 * sizeof(float));
    }
    if (!inputMask) {
      cudaMalloc(&inputMask, sizeof(float) * 256 * 256);
    }

    if (hasMaskInput) {
      Mat inputMaskMatHost;
      inputMaskMat.download(inputMaskMatHost);
      threshold(inputMaskMatHost, inputMaskMatHost, 0.0, 255.0, THRESH_BINARY);
      inputMaskMatHost.convertTo(inputMaskMatHost,CV_8U);
      imwrite("/media/max/Data/pathcam_SAM/"+std::to_string(location.x)+"_"+std::to_string(location.y)+"input.png",inputMaskMatHost);

      cudaMemcpy2D(inputMask,sizeof(float) * 256,inputMaskMat.data,inputMaskMat.step, sizeof(float) * 256,256,cudaMemcpyDeviceToDevice);
    }

    //consolidate into a single buffer and run
    std::vector buffer{feats_1_data_d_, clicksGPU, clickLabelsGPU, inputMask, hasMaskInputGPU, outputMask, confidence};

    // Set input dimensions for coordinates
    as->speedSam->mMaskDecoder->mContext->setBindingDimensions(1, Dims3{1, int(clicksVec.size()), 2});

    // Set input dimensions for labels
    as->speedSam->mMaskDecoder->mContext->setBindingDimensions(2, Dims2{1, int(clicksVec.size())});


    //run inference, sync the stream before trying to access output
    auto ok = as->speedSam->mMaskDecoder->mContext->enqueueV2(buffer.data(),as->speedSam->mMaskDecoder->mCudaStream,nullptr);
    auto err = cudaStreamSynchronize(as->speedSam->mMaskDecoder->mCudaStream);

    if (!ok || err != cudaSuccess) {
      throw std::runtime_error("TensorRT failed: " + std::string(cudaGetErrorString(err)));
    }

    cudaFree(clicksGPU);
    cudaFree(clickLabelsGPU);
    clicksVec.clear();
    hasMaskInput = false;

    cuda::GpuMat output(256, 256,CV_32FC1, outputMask);

    int interval = 256 * 256 / size;

    //distribute mask information as input to neighbors
    int count = 0;
    for (auto &neighbor: neighbors) {
      for (auto &linkedSubTile: neighbor.second) {
        //figure out region of my output to give each tile
        auto mySubTileIndex = linkedSubTile - location;
        Rect myRoi(interval * mySubTileIndex.x, interval * mySubTileIndex.y, interval, interval);

        auto theirSubTileIndex = linkedSubTile - neighbor.first->location;
        Rect theirRoi(interval * theirSubTileIndex.x, interval * theirSubTileIndex.y, interval, interval);

        //take largest logit from mine and theirs as theirs
        auto src1 = output(myRoi);
        auto src2 = neighbor.first->inputMaskMat(theirRoi);
        cuda::max(src1, src2, src2);
      }
      if (neighbor.first->segmentations.find(_segmentationID) == neighbor.first->segmentations.end()
          && cuda::countNonZero(neighbor.first->inputMaskMat)) {
        neighbor.first->hasMaskInput = true;
        as->segmentProcessQ.push(neighbor.first);
      }
    }

    //send my own mask info for display
    cuda::resize(output, output, {int(size), int(size)});
    cuda::threshold(output, output, 0.0, 255.0, THRESH_BINARY);
    output.convertTo(output,CV_8U);
    segmentations[_segmentationID] = output;
    Mat hostOutput;
    output.download(hostOutput);

    auto ans = debug_draw_tile_with_clicks_and_mask(ncwStoreLocal,hostOutput,Scalar(0,180,150,255),0.5);
    imwrite("/media/max/Data/pathcam_SAM/"+std::to_string(location.x)+"_"+std::to_string(location.y)+".png",ans);
    int k = 0;


    //part out the mask to tiles
    int tileSize = as->parent->tileSize;
    for (auto &tile: componentTiles) {
      Rect maskRoi(tile.first.x * tileSize, tile.first.y * tileSize, tileSize, tileSize);
      as->push_mask_for_display(tile.first, componentIndex, hostOutput(maskRoi), _segmentationID);
    }


  }

  void SAMTile::set_component_tile(Point2i _tileID, Point2i _subLocation, cuda::GpuMat &_tileMat) {
    componentTiles.emplace_back(_subLocation, _tileID);
    Rect ROI(_subLocation.x * _tileMat.cols, _subLocation.y * _tileMat.rows, _tileMat.cols, _tileMat.rows);
    try {
      _tileMat.copyTo(noncontiguousWrapper(ROI));
    } catch (...) {
      int k = 0;
    }
  }

  void SAMTile::make_raw_buffer(void *_buffer) {
    //convert to F32, normalize for imagenet. keep in mind the mat is currently in BGRA
    noncontiguousWrapper.convertTo(noncontiguousWrapper,CV_32F);
    cuda::divide(noncontiguousWrapper, Scalar(255, 255, 255, 255), noncontiguousWrapper);
    cuda::subtract(noncontiguousWrapper, Scalar(0.406, 0.456, 0.485, 0), noncontiguousWrapper);
    cuda::divide(noncontiguousWrapper, Scalar(0.225, 0.224, 0.229, 1), noncontiguousWrapper);


    std::vector<cuda::GpuMat> split_channels;
    cuda::split(noncontiguousWrapper, split_channels);

    size_t nBytesPerChannel = sizeof(float) * noncontiguousWrapper.cols * noncontiguousWrapper.rows;
    //cudaMalloc(&rawBuffer, nBytesPerChannel * 3);

    for (int i = 0; i < 3; ++i) {
      void *dst = static_cast<char *>(_buffer) + i * nBytesPerChannel;
      CHECK_CUDA_ERROR(cudaMemcpy2D(dst,
        split_channels[2 - i].cols * sizeof(float),
        split_channels[2 - i].data,
        split_channels[2 - i].step,
        split_channels[2 - i].cols * sizeof(float),
        split_channels[2 - i].rows,
        cudaMemcpyDeviceToDevice));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    noncontiguousWrapper.release();
  }

  void SAMTile::get_tile_data(CompositeVoronoi *_comp, unsigned int _interval) {
    for (int xx = 0; xx < _interval; ++xx) {
      for (int yy = 0; yy < _interval; ++yy) {
        auto gMat = _comp->imagePyramid->level[0]->getTile(location.x + xx, location.y + yy);
        set_component_tile({location.x + xx - 1, location.y + yy - 1}, {xx, yy}, gMat.image);
      }
    }
    //debug
    noncontiguousWrapper.download(ncwStoreLocal);
  }


  void SAMTile::increase_embed_priority() {
    priority = 0;
    for (auto &p: neighbors) {
      p.first->priority = 1;
    }
  }

  Mat SAMTile::debug_draw_tile_with_clicks_and_mask(const cv::Mat &bgraImage, const cv::Mat &binaryMask,
                                                     const cv::Scalar &shadeColor, float alpha)
  {
    CV_Assert(bgraImage.type() == CV_8UC4);
    CV_Assert(binaryMask.type() == CV_8UC1);
    CV_Assert(bgraImage.size() == binaryMask.size());

    // Clone so original isn't modified
    cv::Mat result = bgraImage.clone();

    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(result, channels); // B, G, R, A

    // Make overlay colors (constant per channel)
    cv::Mat overlayB(bgraImage.size(), CV_8UC1, cv::Scalar(shadeColor[0]));
    cv::Mat overlayG(bgraImage.size(), CV_8UC1, cv::Scalar(shadeColor[1]));
    cv::Mat overlayR(bgraImage.size(), CV_8UC1, cv::Scalar(shadeColor[2]));

    // Blend for all pixels
    cv::Mat blendedB, blendedG, blendedR;
    cv::addWeighted(channels[0], alpha, overlayB, 1.0f - alpha, 0.0, blendedB);
    cv::addWeighted(channels[1], alpha, overlayG, 1.0f - alpha, 0.0, blendedG);
    cv::addWeighted(channels[2], alpha, overlayR, 1.0f - alpha, 0.0, blendedR);

    // Copy only where mask != 0
    blendedB.copyTo(channels[0], binaryMask);
    blendedG.copyTo(channels[1], binaryMask);
    blendedR.copyTo(channels[2], binaryMask);

    // Merge back
    cv::merge(channels, result);

    for (auto & point : clicksVec) {
      circle(result,Point2f(point.x,point.y),20,Scalar(0,0,0,255),-1);
    }

    return result;
  }



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
    speedSam = new SpeedSam(parent->SAM_encoder_path.toString(), parent->SAM_decoder_path.toString());
  }


  void AccessSAM::embed_SAM_tiles() {
    load_model();

    std::deque<SAMTile *> embedQueue;
    for (auto &st: tiles) {
      embedQueue.push_back(st);
    }
    size_t nElementsPerChannel = parent->SAMTileSize * parent->SAMTileSize;
    auto comp = parent->composites[0];

    SAMTile *tileToFree = nullptr;

    while (!embedQueue.empty()) {
      std::sort(embedQueue.begin(), embedQueue.end(), tile_compare);
      auto tile = embedQueue.front();
      embedQueue.pop_front();

      tile->get_tile_data(comp, parent->SAMTileSize / parent->tileSize);
      cudaMalloc(&tile->rawBuffer, nElementsPerChannel * 3 * sizeof(float));
      tile->make_raw_buffer(tile->rawBuffer);

      //cudaMalloc(&tile->feats_0_data_d_, nElementsPerChannel * 2 * sizeof(float));
      cudaMalloc(&tile->feats_1_data_d_, nElementsPerChannel * sizeof(float));
      //cudaMalloc(&tile->embed_data_d_, nElementsPerChannel * sizeof(float));

      /*the more intuitive way to do this is to use an 'input consumed' event and then free the raw buffer as soon as that
       * turns true rather than wait for the next iteration, sync the whole stream and then free the input. The reason
       * for doing it this way is to control what gets run next after each run is completed while allowing as much parallelism
       * between runs as possible. EnqueuV2 launches the kernel on its own thread, meanwhile the nex tile can prep its data
       * but if the priority of the tiles changes, theres still a chance between iterations for that to take effect. If we
       * enqueued them all at once, this wouldnt be the case. If we block for input consumed, we dont allow parallelism.
       */
      CHECK_CUDA_ERROR(cudaStreamSynchronize(speedSam->mImageEncoder->mCudaStream));
      if (tileToFree) {
        tile->embeddingComplete = true;
        cudaFree(tileToFree->rawBuffer);
      }

      std::vector buffer{tile->rawBuffer, tile->feats_1_data_d_};
      speedSam->mImageEncoder->mContext->enqueueV2(buffer.data(), speedSam->mImageEncoder->mCudaStream, nullptr);
      cudaEventRecord(tile->embeddingCompleteCudaEvent,speedSam->mImageEncoder->mCudaStream);


      tileToFree = tile;

    }

    //free last tile
    if (tileToFree) {
      cudaFree(tileToFree->rawBuffer);
    }

  }


  void AccessSAM::create_segmentation(std::vector<Point3f> &_clicks, int _segID) {
    cudaSetDevice(parent->compositorCudaDevice);

    // Set the optimization profile
    speedSam->mMaskDecoder->mContext->setOptimizationProfileAsync(0, speedSam->mMaskDecoder->mCudaStream);

    while (!segmentProcessQ.empty()) {
      //clear the queue. the only thing that could be in here at this point is from debug
      segmentProcessQ.pop();
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
      segmentProcessQ.push(tile);
    }

    process_segmentation_Q(_segID);
  }

  void AccessSAM::process_segmentation_Q(int _segID) {
    while (!segmentProcessQ.empty()) {
      //get front tile
      auto tile = segmentProcessQ.front();
      segmentProcessQ.pop();

      //skip tile if its already been processed. This happens because it gets added once if it has clicks
      //and can then be subsequently added by neighbors if there is mask > 0 in overlapping regions
      if (tile->segmentations.find(_segID) != tile->segmentations.end()) {
        continue;
      }

      //ensure the tiles embedding has completed
      cudaEventSynchronize(tile->embeddingCompleteCudaEvent);

      //process the segmentation
      tile->run_segmentation(_segID);
    }
  }



  std::vector<int> AccessSAM::get_tiles_covering_point(const Point2f &_point, int _stride) {
    std::vector<int> out;

    auto comp = parent->composites[0];
    auto rootTile = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto myTile = comp->imagePyramid->level[0]->getIJ(_point);
    auto locInTile4 = _point - Point2f((float) parent->tileSize * myTile.x, (float) parent->tileSize * myTile.y);
    if (locInTile4.x < 0 || locInTile4.y < 0 || locInTile4.y > 256 || locInTile4.x > 256) {
      int k = 0;
    }

    auto primarySAMTileID = get_tile_id(myTile, 0);
    auto primarySAMTile = tiles[primarySAMTileID];
    auto locInTile12 = _point - Point2f((float) parent->tileSize * tiles[primarySAMTileID]->location.x,
                                        (float) parent->tileSize * tiles[primarySAMTileID]->location.y);
    if (locInTile12.x > 1024 || locInTile12.x < 0 || locInTile12.y > 1024 || locInTile12.y < 0) {
      throw std::exception();
    }

    out.push_back(primarySAMTileID);

    auto diff = myTile - rootTile;
    Point2i locInTile(diff.x % 3, diff.y % 3);


    if (locInTile.x == 0) {
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
    if (locInTile.y == 0) {
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
    if (locInTile.x == 0 && locInTile.y == 0) {
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


  void AccessSAM::push_mask_for_display(Point2i _tile, unsigned int _componentIndex, const Mat &_mask, int _segID) {
    auto pyrBase = parent->composites[_componentIndex]->imagePyramid->level[0];
    TileObj &tileObj = pyrBase->getTile(_tile.x, _tile.y);
    tileObj.SAMMasks[_segID] = _mask;
  }
}
