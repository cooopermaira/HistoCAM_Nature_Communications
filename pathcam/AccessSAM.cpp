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
  }

  void SAMTile::run_segmentation(int _segmentationID) {
    if (clicksVec.empty() && !hasMaskInput) { return; }

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

    //consolidate into a single buffer and run
    std::vector buffer{feats_1_data_d_, clicksGPU, clickLabelsGPU, inputMask, hasMaskInputGPU, outputMask, confidence};
    as->speedSam->mMaskDecoder->mContext->setOptimizationProfileAsync(0, as->speedSam->mMaskDecoder->mCudaStream);
    // Set the optimization profile
    as->speedSam->mMaskDecoder->mContext->setBindingDimensions(1, Dims3{1, int(clicksVec.size()), 2});
    // Set input dimensions for coordinates
    as->speedSam->mMaskDecoder->mContext->setBindingDimensions(2, Dims2{1, int(clicksVec.size())});
    // Set input dimensions for labels
    as->speedSam->mMaskDecoder->mContext->executeV2(buffer.data());

    cudaFree(clicksGPU);
    cudaFree(clickLabelsGPU);
    clicksVec.clear();
    hasMaskInput = false;

    cuda::GpuMat output(256, 256,CV_32FC1, outputMask);
    //cuda::threshold(output,output,0.0,1.0,THRESH_BINARY);
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
        cuda::max(output(myRoi), neighbor.first->inputMaskMat(theirRoi), neighbor.first->inputMaskMat(theirRoi));
      }
      if (neighbor.first->segmentations.find(_segmentationID) != neighbor.first->segmentations.end()
          && cuda::countNonZero(neighbor.first->inputMaskMat)) {
        neighbor.first->hasMaskInput - true;
        as->segmentProcessQ.push(neighbor.first);
      }
    }

    //send my own mask info for display
    cuda::resize(output, output, {int(size), int(size)});
    cuda::threshold(output, output, 0.0, 255.0, THRESH_BINARY);
    output.convertTo(output,CV_8U);
    Mat hostOutput;
    output.download(hostOutput);

    //part out the mask to tiles
    int tileSize = as->parent->tileSize;
    for (auto &tile: componentTiles) {
      Rect maskRoi(tile.first.x * tileSize, tile.first.y * tileSize, tileSize, tileSize);
      as->push_mask_for_display(tile.first,componentIndex,hostOutput(maskRoi),_segmentationID);
      //as->segmentationMasks[tile.second].emplace_back(_segmentationID, hostOutput(maskRoi));
    }


    // Mat test;
    // output.download(test);
    // Mat binaryMask;
    // compare(test, 0, binaryMask, cv::CMP_GT);
    // binaryMask.convertTo(test,CV_8U);
    // imwrite("/media/max/Data/pathcam_SAM/maskoutput_runseg.png", test);
    // int k = 0;


    //get mask back
    //distribute mask input to neighbors
    //let neighbor know not to rerun this tile
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


  void SAMTile::on_click() {
    priority = 0;
    for (auto &p: neighbors) {
      p.first->priority = 1;
    }
  }


  void AccessSAM::initialize() {
    cudaSetDevice(parent->compositorCudaDevice);
    int interval = (parent->SAMTileSize / parent->tileSize);
    assert(interval % 2 == 0);
    auto comp = parent->composites[0];

    auto ul = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto lr = comp->imagePyramid->level[0]->getIJ(Point2f(comp->max_offset.x, comp->max_offset.y));


    int id = 0;
    int numSAMTiles = ceil((lr.x - ul.x + 1) / (interval - 1)) * ceil((lr.y - ul.y + 1) / (interval - 1));
    //size_t nBytesPerImage = 4 * 3 * parent->SAMTileSize * parent->SAMTileSize;
    //cudaMalloc(&batchImageEmbedBuffer,numSAMTiles * nBytesPerImage);

    int yTileCount = 0;
    for (int y = ul.y; y <= lr.y; ++y) {
      if ((yTileCount - 1) % (interval - 1) == 0) {
        int xTileCount = 0;

        for (int x = ul.x; x <= lr.x; ++x) {
          if ((xTileCount - 1) % (interval - 1) == 0) {
            auto st = new SAMTile(id, {x, y}, this, 0,parent->SAMTileSize);
            tiles.push_back(st);
            ++id;

            //add links to neighbors
            if (xTileCount > 0) {
              auto brotherX = tiles[get_tile_id({x - interval + 1, y}, comp->componentIndex)];
              std::vector<Point2i> temp;
              for (int yy = 0; yy < interval; ++yy) {
                temp.emplace_back(x, y + yy);
              }
              tiles.back()->neighbors.emplace_back(brotherX, temp);
              brotherX->neighbors.emplace_back(tiles.back(), temp);
            }
            if (yTileCount > 0) {
              auto brotherY = tiles[get_tile_id({x, y - interval + 1}, comp->componentIndex)];
              std::vector<Point2i> temp;
              for (int xx = 0; xx < interval; ++xx) {
                temp.emplace_back(x + xx, y);
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

      CHECK_CUDA_ERROR(cudaStreamSynchronize(speedSam->mImageEncoder->mCudaStream));
      if (tileToFree) {
        cudaFree(tileToFree->rawBuffer);
      } {
        std::vector buffer{tile->rawBuffer, tile->feats_1_data_d_};
        speedSam->mImageEncoder->mContext->enqueueV2(buffer.data(), speedSam->mImageEncoder->mCudaStream, nullptr);
      }

      tileToFree = tile;

    }

    //free last tile
    if (tileToFree) {
      cudaFree(tileToFree->rawBuffer);
    }
  }


  void AccessSAM::create_segmentation(std::vector<Point3f> &_clicks, int _segID) {
    //a tile, its list of clicks, and if each click appears in other tiles as well
    std::map<int,std::vector<std::pair<Point3f,bool>>> tilesAndTheirClicks;

    for (auto &click : _clicks) {

      //get list of tiles this click falls in
      auto ans = get_tiles_covering_point(Point2f(click.x,click.y), parent->SAMTileSize - parent->tileSize);

      //for each tile, add this click to its list
      for (auto &tile : ans) {
        tilesAndTheirClicks[tile].push_back({click,ans.size() > 1});
      }
    }

    //choose which tiles should run with which clicks
    auto ans = choose_clicks_for_each_tile(tilesAndTheirClicks);
    int i = 0;
    for (auto & kv : ans) {
      auto tile = tiles[kv.first];

      for (auto &point : kv.second) {
        Point3f pointInTileSpace = point - Point3f(parent->tileSize * tile->location.x,parent->tileSize * tile->location.y,0);
        tile->clicksVec.push_back(pointInTileSpace);
        cv::circle(tile->ncwStoreLocal,Point2f(pointInTileSpace.x,pointInTileSpace.y),50,Scalar(0,0,0,255));
      }
      imwrite("/media/max/Data/pathcam_SAM/SAMTILE_with_clicks"+std::to_string(i++)+".png",tile->ncwStoreLocal);

      int k = 0;


    }


    //
  }


  std::vector<int> AccessSAM::get_tiles_covering_point(const Point2f &_point, int _stride) {
    std::vector<int> out;

    auto comp = parent->composites[0];
    auto rootTile = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto myTile = comp->imagePyramid->level[0]->getIJ(_point);

    auto primarySAMTile = get_tile_id(myTile,0);
    out.push_back(primarySAMTile);

    auto diff = myTile - rootTile;
    Point2i locInTile(diff.x % 3,diff.y % 3);


    if (locInTile.x == 0) {
      //pushback SAM tile (-1,0) from primary tile
      auto pointThatWillFallInTile = myTile - Point2i(_stride / parent->tileSize, 0);
      auto id = get_tile_id(pointThatWillFallInTile, 0);
      out.push_back(id);
    }
    if (locInTile.y == 0) {
      //pushback SAM tile (0,-1) from primary tile
      auto pointThatWillFallInTile = myTile - Point2i(0,_stride / parent->tileSize);
      auto id = get_tile_id(pointThatWillFallInTile, 0);
      out.push_back(id);
    }
    if (locInTile.x == 0 && locInTile.y == 0) {
      //pushback SAM tile (-1,-1) from primary tile
      auto pointThatWillFallInTile = myTile - Point2i(_stride / parent->tileSize, _stride / parent->tileSize);
      auto id = get_tile_id(pointThatWillFallInTile, 0);
      out.push_back(id);
    }

    return out;
  }


  std::map<int, std::vector<Point3f>> AccessSAM::choose_clicks_for_each_tile(const std::map<int, std::vector<std::pair<Point3f,bool>>>& clicksByTile, int cap)
{
    std::map<int, std::vector<Point3f>> result;

    for (const auto& [tileId, clicks] : clicksByTile)
    {
        // Split by uniqueness (non-overlap vs overlap) and label (yes/no)
        std::vector<cv::Point3f> Uyes, Uno, Oyes, Ono;
        Uyes.reserve(clicks.size()); Uno.reserve(clicks.size());
        Oyes.reserve(clicks.size()); Ono.reserve(clicks.size());

        for (const auto& c : clicks)
        {
            const Point3f& p = c.first;
            const bool isOverlap = c.second;
            const bool isYes = (p.z >= 0.5f);

            if (!isOverlap) (isYes ? Uyes : Uno).push_back(p);
            else            (isYes ? Oyes : Ono).push_back(p);
        }

        auto pick = [](std::vector<cv::Point3f>& src, int k, std::vector<cv::Point3f>& out) -> int {
            const int n = std::min<int>(k, static_cast<int>(src.size()));
            out.insert(out.end(), src.begin(), src.begin() + n);
            if (n > 0) src.erase(src.begin(), src.begin() + n);
            return n;
        };

        std::vector<Point3f> sel;
        sel.reserve(std::min<int>(cap, clicks.size()));

        if ((int)clicks.size() <= cap) {
            // If fewer than cap total, just take all (max possible per tile).
            for (const auto& c : clicks) sel.push_back(c.first);
            result.emplace(tileId, std::move(sel));
            continue;
        }

        const int targetYes = cap / 2;      // 5
        const int targetNo  = cap - targetYes; // 5
        int Ayes = 0, Ano = 0;

        // Step 1a: take uniques toward targets
        Ayes += pick(Uyes, targetYes, sel);
        Ano  += pick(Uno,  targetNo,  sel);
        int rem = cap - (Ayes + Ano);

        // Step 1b: still uniques left? fill while reducing imbalance
        while (rem > 0 && (!Uyes.empty() || !Uno.empty())) {
            const bool takeYes = (Ayes < targetYes && !Uyes.empty()) || Uno.empty();
            if (takeYes) { Ayes += pick(Uyes, 1, sel); }
            else         { Ano  += pick(Uno,  1, sel); }
            rem = cap - (Ayes + Ano);
        }

        // Step 2a: pull from overlaps to hit targets
        if (rem > 0) {
            const int needYes = std::max(0, targetYes - Ayes);
            const int needNo  = std::max(0, targetNo  - Ano);
            Ayes += pick(Oyes, needYes, sel);
            Ano  += pick(Ono,  needNo,  sel);
            rem = cap - (Ayes + Ano);
        }

        // Step 2b: fill remaining from whatever is left (still bias toward balance)
        while (rem > 0 && (!Oyes.empty() || !Ono.empty())) {
            const bool takeYes = (Ayes < Ano && !Oyes.empty()) || Ono.empty();
            if (takeYes) { Ayes += pick(Oyes, 1, sel); }
            else         { Ano  += pick(Ono,  1, sel); }
            rem = cap - (Ayes + Ano);
        }

        result.emplace(tileId, std::move(sel));
    }

    return result;
}


  int AccessSAM::get_tile_id(Point2i _tileIndexPoint, unsigned int _componentIndex) const {
    auto comp = parent->composites[_componentIndex];
    auto rootOffset = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto maxOffset = comp->imagePyramid->level[0]->getIJ(Point2f(comp->max_offset.x, comp->max_offset.y));
    int interval = parent->SAMTileSize / parent->tileSize - 1;
    auto pt1 = (_tileIndexPoint.y - rootOffset.y) / interval;
    auto pt2 = (maxOffset.x - rootOffset.x) / interval;
    auto pt3 = (_tileIndexPoint.x - rootOffset.x) / interval;
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



  void AccessSAM::push_mask_for_display(Point2i _tile, unsigned int _componentIndex, const Mat& _mask, int _segID) {
    auto pyrBase = parent->composites[_componentIndex]->imagePyramid->level[0];
    TileObj &tileObj = pyrBase->getTile(_tile.x,_tile.y);
    tileObj.SAMMasks[_segID] = _mask;
  }

}
