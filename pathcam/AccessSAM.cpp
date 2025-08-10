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


    Mat test;
    output.download(test);
    Mat binaryMask;
    compare(test, 0, binaryMask, cv::CMP_GT);
    binaryMask.convertTo(test,CV_8U);
    imwrite("/media/max/Data/pathcam_SAM/maskoutput_runseg.png", test);
    int k = 0;


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

      //sie.Infer(buffer);

      tileToFree = tile;

      if (tile->ID == 83) {
        std::vector<Point3f> clicksVec(10);
        clicksVec[0] = {337, 380, 1};
        clicksVec[1] = {410, 373, 1};
        clicksVec[2] = {65, 514, 1};
        clicksVec[3] = {66, 734, 1};
        clicksVec[4] = {457, 613, 1};
        clicksVec[5] = {314, 822, 1};
        clicksVec[6] = {312, 472, 0};
        clicksVec[7] = {193, 650, 0};
        clicksVec[8] = {212, 726, 0};
        clicksVec[9] = {500, 395, 0};
        tile->clicksVec = clicksVec;
        tile->run_segmentation(0);
      }
    }

    //free last tile
    if (tileToFree) {
      cudaFree(tileToFree->rawBuffer);
    }
  }

  void AccessSAM::push_mask_for_display(Point2i _tile, unsigned int _componentIndex, const Mat& _mask, int _segID) {
    auto pyrBase = parent->composites[_componentIndex]->imagePyramid->level[0];
    TileObj &tileObj = pyrBase->getTile(_tile.x,_tile.y);
    tileObj.SAMMasks[_segID] = _mask;
  }



  int AccessSAM::get_tile_id(Point2i _location, unsigned int _componentIndex) const {
    auto comp = parent->composites[_componentIndex];
    auto rootOffset = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto maxOffset = comp->imagePyramid->level[0]->getIJ(Point2f(comp->max_offset.x, comp->max_offset.y));
    auto pt1 = (_location.y - rootOffset.y) / 3;
    auto pt2 = (maxOffset.x - rootOffset.x) / 3;
    auto pt3 = (_location.x - rootOffset.x) / 3;
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
}
