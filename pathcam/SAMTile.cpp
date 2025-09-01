//
// Created by cooper maira on 8/25/25.
//

#include "pathCam.h"


namespace pathCam {
  using namespace nvinfer1;

  SAMTile::SAMTile(int _ID, Point2i _location, AccessSAM *_as, unsigned _componentIndex, unsigned _size) : ID(_ID),
    location(_location),
    componentIndex(_componentIndex),
    size(_size),
    as(_as) {
    noncontiguousWrapper = cuda::GpuMat(size, size,CV_8UC4, Scalar(0, 0, 0, 0));
    maskInput = nullptr;
    outputMask = nullptr;
    confidence = nullptr;
    hasMaskInputGPU = nullptr;
    inputMaskMat = cuda::GpuMat(256, 256,CV_32FC1, Scalar(0));
    cudaEventCreate(&embeddingCompleteCudaEvent);
  }

  cuda::GpuMat SAMTile::run_segmentation(int _segmentationID) {
    if (clicksVec.size() > 10) {
      int k = 0;
    }
    if (clicksVec.empty() && !clicksFromMasks.empty()/*!hasMaskInput*/) { return cuda::GpuMat(); }

    bool letMaskShrink = false;
    for (auto point : clicksVec) {
      if (point.z == 0) {
        letMaskShrink = true;
        break;
      }
    }
    std::vector<Point3f> clicksForCurrentRun = clicksVec;
    clicksVec.clear();

    if (!clicksFromMasks.empty() && !runIsRepeat) {
      int i = 0;
      while (clicksForCurrentRun.size() < 10 && i < clicksFromMasks.size()) {
        clicksForCurrentRun.push_back(clicksFromMasks[i++]);
      }
    }

    if (clicksForCurrentRun.empty()) {
      //give dummy input to avoid input requirements, -1 means ignore
      clicksForCurrentRun.push_back({0, 0, -1});
    }

    cuda::GpuMat temp;
    cuda::compare(inputMaskMat, Scalar(0), temp, CMP_GT);
    if (cuda::countNonZero(temp)) {
      hasMaskInput = true;
    }

    //get embeddings
    AccessSAM::get_clicks_embedding(clicksForCurrentRun, clicksGPU, clickLabelsGPU);

    if (!hasMaskInputGPU) {
      cudaMalloc(&hasMaskInputGPU, sizeof(float));
    }
    auto maskInputVal = static_cast<float>(hasMaskInput); // 1.0f or 0.0f
    cudaMemcpy(hasMaskInputGPU, &maskInputVal, sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemset(hasMaskInputGPU, 0.f, sizeof(float));

    //allocate if not already allocated
    if (!outputMask) {
      cudaMalloc(&outputMask, sizeof(float) * 256 * 256 * 3);
    }
    if (!confidence) {
      cudaMalloc(&confidence, 3 * sizeof(float));
    }
    if (!maskInput) {
      cudaMalloc(&maskInput, sizeof(float) * 256 * 256);
    }

    // if (hasMaskInput) {
    //   Mat inputMaskMatHost;
    //
    //   cudaMemcpy2D(maskInput, sizeof(float) * 256, inputMaskMat.data, inputMaskMat.step, sizeof(float) * 256, 256,
    //                cudaMemcpyDeviceToDevice);
    // }

    if (clicksForCurrentRun.size() > 10) {
      int k = 0;
    }
    //set binding dimension for dynamic input (clicks)
    as->decoderCtx->setInputShape("point_coords", Dims3{1, static_cast<int>(clicksForCurrentRun.size()), 2});
    as->decoderCtx->setInputShape("point_labels", Dims2{1, static_cast<int>(clicksForCurrentRun.size())});

    //set input and output addresses
    as->decoderCtx->setInputTensorAddress("image_embed", image_embed);
    as->decoderCtx->setInputTensorAddress("high_res_feats_0", high_res_feats_0);
    as->decoderCtx->setInputTensorAddress("high_res_feats_1", high_res_feats_1);
    as->decoderCtx->setInputTensorAddress("point_coords", clicksGPU);
    as->decoderCtx->setInputTensorAddress("point_labels", clickLabelsGPU);
    as->decoderCtx->setInputTensorAddress("mask_input", maskInput);
    as->decoderCtx->setInputTensorAddress("has_mask_input", hasMaskInputGPU);

    as->decoderCtx->setOutputTensorAddress("masks", outputMask);
    as->decoderCtx->setOutputTensorAddress("iou_predictions", confidence);

    as->decoderCtx->enqueueV3(as->decoderStream);
    CHECK_CUDA(cudaStreamSynchronize(as->decoderStream));


    cudaFree(clicksGPU);
    cudaFree(clickLabelsGPU);
    //clicksVec.clear(); TODO
    hasMaskInput = false;

    cuda::GpuMat output(256, 256,CV_32FC1, outputMask);

    if (ID == -1) {
      return output;
    }

    cuda::GpuMat outputBinary;
    cuda::threshold(output, outputBinary, 0, 255, THRESH_BINARY);
    outputBinary.convertTo(outputBinary,CV_8U);

    auto val = cuda::countNonZero(outputBinary);
    if (val > 0.20 * outputBinary.rows * outputBinary.cols) {
      int interval = 256 * 256 / size;

      //distribute mask information as input to neighbors
      int scale = as->parent->SAMTileSize / as->parent->tileSize;
      for (auto &neighbor: neighbors) {
        int roix, roiy, roixLen, roiyLen;
        if ((neighbor.second[1].x - location.x) == 0) {
          roix = 0;
          roixLen = as->parent->tileSize / scale;
          roiy = 0;
          roiyLen = as->parent->SAMTileSize / scale;
        } else if ((neighbor.second[1].x - location.x) == 3) {
          roix = (as->parent->SAMTileSize - as->parent->tileSize) / scale;
          roixLen = as->parent->tileSize / scale;
          roiy = 0;
          roiyLen = as->parent->SAMTileSize / scale;
        } else if (neighbor.second[1].y - location.y == 0) {
          roix = 0;
          roixLen = as->parent->SAMTileSize / scale;
          roiy = 0;
          roiyLen = as->parent->tileSize / scale;
        } else if (neighbor.second[1].y - location.y == 3) {
          roix = 0;
          roixLen = as->parent->SAMTileSize / scale;
          roiy = (as->parent->SAMTileSize - as->parent->tileSize) / scale;
          roiyLen = as->parent->tileSize / scale;
        } else {
          throw std::exception();
        }
        Rect roi(roix, roiy, roixLen, roiyLen);

        //take one high logit and one low logit as clicks from mask
        Point highloc, lowloc;
        double highVal, lowVal;
        //cuda::minMaxLoc(output(roi),&lowVal,&highVal,&lowloc,&highloc);


        auto seeds = AccessSAM::pickSeedsFromLogits_v2(output(roi), outputBinary(roi), 9);

        highloc = seeds.inlier;
        lowloc = seeds.outlier;

        highVal = seeds.inlierScore;
        lowVal = seeds.outlierScore;

        highloc.x *= scale;
        lowloc.x *= scale;

        highloc.y *= scale;
        lowloc.y *= scale;

        //map click to neighbor space
        if (neighbor.first->location.x == location.x) {
          //neighbor is brotherY
          if (neighbor.first->location.y == location.y) {
            //edge case
            continue;
          } else if (neighbor.first->location.y < location.y) {
            highloc.y += as->parent->SAMTileSize - (int) as->parent->tileSize;
            lowloc.y += as->parent->SAMTileSize - (int) as->parent->tileSize;
          }
        } else if (neighbor.first->location.x < location.x) {
          //brotherX
          highloc.y += as->parent->SAMTileSize - (int) as->parent->tileSize;
          lowloc.y += as->parent->SAMTileSize - (int) as->parent->tileSize;
        }

        if (highVal > 0) {
          neighbor.first->clicksFromMasks.push_back(Point3f(highloc.x, highloc.y, 1.f));

          if (lowVal < 0) {
            neighbor.first->clicksFromMasks.push_back(Point3f(lowloc.x, lowloc.y, 0.f));
          }
        }

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

        // //proof of seed clicks falling where they should in mask
        // cuda::GpuMat tempD;
        // Mat tempH;
        //
        // cuda::threshold(neighbor.first->inputMaskMat, tempD, 0, 255, THRESH_BINARY);
        // tempD.convertTo(tempD,CV_8U);
        // cuda::resize(tempD, tempD, {int(size), int(size)});
        // tempD.download(tempH);
        // for (auto p: neighbor.first->clicksFromMasks) {
        //   circle(tempH, Point(p.x, p.y), 50, p.z > .5 ? Scalar(50) : Scalar(200), -1);
        // }
        // imwrite("/media/max/Data/pathcam_SAM/"+std::to_string(neighbor.first->location.x)+"_"+std::to_string(neighbor.first->location.y)+"_input.png", tempH);
        // int k = 0;

        if (neighbor.first->segmentations.find(_segmentationID) == neighbor.first->segmentations.end()
            && !clicksFromMasks.empty()/*cuda::countNonZero(neighbor.first->inputMaskMat*/) {
          neighbor.first->hasMaskInput = true;
          as->segmentProcessQ.push_back(neighbor.first);
        }
      }
    }

    //send my own mask info for display
    cuda::resize(output, output, {int(size), int(size)});
    cuda::threshold(output, output, 0.0, 255.0, THRESH_BINARY);
    output.convertTo(output,CV_8U);
    segmentations[_segmentationID] = output;

    // Mat hostOutput;
    // output.download(hostOutput);
    //
    // Mat ans = debug_draw_tile_with_clicks_and_mask(ncwStoreLocal, hostOutput, Scalar(0, 180, 150, 255), 0.5);
    // for (auto p : clicksForCurrentRun) {
    //   Scalar color = p.z > 0.5 ? Scalar(0,200,0,255) : Scalar(0,0,200,255);
    //   circle(ans,Point(p.x,p.y),30,color,-1);
    // }
    // imwrite("/media/max/Data/pathcam_SAM/" + std::to_string(location.x) + "_" + std::to_string(location.y) + "_output.png",
    //         ans);
    // int k = 0;


    //part out the mask to tiles
    int tileSize = as->parent->tileSize;
    for (auto &tile: componentTiles) {
      if (tile.first.x == 3 || tile.first.y == 3) {
        auto transform = as->get_transformation_to_display(this, tile.first);
        int k = 0;
      }
      Rect maskRoi(tile.first.x * tileSize, tile.first.y * tileSize, tileSize, tileSize);
      if (cuda::countNonZero(output(maskRoi))) {
        as->push_mask_for_display(tile.second, componentIndex, output(maskRoi), _segmentationID, !letMaskShrink);
      }
    }
    runIsRepeat = false;
    canRun = false;

    return output;
  }


  void SAMTile::get_tile_data(CompositeVoronoi *_comp, unsigned int _interval) {
    for (int xx = 0; xx < _interval; ++xx) {
      for (int yy = 0; yy < _interval; ++yy) {
        auto gMat = _comp->imagePyramid->level[0]->getTile(location.x + xx, location.y + yy);
        set_component_tile({location.x + xx, location.y + yy}, {xx, yy}, gMat.image);
      }
    }
    // //debug
    // noncontiguousWrapper.download(ncwStoreLocal);
  }


  void SAMTile::set_component_tile(Point2i _tileID, Point2i _subLocation, cuda::GpuMat &_tileMat) {
    componentTiles.emplace_back(_subLocation, _tileID);
    Rect ROI(_subLocation.x * _tileMat.cols, _subLocation.y * _tileMat.rows, _tileMat.cols, _tileMat.rows);
    try {
      _tileMat.copyTo(noncontiguousWrapper(ROI));
    } catch (...) {
      throw std::exception();
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

    if (!rawBuffer && !_buffer) {
      //_buffer wasnt passed so were writing to member rawBuffer, but its not allocated yet
      cudaMalloc(&rawBuffer, nBytesPerChannel * 3);
    }

    for (int i = 0; i < 3; ++i) {
      void *dst;

      if (_buffer) {
        dst = static_cast<char *>(_buffer) + i * nBytesPerChannel;
      } else {
        dst = static_cast<char *>(rawBuffer) + i * nBytesPerChannel;
      }

      CHECK_CUDA(cudaMemcpy2D(dst,
        split_channels[2 - i].cols * sizeof(float),
        split_channels[2 - i].data,
        split_channels[2 - i].step,
        split_channels[2 - i].cols * sizeof(float),
        split_channels[2 - i].rows,
        cudaMemcpyDeviceToDevice));
    }
    noncontiguousWrapper.release();
  }


  void SAMTile::increase_embed_priority() {
    priority = 0;
    for (auto &p: neighbors) {
      p.first->priority = 1;
    }
  }


  void SAMTile::embed_tile_with_engine(nvinfer1::IExecutionContext *_encoderCtx) {
    cudaMalloc(&high_res_feats_0, 32 * 256 * 256 * sizeof(float));
    cudaMalloc(&high_res_feats_1, 64 * 128 * 128 * sizeof(float));
    cudaMalloc(&image_embed, 256 * 64 * 64 * sizeof(float));

    _encoderCtx->setInputTensorAddress("image", rawBuffer);
    _encoderCtx->setOutputTensorAddress("high_res_feats_0", high_res_feats_0);
    _encoderCtx->setOutputTensorAddress("high_res_feats_1", high_res_feats_1);
    _encoderCtx->setOutputTensorAddress("image_embed", image_embed);

    _encoderCtx->enqueueV3(as->decoderStream);
  }


  Mat SAMTile::debug_draw_tile_with_clicks_and_mask(const cv::Mat &bgraImage, const cv::Mat &binaryMask,
                                                    const cv::Scalar &shadeColor, float alpha, int _segmentationID) {
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

    for (auto &point: clicksVec) {
      circle(result, Point2f(point.x, point.y), 20, Scalar(0, 0, 0, 255), -1);
    }

    return result;
  }
}
