//
// Created by cooper maira on 7/25/25.
//

#include "AccessSAM.h"

namespace pathCam {
  SAMTile::SAMTile(int _ID, Point2i _location, unsigned int _size) : ID(_ID), location(_location), size(_size) {
    noncontiguousWrapper = cuda::GpuMat(size, size,CV_8UC4, Scalar(0, 0, 0, 0));
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

  void SAMTile::make_raw_buffer(char *_buffer) {
    //convert to F32, normalize for imagenet. keep in mind the mat is currently in BGRA
    noncontiguousWrapper.convertTo(noncontiguousWrapper,CV_32F);
    cuda::divide(noncontiguousWrapper, Scalar(255, 255, 255, 255), noncontiguousWrapper);
    cuda::subtract(noncontiguousWrapper, Scalar(0.406, 0.456, 0.485, 0), noncontiguousWrapper);
    cuda::divide(noncontiguousWrapper, Scalar(0.225, 0.224, 0.229, 1), noncontiguousWrapper);

    std::vector<cuda::GpuMat> split_channels;
    cuda::split(noncontiguousWrapper, split_channels);

    size_t nBytesPerChannel = 4 * noncontiguousWrapper.cols * noncontiguousWrapper.rows;
    //cudaMalloc(&rawBuffer, nBytesPerChannel * 3);

    for (int i = 0; i < 3; ++i) {
      cudaMemcpy2D(_buffer + i * nBytesPerChannel,
                   noncontiguousWrapper.cols * sizeof(float),
                   split_channels[2 - i].data,
                   noncontiguousWrapper.step,
                   noncontiguousWrapper.cols * sizeof(float),
                   noncontiguousWrapper.rows,
                   cudaMemcpyDeviceToDevice);
    }

    noncontiguousWrapper.release();

  }


  void AccessSAM::initialize() {
    int interval = (parent->SAMTileSize / parent->tileSize);
    assert(interval % 2 == 0);
    auto comp = parent->composites[0];

    auto ul = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto lr = comp->imagePyramid->level[0]->getIJ(Point2f(comp->max_offset.x, comp->max_offset.y));


    int id = 0;
    int numSAMTiles = ceil((lr.x - ul.x + 1) / 3) * ceil((lr.y - ul.y + 1) / 3);
    size_t nBytesPerImage = 4 * 3 * parent->SAMTileSize * parent->SAMTileSize;
    cudaMalloc(&batchImageEmbedBuffer,numSAMTiles * nBytesPerImage);

    int yTileCount = 0;
    for (int y = ul.y; y <= lr.y; ++y) {
      if ((yTileCount - 1) % (interval - 1) == 0) {
        int xTileCount = 0;

        for (int x = ul.x; x <= lr.x; ++x) {
          if ((xTileCount - 1) % (interval - 1) == 0) {
            auto st = new SAMTile(id++, {x, y}, parent->SAMTileSize);
            tiles.push_back(st);

            //add subtiles
            for (int xx = 0; xx < interval; ++xx) {
              for (int yy = 0; yy < interval; ++yy) {
                auto gMat = comp->imagePyramid->level[0]->getTile(x + xx, y + yy);
                st->set_component_tile({x + xx - 1, y + yy - 1}, {xx, yy}, gMat);
              }
            }
            st->make_raw_buffer(batchImageEmbedBuffer + id * nBytesPerImage);

            //add links to neighbors
            if (xTileCount > 0) {
              auto brotherX = tiles[get_tile_id({x - interval + 1, y}, comp->componentIndex)];
              std::vector<Point2i> temp;
              for (int xx = 0; xx < interval; ++xx) {
                temp.emplace_back(x + xx, y);
              }
              tiles.back()->neighbors.push_back({brotherX, temp});
              brotherX->neighbors.push_back({tiles.back(), temp});
            }
            if (yTileCount > 0) {
              auto brotherY = tiles[get_tile_id({x, y - interval + 1}, comp->componentIndex)];
              std::vector<Point2i> temp;
              for (int yy = 0; yy < interval; ++yy) {
                temp.emplace_back(x, y + yy);
              }
              tiles.back()->neighbors.push_back({brotherY, temp});
              brotherY->neighbors.push_back({tiles.back(), temp});
            }
            if (yTileCount > 0 && xTileCount > 0) {
              auto brotherXY = tiles[get_tile_id({x - interval + 1, y - interval + 1}, comp->componentIndex)];
              tiles.back()->neighbors.push_back({brotherXY, {{x, y}}});
              brotherXY->neighbors.push_back({tiles.back(), {{x, y}}});
              int k = 0;
            }
          }
          ++xTileCount;
        }
      }
      ++yTileCount;
    }


  }

  void AccessSAM::load_model() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sam2_embedder");
    Ort::SessionOptions session_options;
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, parent->compositorCudaDevice);

    // Load the ONNX model
    session = new Ort::Session(env, parent->SAM_encoder_path.toString().c_str(), session_options);
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
}
