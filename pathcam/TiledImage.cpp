//
//  TiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/6/24.
//

#include "pathCam.h"

void (*TileObj::destroyPreferredObj)(void *) = nullptr;

template<typename T>
cv::Rect_<T> rect_mult(cv::Rect_<T> r, T s) {
  return cv::Rect_<T>(r.x * s, r.y * s, r.width * s, r.height * s);
}

TiledImage::TiledImage(std::shared_ptr<MRTiledImage> parent, unsigned int _tile_size, unsigned int _logic_size,
                       int levelWithinPyramid) : tile_size(_tile_size),
                                                 logic_size(_logic_size),
                                                 logicRatio((float) _tile_size / (float) _logic_size),
                                                 levelWithinPyramid(levelWithinPyramid),
                                                 tiles(-1024 * logicRatio, 1024 * logicRatio, -1024 * logicRatio,
                                                       1024 * logicRatio),
                                                 parent(parent) {
  if (tile_size == 0) {
    tile_size = parent->tile_size;
  }
};


void TiledImage::insertMatAtBase(cv::Mat image_in, cv::Rect_<float> box, std::vector<Point2i> retileIndices) {
  unsigned int width = image_in.cols;
  unsigned int height = image_in.rows;
  float scale = ((float) tile_size / (float) logic_size);

  assert(int(box.width*scale) == width &&
    int(box.height*scale) == height);

  bounds = bounds | box;

  for (auto &tile: retileIndices) {
    int x = tile.x;
    int y = tile.y;

    makeTile(x, y);

    cv::Rect_<float> tile_box = cv::Rect_<float>(x * float(logic_size),
                                                 y * float(logic_size),
                                                 logic_size,
                                                 logic_size);

    cv::Rect_<float> image_box = tile_box & box;

    Point2f offset = box.tl(); //equivalent of root offset

    matToImage4Channel(image_in, x, y, offset * scale,
                       rect_mult<float>(image_box, scale),
                       rect_mult<float>(tile_box, scale));
  }
}


void TiledImage::matToImage4Channel(const cv::Mat &mat, int x, int y, Point2f rootOffset, cv::Rect_<float> image_box,
                                    cv::Rect_<float> tile_box) {
  cv::Rect ROIrect((int) (image_box.x - rootOffset.x),
                   (int) (image_box.y - rootOffset.y),
                   (int) image_box.width,
                   (int) image_box.height);
  if (ROIrect.width * ROIrect.height > 0) {
    cv::Mat matROI = mat(ROIrect);
    auto tile = getTile(x, y);
    //prepare to tile upward
    auto tileROI = cv::Rect(image_box.x - tile_box.x, image_box.y - tile_box.y, matROI.cols,
                            matROI.rows);
    matROI.copyTo(tile->image(tileROI));

    tileUpwards(Point2i(x, y), tile_box, tile, Rect(0, 0, tile_size, tile_size));
  }
}


void TiledImage::matToImage2(const cv::Mat mat, cv::Mat image,
                             Point2f offset, cv::Rect_<float> image_box,
                             cv::Rect_<float> tile_box) {
  assert(mat.type() == CV_8UC4);

  cv::Rect ROIrect((int) (image_box.x - offset.x),
                   (int) (image_box.y - offset.y),
                   (int) image_box.width,
                   (int) image_box.height);

  cv::Mat ROI = mat(ROIrect);
  if (ROI.cols * ROI.rows > 0) {
    ROI.copyTo(image(cv::Rect(image_box.x - tile_box.x,
                              image_box.y - tile_box.y,
                              ROI.cols,
                              ROI.rows)));
  }
}

void TiledImage::resetEdges(Point2i topLeft, Point2i bottomRight) {
  auto tL = getIJ(topLeft);
  auto bR = getIJ(bottomRight);
  for (int x = tL.x; x <= bR.x; x++) {
    for (int y = tL.y; y <= bR.y; y++) {
      //get tile

      if (tiles(x, y) && tiles(x, y)->image.data) {
        tiles(x, y)->image.setTo(Scalar(0, 0, 0, 0));

        Point_ loc = Point2i(x, y);
        Rect levelRegion = cv::Rect(x * tile_size, y * tile_size, tile_size, tile_size);
        tileUpwards(loc, levelRegion, getTile(x, y), Rect(0, 0, tile_size, tile_size));
      }
      /*
      //create vector of all 4 channes R, G, B and alpha
      std::vector<Mat> individualChannels(4);
      //split the tile into the 4 channels
      split(tile, individualChannels);
      //if the number of nonzero alphachannels is not the same as the number of channels, the tile needs to be removed
      if (countNonZero(individualChannels[3]) != tile.rows * tile.cols)
      {
        //remove the tile and replce with empty mat
        //tiles(x,y)->release();
        //*tiles(x,y) = Mat(tile_size, tile_size, CV_8UC4, Scalar(0, 0, 0, 0));
        *tiles(x,y) = Mat::zeros(tile_size, tile_size, CV_8UC4);
        //update the image pyramid
        Point_ loc = Point2i(x, y);
        Rect levelRegion = cv::Rect(x * tile_size, y * tile_size, tile_size,tile_size);
        tileUpwards(loc, levelRegion, *tiles(x,y));
      }
       */
    }
  }
}


void TiledImage::matToImage(const cv::Mat &mat, cv::Mat *image,
                            Point2f offset, cv::Rect_<float> image_box,
                            cv::Rect_<float> tile_box) {
  assert(mat.type() == CV_8UC3);

  cv::Rect ROIrect((int) (image_box.x - offset.x),
                   (int) (image_box.y - offset.y),
                   (int) image_box.width,
                   (int) image_box.height);

  cv::Mat ROI = mat(ROIrect);
  ROI.copyTo((*image)(cv::Rect(image_box.x - tile_box.x,
                               image_box.y - tile_box.y,
                               ROI.cols,
                               ROI.rows)));
}

void TiledImage::insertMat(cv::Mat image_in, cv::Rect_<float> box) {
  unsigned int width = image_in.cols;
  unsigned int height = image_in.rows;
  float scale = ((float) tile_size / (float) logic_size);

  assert(int(box.width*scale) == width &&
    int(box.height*scale) == height);

  bounds = bounds | box;

  Point2f top_left = box.tl();
  Point2f bottom_right = box.br();
  bottom_right.x -= 1;
  bottom_right.y -= 1;


  for (int i = getIJ(top_left).x; i <= getIJ(bottom_right).x; i++) {
    for (int j = getIJ(top_left).y; j <= getIJ(bottom_right).y; j++) {
      makeTile(i, j);

      cv::Rect_<float> tile_box = cv::Rect_<float>(i * float(logic_size),
                                                   j * float(logic_size),
                                                   logic_size,
                                                   logic_size);

      cv::Rect_<float> image_box = tile_box & box;

      Point2f offset = box.tl();

      if (image_in.type() == CV_8UC4) {
        Mat &temp = tiles(i, j)->image;
        matToImage2(image_in, temp, offset * scale,
                    rect_mult<float>(image_box, scale),
                    rect_mult<float>(tile_box, scale));
      } else {
        throw std::invalid_argument("received wrong channel of matrix");
      }
    }
  }
}

std::vector<TileQuery> TiledImage::getTiles(cv::Rect_<float> box) {
  std::vector<TileQuery> box_tiles;

  Point2f top_left = box.tl();
  Point2f bottom_right = box.br();
  bottom_right.x--;
  bottom_right.y--;


  int iMin = std::max(getIJ(top_left).x, tiles.minX);
  int iMax = std::min(getIJ(bottom_right).x, tiles.minX + tiles.width - 1);
  int jMin = std::max(getIJ(top_left).y, tiles.minY);
  int jMax = std::min(getIJ(bottom_right).y, tiles.minY + tiles.height - 1);

  for (int i = iMin; i <= iMax; i++) {
    for (int j = jMin; j <= jMax; j++) {
      int x = i * (int) logic_size - box.x;
      int y = j * (int) logic_size - box.y;
      Rect_<float> rect = cv::Rect_<float>(x, y, logic_size, logic_size);
      if (tiles(i, j)) {
        box_tiles.emplace_back(getTile(i, j), i, j, rect);
      }
    }
  }

  return box_tiles;
}

void TiledImage::saveBaseTilesToDisk() {
  int minx = 0;
  int miny = 0;
  for (int x = tiles.minX; x < tiles.minX + tiles.width; x++) {
    for (int y = tiles.minY; y < tiles.minY + tiles.width; y++) {
      if (tiles(x, y) != nullptr) {
        if (x < minx) {
          minx = x;
        }
        if (y < miny) {
          miny = y;
        }
      }
    }
  }

  for (int x = tiles.minX; x < tiles.minX + tiles.width; x++) {
    for (int y = tiles.minY; y < tiles.minY + tiles.width; y++) {
      if (tiles(x, y) != nullptr) {
        assert(tile_size % 256 == 0);
        for (int i = 0; i < pow(tile_size / 256, 2); i++) {
          int xsubtile = i % (tile_size / 256);
          int ysubtile = i / (tile_size / 256);
          auto roi = Rect(256 * xsubtile, 256 * ysubtile, 256, 256);

          int xloc = (x - minx) * (int) tile_size + xsubtile * 256;
          int yloc = (y - miny) * (int) tile_size + ysubtile * 256;
          Mat &temp = (*tiles(x, y)).image;
          Mat gry;
          cvtColor(temp(roi), gry, COLOR_BGR2GRAY);
          if (countNonZero(gry) > 0.95 * 256 * 256) {
            std::string filename = std::to_string(xloc) + "x_" + std::to_string(yloc) + "y.png";
            cv::imwrite(filename, temp(roi));
          }
        }
      }
    }
  }
}

// #ifdef HAVE_OPENCV_CUDAARITHM
// void TiledImage::matToTile(const cuda::GpuMat &mat, const cuda::GpuMat &mask, int x, int y, Point2f rootOffset,
//                            Rect_<float> image_box, cv::Rect_<float> tile_box) {
//   Rect ROIrect((int) (image_box.x - rootOffset.x),
//                (int) (image_box.y - rootOffset.y),
//                (int) image_box.width,
//                (int) image_box.height);
//
//   if (ROIrect.width * ROIrect.height > 0) {
//     try {
//       cuda::GpuMat matROI;
//       Rect tileROI;
//
//       matROI = mat(ROIrect);
//
//       auto tileObject = getTile(x, y);
//       ++tileObject->updateCount;
//
//       tileROI = Rect(image_box.x - tile_box.x, image_box.y - tile_box.y, matROI.cols,
//                      matROI.rows);
//
//
//       tileObject->mutex.lock();
//       if (mask.data) {
//         matROI.copyTo(tileObject->image(tileROI), mask(ROIrect));
//       } else {
//         matROI.copyTo(tileObject->image(tileROI));
//       }
//       tileObject->newData = true;
//       tileObject->mutex.unlock();
//
//       assert(tiles(x, y)->image.rows == tile_size && tiles(x, y)->image.cols == tile_size);
//     } catch (cv::Exception &e) {
//       std::cout << "cv error in matToTile" << std::endl;
//       std::cout << e.what() << std::endl;
//       throw std::exception();
//     }
//     tileUpwards(Point2i(x, y), tile_box, getTile(x, y), Rect(0, 0, tile_size, tile_size));
//   }
// }

void TiledImage::matToTile(const cv::Mat &mat, const cv::Mat &mask, int x, int y, cv::Point2f rootOffset,
                           cv::Rect_<float> image_box, cv::Rect_<float> tile_box) {
  Rect ROIrect((int) (image_box.x - rootOffset.x), (int) (image_box.y - rootOffset.y),
               (int) image_box.width, (int) image_box.height);

  if (ROIrect.width * ROIrect.height > 0) {
    Mat matROI = mat(ROIrect);

    auto tileObject = getTile(x, y);
    if (auto imagePyramid = parent.lock()) {
      imagePyramid->liveTiles.insert({x, y});
    }else {
      throw std::runtime_error("failed to grab weak pointer parent in matToTile");
    }

    //profiling
    ++tileObject->updateCount;


    Rect tileROI = Rect(image_box.x - tile_box.x, image_box.y - tile_box.y, matROI.cols, matROI.rows);

    tileObject->mutex.lock();
    if (!mask.empty()) {
      matROI.copyTo(tileObject->image(tileROI), mask(ROIrect));
    } else {
      matROI.copyTo(tileObject->image(tileROI));
    }

    tileObject->newData = true;
    tileObject->mutex.unlock();

    assert(tiles(x, y)->image.rows == tile_size && tiles(x, y)->image.cols == tile_size);

    //parent->parent->push_pyramid_builder_Q(Point2i(x,y),parent->componentIndex);
    tileUpwards(Point2i(x, y), tile_box, getTile(x, y), Rect(0, 0, tile_size, tile_size));
  }
}

// void TiledImage::insertTilesAtBase(cuda::GpuMat &image_in, cuda::GpuMat &mask, cv::Rect_<float> box,
//                                    const std::vector<Point2i> &retileIndices) {
//   unsigned int width = image_in.cols;
//   unsigned int height = image_in.rows;
//
//   assert(int(box.width*logicRatio) == width &&
//     int(box.height*logicRatio) == height);
//
//   bounds = bounds | box;
//
//   for (auto &tile: retileIndices) {
//     try {
//       int x = tile.x;
//       int y = tile.y;
//
//       cv::Rect_<float> tile_box = cv::Rect_<float>(x * float(logic_size),
//                                                    y * float(logic_size),
//                                                    logic_size,
//                                                    logic_size);
//
//       cv::Rect_<float> image_box = tile_box & box;
//
//       Point2f offset = box.tl(); //equivalent of root offset
//
//
//       matToTile(image_in, mask, x, y, offset * logicRatio,
//                 rect_mult<float>(image_box, logicRatio),
//                 rect_mult<float>(tile_box, logicRatio));
//     } catch (cv::Exception &e) {
//       std::cout << "cv error in insertTilesAtBase" << std::endl;
//       std::cout << e.what() << std::endl;
//       throw std::exception();
//     }
//   }
// }

void TiledImage::insertTilesAtBase(cv::Mat &image_in, cv::Mat &mask, cv::Rect_<float> &box,
                                   std::vector<Point2i> &retileIndices) {
  unsigned int width = image_in.cols;
  unsigned int height = image_in.rows;

  assert(int(box.width*logicRatio) == width &&
    int(box.height*logicRatio) == height);

  bounds = bounds | box;

  for (auto &tile: retileIndices) {
    try {
      int x = tile.x;
      int y = tile.y;

      cv::Rect_<float> tile_box = cv::Rect_<float>(x * float(logic_size),
                                                   y * float(logic_size),
                                                   logic_size,
                                                   logic_size);

      cv::Rect_<float> image_box = tile_box & box;

      Point2f offset = box.tl(); //equivalent of root offset


      matToTile(image_in, mask, x, y, offset * logicRatio,
                rect_mult<float>(image_box, logicRatio),
                rect_mult<float>(tile_box, logicRatio));
    } catch (cv::Exception &e) {
      int k = 0;
    }
  }
}


void TiledImage::tileUpwards(Point2i myTileIndex, cv::Rect_<float> myLevelRegion, std::shared_ptr<TileObj> myTileObj,
                             Rect cvRoi,
                             int _segID) {
  //this function takes a tiles data at a lower level of the pyramid and resizes it into the tile directly above it in the pyramid
  if (auto imagePyramid = parent.lock()){
  try {
    //find appropriate region of upper level
    float xloc = myLevelRegion.x / 2.f;
    float yloc = myLevelRegion.y / 2.f;
    float width = myLevelRegion.width / 2.f;
    float height = myLevelRegion.height / 2.f;
    auto theirLevelRegion = cv::Rect_<float>(xloc, yloc, width, height);

    //find appropriate tiles
    Point2i theirTileIndex;
    theirTileIndex.x = myTileIndex.x < 0 ? (myTileIndex.x - 1) / 2 : myTileIndex.x / 2;
    theirTileIndex.y = myTileIndex.y < 0 ? (myTileIndex.y - 1) / 2 : myTileIndex.y / 2;

    //make theirLevelRegion relative to tile
    float theirTileRegionX = (long) theirLevelRegion.x % tile_size;
    theirTileRegionX = theirTileRegionX < 0 ? theirTileRegionX + tile_size : theirTileRegionX;
    float theirTileRegionY = (long) theirLevelRegion.y % tile_size;
    theirTileRegionY = theirTileRegionY < 0 ? theirTileRegionY + tile_size : theirTileRegionY;

    //calculate theirROI and grab tile
    cv::Rect theirROI(theirTileRegionX, theirTileRegionY, theirLevelRegion.width, theirLevelRegion.height);
    auto theirTileObj = imagePyramid->level[levelWithinPyramid + 1]->getTile(theirTileIndex.x, theirTileIndex.y);

    //resize self cv image into their cv image ROI
    auto newSize = Size(theirTileObj->image(theirROI).cols, theirTileObj->image(theirROI).rows);
    theirTileObj->mutex.lock();

    if (_segID < 0) {
      assert(
        theirTileObj->image(theirROI).rows == myTileObj->image(cvRoi).rows / 2 && theirTileObj->image(theirROI).cols ==
        myTileObj->image(cvRoi).cols / 2);
      resize(myTileObj->image(cvRoi), theirTileObj->image(theirROI), newSize);
      theirTileObj->newData = true;
    } else {
      if (theirTileObj->SAMMasks.find(_segID) == theirTileObj->SAMMasks.end()) {
        theirTileObj->SAMMasks[_segID] = {Mat(imagePyramid->tile_size, imagePyramid->tile_size,CV_8U, Scalar(0)), nullptr};
      }

      assert(theirTileObj->SAMMasks[_segID].first(theirROI).rows == myTileObj->SAMMasks[_segID].first(cvRoi).rows / 2
        && theirTileObj->SAMMasks[_segID].first(theirROI).cols == myTileObj->SAMMasks[_segID].first(cvRoi).cols / 2);

      resize(myTileObj->SAMMasks[_segID].first(cvRoi), theirTileObj->SAMMasks[_segID].first(theirROI), newSize);
      theirTileObj->newAnnoData = true;
    }

    theirTileObj->mutex.unlock();

    //continue up pyramid
    if (levelWithinPyramid + 1 < imagePyramid->level.size() - 1) {
      imagePyramid->level[levelWithinPyramid + 1]->tileUpwards(theirTileIndex, theirLevelRegion, theirTileObj, theirROI,_segID);
    }
  } catch (cv::Exception &e) {
    std::cout << "cv error in tileUpwards" << std::endl;
    std::cout << e.what() << std::endl;
    throw std::runtime_error("cv error in tileUpwards");
  }
}
}


std::shared_ptr<TileObj> TiledImage::getTile(int x, int y) {
  if (!tiles(x, y)) {
    tiles(x, y) = std::make_shared<TileObj>(tile_size, Point2i(x, y));
  }
  return tiles(x, y);
}

std::shared_ptr<TileObj> TiledImage::getTile(Point2i _index) {
  if (!tiles(_index.x, _index.y)) {
    tiles(_index.x, _index.y) = std::make_shared<TileObj>(tile_size, _index);
  }
  return tiles(_index.x, _index.y);
}


void TiledImage::makeTile(int x, int y) {
  if (!tiles(x, y)) {
    tiles(x, y) = std::make_shared<TileObj>(tile_size);
  }
}
