//
//  TiledImage.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/6/24.
//

#pragma once

#include "pathCam.h"
#include "MRTiledImage.h"

template<typename T>
inline cv::Rect_<T> RecMult(cv::Rect_<T> r, T scalar) {
  return cv::Rect_<T>(r.x * scalar, r.y * scalar, r.width * scalar, r.height * scalar);
}


template<typename T>
class Dense2DArray {
public:
  Dense2DArray(int minX = -512, int maxX = 512, int minY = -512, int maxY = 512)
    : minX(minX), minY(minY), width(maxX - minX + 1), height(maxY - minY + 1) {
    data.resize(width * height);
  }

  ~Dense2DArray() = default;

  T &operator()(int x, int y) {
    return data[getIndex(x, y)];
  }

  const T &operator()(int x, int y) const {
    return data[getIndex(x, y)];
  }

  int minX, minY, width, height;

private:
  std::vector<T> data;


  inline int64_t getIndex(int x, int y) const {
    assert(x >= minX && x < minX + width);
    assert(y >= minY && y < minY + height);
    return (x - minX) + (y - minY) * width;
  }
};


struct TileObj {
  enum {
    noCoverage = 0, partialCoverage, singleFrameCoverage
  };

  Point2i index;

  int status = noCoverage;
  pathCam::Image *owner = nullptr;

  int updateCount = 0;
  void *preferredObj;

  static void (*destroyPreferredObj)(void *);

  bool usingPreferred = false;
  bool newData = false;
  bool newAnnoData = false;

  Poco::FastMutex mutex;
  cv::Mat image;

  std::map<int, std::pair<Mat, void *> > SAMMasks;

  std::unordered_set<Image*> coveringFrames;

  TileObj(int _tileSize, Point2i _index = {}) : index(_index) {
    image = cv::Mat(_tileSize, _tileSize, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    preferredObj = nullptr;
    usingPreferred = false;
    newData = false;
  }

  ~TileObj() {
    mutex.lock();
    if (usingPreferred && preferredObj) {
      if (destroyPreferredObj) {
        destroyPreferredObj(preferredObj);
      }
      preferredObj = nullptr;
      usingPreferred = false;
    }
    mutex.unlock();
  }
};

struct TileQuery {
public:
  int i, j;
  Rect_<float> bounds;
  std::shared_ptr<TileObj> image;


  TileQuery(std::shared_ptr<TileObj> image, int i, int j, Rect_<float> bounds) : image(image), i(i), j(j),
    bounds(bounds) {
  };
};


class MRTiledImage;

class TiledImage {
public:

  std::weak_ptr<MRTiledImage> parent;
  unsigned int tile_size;
  unsigned int logic_size;
  float logicRatio;

  Dense2DArray<std::shared_ptr<TileObj> > tiles;
  Rect_<float> bounds;

  TiledImage(std::shared_ptr<MRTiledImage> parent = nullptr, unsigned int tile_size = 0,
             unsigned int logic_size = 256, int levelWithinPyramid = 0);

  ~TiledImage() {};

  int levelWithinPyramid;

  unsigned int getTileSize() { return tile_size; }

  unsigned int getLogicSize() { return logic_size; }

  void resetEdges(Point2i topLeft, Point2i bottomRight);

  void makeTile(int x, int y);

  void insertMat(cv::Mat image_in, cv::Rect_<float> i_bounds);

  void insertMatAtBase(cv::Mat image_in, cv::Rect_<float> box, std::vector<Point2i> retileIndices);


  void saveBaseTilesToDisk();

  void matToTile(const cv::Mat &mat, const cv::Mat &mask, int x, int y, Point2f rootOffset,
                 cv::Rect_<float> image_box, cv::Rect_<float> tile_box);

  void insertTilesAtBase(cv::Mat &image_in, cv::Mat &mask, cv::Rect_<float> &box, std::vector<Point2i> &retileIndices);

// #ifdef HAVE_OPENCV_CUDAARITHM
  void matToTile(const cuda::GpuMat &mat, const cuda::GpuMat &mask, int x, int y, Point2f rootOffset,
                 cv::Rect_<float> image_box, cv::Rect_<float> tile_box);

  void insertTilesAtBase(cuda::GpuMat &image_in, cuda::GpuMat &mask, cv::Rect_<float> box,
                         const std::vector<Point2i> &retileIndices);


  void tileUpwards(Point2i myTileIndex, Rect_<float> myLevelRegion, std::shared_ptr<TileObj> myTileObj, Rect cvRoi,
                   int _segID = -1);

  std::shared_ptr<TileObj> getTile(int x, int y);

  std::shared_ptr<TileObj> getTile(Point2i);

// #else
//
//
//   void tileUpwards(Point2i myTileIndex, cv::Rect_<float> myLevelRegion, const cv::Mat &mat);
//
//   Mat getTile(int x, int y);
// #endif

  inline Point2i getIJ(Point2f p) {
    Point2i ij = Point2i(p.x / (int) logic_size, p.y / (int) logic_size);
    if (p.x < 0) { ij.x--; }
    if (p.y < 0) { ij.y--; }
    return ij;
  }

  std::vector<TileQuery> getTiles(cv::Rect_<float> box);

private:
  void matToImage(const cv::Mat &mat, cv::Mat *image, Point2f offset,
                  cv::Rect_<float> image_box, cv::Rect_<float> tile_box);


  void matToImage4Channel(const cv::Mat &mat, int x, int y, Point2f rootOffset,
                          cv::Rect_<float> image_box, cv::Rect_<float> tile_box);

  void matToImage2(const cv::Mat mat, cv::Mat image,
                   Point2f offset, cv::Rect_<float> image_box,
                   cv::Rect_<float> tile_box);
};
