//
//  MRTiledImage.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/11/24.
//

#ifndef MRTiledImage_h
#define MRTiledImage_h

#include "pathCam.h"
#include "StreamCam.h"
#include "TiledImage.h"

struct Point2iLess {
  bool operator()(const cv::Point2i &a,
                  const cv::Point2i &b) const {
    return (a.y < b.y) || (a.y == b.y && a.x < b.x);
  }
};

class MRTiledImageSet;

class MRTiledImage {
  friend class LoadingThread;

public:
  bool suspended = false;
  cv::Rect_<float> bounds;
  int tile_size, magLabel, componentIndex = 0;
  float scale;
  Point2f offset;
  Poco::Event scaleSet;
  pathCam::StreamCam *parent;
  std::weak_ptr<MRTiledImageSet> MRImageSet;
  std::vector<std::shared_ptr<TiledImage> > level;
  std::set<Point2i, Point2iLess> liveTiles;
  std::vector<Point2i> liveTilesOrderedVec;

  std::string strCachePath;

  std::atomic<bool> cachedToDisk = false;
  std::atomic<bool> inMemory = true;


  MRTiledImage(pathCam::StreamCam *parent = nullptr, int _tile_size = 0);

  ~MRTiledImage() = default;

  void insertMat(cv::Mat &image_in, cv::Rect_<float> box);

  std::shared_ptr<TileObj> get_base_tile(int x, int y) {
    return level[0]->getTile(x, y);
  }

  std::shared_ptr<TileObj> get_base_tile(Point2i _index) {
    return level[0]->getTile(_index);
  }

  // #ifdef HAVE_OPENCV_CUDAARITHM
  void insertTilesAtBase(cuda::GpuMat &image_in, cuda::GpuMat &mask, cv::Rect_<float> box,
                         std::vector<Point2i> &retileIndices) {
    level[0]->insertTilesAtBase(image_in, mask, box, retileIndices);
    bounds = level[0]->bounds;
  };
  // #else
  void insertTilesAtBase(cv::Mat &image_in, cv::Mat &mask, cv::Rect_<float> &box, std::vector<Point2i> &retileIndices) {
    level[0]->insertTilesAtBase(image_in, mask, box, retileIndices);
    bounds = level[0]->bounds;
  }

  // #endif

  int get_class_for_tile(std::tuple<int, int, unsigned> _tile);

  void cache_to_disk(const std::string &_cwd, bool _keepInMemory = false);

  void uncache_from_disk();

  void set_scale(double _scale) { scale = _scale; }

  void set_mag_label(unsigned int _magLabel) { magLabel = _magLabel; }

  void set_offset(Point2f _offset) { offset = _offset; }

  std::vector<TileQuery> getTiles(cv::Rect_<float> bounds, cv::Rect_<int> screen, bool pullFromBase = false);

private:
  inline cv::Rect_<float> worldToLevel(cv::Rect_<float> r, unsigned int level) {
    float denom = (2.0f * level);
    return cv::Rect_<float>(r.x / denom, r.y / denom, r.width / denom, r.height / denom);
  }

  inline cv::Rect_<float> levelToWorld(cv::Rect_<float> r, unsigned int level) {
    float denom = (2.0f * level);
    return cv::Rect_<float>(r.x * denom, r.y * denom, r.width * denom, r.height * denom);
  }

  inline Point2f worldToLevel(Point2f p, unsigned int level) {
    return p / (2.0f * level);
  }

  inline Point2f levelToWorld(Point2f p, unsigned int level) {
    return p * (2.0f * level);
  }
};


class MRTiledImageSet : public std::enable_shared_from_this<MRTiledImageSet>{
  friend class ImageViewComponent;
  friend class CaptureComponent;

public:
  std::vector<std::shared_ptr<MRTiledImage> > MRImages;

  cv::Rect_<float> bounds;
  std::string labelName;
  Poco::Path cwd;

  std::atomic<bool> headerWritten = false;
  std::atomic<bool> inMemory = true;
  std::atomic<bool> loadFromCacheQueued = false;
  std::atomic<bool> cachedToDisk = false;
  std::atomic<bool> completed = false;
  int index;

  std::vector<Point2i> AbCs;
  std::vector<unsigned> frameLabels;
  std::unordered_map<int,float> labelScaleLookup;
  float framesPerMillisecond;
  long captureTimeMS;

  inline static int frameHeight = 0;
  inline static int frameWidth = 0;
  inline static int scopeRadius = 0;
  inline static int tileSize = 0;


  Poco::FastMutex mutex;


  MRTiledImageSet() {};
  ~MRTiledImageSet() {
    // Poco::File workDir(cwd);
    // if (workDir.exists() && workDir.isDirectory()) {
    //   workDir.remove(true);
    // }
  }


  Point2f get_display_coords_for_zero_scale(std::shared_ptr<MRTiledImage> _member) const {
    Point2f startPoint(bounds.br().x, 0);
    for (auto mrimg: MRImages) {
      if (mrimg == _member) {
        startPoint.x -= _member->bounds.tl().x;
        startPoint.x += 500;
        return startPoint;
      }
      if (!mrimg->suspended && mrimg->scale == 0) {
        startPoint.x += (mrimg->bounds.br().x - mrimg->bounds.tl().x);
      }
    }
    throw std::runtime_error("did not find calling member in list of MRTiledImages");
  }

  void add(std::shared_ptr<MRTiledImage> image) {
    MRImages.push_back(image);
  }

  void sort_by_scale() {
    std::stable_sort(MRImages.begin(), MRImages.end(),
                     [](const auto &a, const auto &b) { return a->scale > b->scale; });
    int k = 0;
  }

  bool empty() { return MRImages.empty(); }

  cv::Rect_<float> get_component_bounds(int _component_index) { return MRImages[_component_index]->bounds; }

  void update_bounds();

  void detach();

  void write_slide_header();

  void read_slide_header();

  void cache_to_disk(bool _keepInMemory = false) {
    auto start = std::chrono::high_resolution_clock::now();
    for (auto &mrImg: MRImages) {
      if(!mrImg->inMemory){continue;}
      mrImg->cache_to_disk(cwd.toString(), _keepInMemory);
    }
    write_slide_header();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    cachedToDisk = true;
    inMemory = _keepInMemory;

    std::cout << "cached " << index << " in " << duration.count() << " ms" << std::endl;
  };

  void uncache_from_disk() {
    auto start = std::chrono::high_resolution_clock::now();
    for (auto &mrImg: MRImages) {
      assert(!mrImg->inMemory);
      mrImg->uncache_from_disk();
      assert(mrImg->inMemory);
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    inMemory = true;
    std::cout << "uncached " << index << " in " << duration.count() << " ms" << std::endl;
  }

  std::vector<Point2i> poly_annotation_from_time_interval(long msTimeStart, long msTimeEnd, long &startFrameIdx, long &endFrameIdx) const;

  std::vector<Point2i> poly_annotations_from_frame_interval(long startFrameIdx, long endFrameIdx) const;

  std::vector<Point2i> generate_frame_vertices(const Point2i &Abc, unsigned label) const;
};


#endif /* MRTiledImage_h */
