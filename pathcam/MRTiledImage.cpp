//
//  MRTiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/11/24.
//

#include "pathCam.h"
#include <fcntl.h>
#include <sys/stat.h>

//STATIC HELPER FUNCTIONS

std::vector<Point2i> MRTiledImageSet::generate_frame_vertices(const Point2i &Abc, unsigned label) const {
  std::vector<Point2i> result;
  if (!(label == pathCam::Image::_2X || label == pathCam::Image::_4X || label == pathCam::Image::_10X || label ==
        pathCam::Image::_20X || label ==
        pathCam::Image::_40X)) { return {}; }

  auto scale = labelScaleLookup.at((int) label);

  if (label == pathCam::Image::_4X || label == pathCam::Image::_10X || label == pathCam::Image::_20X || label ==
      pathCam::Image::_40X) {
    //rectangle
    result.reserve(4);
    result.push_back(Abc);
    result.push_back(Abc + scale * Point2i(MRTiledImageSet::frameWidth, 0));
    result.push_back(Abc + scale * Point2i(MRTiledImageSet::frameWidth, MRTiledImageSet::frameHeight));
    result.push_back(Abc + scale * Point2i(0, MRTiledImageSet::frameHeight));
  } else if (label == pathCam::Image::_2X) {
    //using octagon
    auto centerPoint = Abc + scale * Point2i(MRTiledImageSet::frameWidth / 2, MRTiledImageSet::frameHeight / 2);
    const int r = scale * MRTiledImageSet::scopeRadius;

    // pick a chamfer amount. r/3 is a decent default; clamp so it never inverts.
    int d = r / 2;
    if (d < 1) d = 1;
    if (d > r - 1) d = r - 1;
    const int cx = centerPoint.x;
    const int cy = centerPoint.y;

    // CW order starting at top edge, moving rightward
    result.reserve(8);

    // Top edge (horizontal): from (cx - (r - d), cy - r) to (cx + (r - d), cy - r)
    result.push_back(Point2i(cx - (r - d), cy - r)); // top-left (after chamfer)
    result.push_back(Point2i(cx + (r - d), cy - r)); // top-right (before chamfer)

    // Right side (vertical): from (cx + r, cy - (r - d)) to (cx + r, cy + (r - d))
    result.push_back(Point2i(cx + r, cy - (r - d))); // upper-right chamfer point
    result.push_back(Point2i(cx + r, cy + (r - d))); // lower-right chamfer point

    // Bottom edge (horizontal)
    result.push_back(Point2i(cx + (r - d), cy + r)); // bottom-right (after chamfer)
    result.push_back(Point2i(cx - (r - d), cy + r)); // bottom-left  (before chamfer)

    // Left side (vertical)
    result.push_back(Point2i(cx - r, cy + (r - d))); // lower-left chamfer point
    result.push_back(Point2i(cx - r, cy - (r - d))); // upper-left chamfer point
  }
  return result; //could be empty, idk. better check return value just sayin
}

static void write_all(int fd, const void *data, size_t size) {
  const char *p = static_cast<const char *>(data);
  size_t written = 0;
  while (written < size) {
    ssize_t n = ::write(fd, p + written, size - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("write failed: ") + std::strerror(errno));
    }
    if (n == 0) {
      throw std::runtime_error("write failed: wrote 0 bytes");
    }
    written += static_cast<size_t>(n);
  }
}

static void read_all(int fd, void *data, size_t size) {
  char *p = static_cast<char *>(data);
  size_t readBytes = 0;
  while (readBytes < size) {
    ssize_t n = ::read(fd, p + readBytes, size - readBytes);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
    }
    if (n == 0) throw std::runtime_error("read failed: unexpected EOF");
    readBytes += static_cast<size_t>(n);
  }
}

static void write_string(int fd, const std::string &str) {
  uint32_t length = static_cast<uint32_t>(str.size());

  write_all(fd, &length, sizeof(length));
  write_all(fd, str.data(), length);
}

std::vector<TileQuery> MRTiledImage::getTiles(cv::Rect_<float> view, cv::Rect_<int> screen, bool pullFromBase) {
  if (pullFromBase) {
    return level[0]->getTiles(view);
  }

  if (level.size() == 0) { return std::vector<TileQuery>(); }
  float scale = max(view.width / float(screen.width),
                    view.height / float(screen.height));

  scale = log2(scale);
  unsigned int i_scale = (unsigned int) (scale + 0.5);
  i_scale = min((unsigned int) (level.size() - 1), i_scale);
  return level[i_scale]->getTiles(view);
}

MRTiledImage::MRTiledImage(pathCam::StreamCam *parent, int _tile_size) : parent(parent), tile_size(_tile_size),
                                                                         scale(0) {
  if (tile_size == 0) {
    tile_size = parent->tileSize;
  }
}

int MRTiledImage::get_class_for_tile(std::tuple<int, int, unsigned> _tile) {
  if (parent) {
    if (parent->tileCoordToClass.find(_tile) != parent->tileCoordToClass.end()) {
      return parent->tileCoordToClass[_tile];
    }
  }
  return -1;
}

void MRTiledImage::cache_to_disk(const std::string &_cwd, bool _keepInMemory) {
  if (!cachedToDisk) {
    Poco::File cwd(_cwd);
    if (!cwd.exists() || !cwd.isDirectory()) {
      throw std::runtime_error("cache_to_disk: cwd invalid: " + _cwd);
    }

    Poco::Path cachePath(_cwd);
    cachePath.makeDirectory();
    cachePath.setFileName(std::to_string(componentIndex));
    cachePath.setExtension("pcRawLayer");

    strCachePath = cachePath.toString();

    int fd = ::open(strCachePath.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd == -1) {
      throw std::runtime_error(std::string("file create failed: ") + strCachePath +
                               " : " + std::strerror(errno));
    }

    // Ensure fd always closes, even if exceptions happen.
    try {
      // Materialize + order tiles (optional but you named it Ordered)
      liveTilesOrderedVec.assign(liveTiles.begin(), liveTiles.end());
      // std::sort(liveTilesOrderedVec.begin(), liveTilesOrderedVec.end()); // if tileIndex is sortable

      auto tileSize = MRTiledImageSet::tileSize;
      const size_t bytesPerTile = tileSize * tileSize * 4; // RGBA8
      const size_t numTiles = liveTilesOrderedVec.size();

      // Check overflow (important if tileSize/numTiles can be large)
      if (tileSize != 0 && bytesPerTile / tileSize / tileSize != 4) {
        throw std::runtime_error("overflow computing bytesPerTile");
      }
      if (numTiles != 0 && bytesPerTile > SIZE_MAX / numTiles) {
        throw std::runtime_error("overflow computing totalBytes");
      }

      const size_t totalBytes = numTiles * bytesPerTile;

      // Preallocate disk space (best-effort / strong guarantee depending on FS)
      // posix_fallocate returns error code directly (does NOT set errno reliably)
      int rc = ::posix_fallocate(fd, 0, static_cast<off_t>(totalBytes));
      if (rc != 0) {
        throw std::runtime_error(std::string("posix_fallocate failed: ") + std::strerror(rc));
      }

      // Now write each tile buffer sequentially
      for (const auto &tileIndex: liveTilesOrderedVec) {
        auto tileObj = level[0]->getTile(tileIndex);
        if (!tileObj) {
          throw std::runtime_error("getTile returned null");
        }

        Poco::FastMutex::ScopedLock lock(tileObj->mutex);

        if (!tileObj->image.data) {
          throw std::runtime_error("tileObj->image has no data");
        }

        write_all(fd, tileObj->image.data, bytesPerTile);

        if (!_keepInMemory) {
          tileObj->image.release();
          if (tileObj->destroyPreferredObj) {
            tileObj->destroyPreferredObj(tileObj->preferredObj);
          }
          tileObj->usingPreferred = false;
          tileObj->preferredObj = nullptr;
        }
      }


      ::close(fd);
    } catch (...) {
      ::close(fd);
      // Optional: remove partially-written file if you consider it invalid
      // ::unlink(outPath.c_str());
      throw;
    }
    cachedToDisk = true;
  } else if (!_keepInMemory){
    assert(!liveTilesOrderedVec.empty());
    for (auto &tileIndex: liveTilesOrderedVec) {
      auto tileObj = level[0]->getTile(tileIndex);
      if (!tileObj) {
        continue;
      }

      Poco::FastMutex::ScopedLock lock(tileObj->mutex);

      if (!tileObj->image.data) {
        continue;
      }

      tileObj->image.release();
      if (tileObj->destroyPreferredObj) {
        tileObj->destroyPreferredObj(tileObj->preferredObj);
      }
      tileObj->usingPreferred = false;
      tileObj->preferredObj = nullptr;
    }
  }

  auto parentSP = level[0]->parent.lock();

  if (!_keepInMemory) {
    for (int i = 1; i < level.size() - 3; ++i) {
      level[i].reset(new TiledImage(parentSP, tile_size, tile_size * (1 << i), i));
    }
  }
  inMemory = _keepInMemory;
}

void MRTiledImage::uncache_from_disk() {
  assert(!strCachePath.empty());
  Poco::File pCacheFile(strCachePath);
  assert(pCacheFile.exists() && pCacheFile.isFile());


  try {
    const auto tileSize = MRTiledImageSet::tileSize;
    const size_t bytesPerTile = tileSize * tileSize * 4; // RGBA8
    const size_t numTiles = liveTilesOrderedVec.size();

    int fd = ::open(strCachePath.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error(std::string("file open failed: ") + strCachePath +
                               " : " + std::strerror(errno));
    }

    {
      struct stat st{};
      if (::fstat(fd, &st) != 0) {
        throw std::runtime_error(std::string("fstat failed: ") + std::strerror(errno));
      }
      const uint64_t expected = static_cast<uint64_t>(numTiles) * static_cast<uint64_t>(bytesPerTile);
      if (static_cast<uint64_t>(st.st_size) < expected) {
        throw std::runtime_error("cache file is smaller than expected (corrupt/incomplete?)");
      }

      for (const auto &tileIndex: liveTilesOrderedVec) {
        auto tileObj = level[0]->getTile(tileIndex);
        if (!tileObj) throw std::runtime_error("getTile returned null");

        Poco::FastMutex::ScopedLock lock(tileObj->mutex);

        if (!tileObj->image.data) {
          tileObj->image = cv::Mat(tileSize, tileSize, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        }

        read_all(fd, tileObj->image.data, bytesPerTile);
        tileObj->newData = true;

        Rect tileROI(0, 0, tileSize, tileSize);
        Rect tileRegion(tileObj->index * tile_size, Size(tile_size, tile_size));
        level[0]->tileUpwards(tileObj->index, tileRegion, tileObj, tileROI);
      }

      ::close(fd);
      inMemory = true;
    }
  } catch (...) {
    // ::close(fd);
    throw;
  }
}

void MRTiledImageSet::write_slide_header() {
  if (headerWritten) { return; }

  Poco::File cwd1(cwd);

  if (!cwd1.exists()) {
    cwd1.createDirectories(); // creates full path recursively
  } else if (!cwd1.isDirectory()) {
    throw std::runtime_error("cache_to_disk: path exists but is not a directory: " + cwd.toString());
  }

  cwd.makeDirectory();
  cwd.append("slide.pcHdr");
  std::string headerPath = cwd.toString();
  int fd = ::open(headerPath.c_str(),
                  O_CREAT | O_TRUNC | O_WRONLY,
                  0644);

  if (fd == -1) {
    throw std::runtime_error(
      std::string("header create failed: ") +
      headerPath + " : " + std::strerror(errno));
  }


  try {
    // MRImageSet metadata
    auto numImages = static_cast<uint16_t>(MRImages.size());
    write_all(fd, &numImages, sizeof(uint16_t));

    //MRTiledImageSet bounds
    write_all(fd, &bounds.x, sizeof(bounds.x)); //float
    write_all(fd, &bounds.y, sizeof(bounds.y)); //float
    write_all(fd, &bounds.width, sizeof(bounds.width)); //float
    write_all(fd, &bounds.height, sizeof(bounds.height)); //float

    write_string(fd, labelName);

    write_all(fd, &framesPerMillisecond,sizeof(framesPerMillisecond)); //float
    // AbCs
    assert(frameLabels.size() == AbCs.size());
    auto numFrames = static_cast<uint16_t>(AbCs.size());
    write_all(fd, &numFrames, sizeof(numFrames));
    for (const auto &abc: AbCs) {
      int32_t x = static_cast<int32_t>(abc.x);
      int32_t y = static_cast<int32_t>(abc.y);

      write_all(fd, &x, sizeof(int32_t));
      write_all(fd, &y, sizeof(int32_t));
    }
    for (const auto &fl: frameLabels) {
      auto fl8 = static_cast<uint8_t>(fl);
      write_all(fd, &fl8, sizeof(fl8));
    }

    //scale lookup
    auto slSize = static_cast<uint8_t>(labelScaleLookup.size());
    write_all(fd, &slSize, sizeof(slSize));
    for (const auto &[label,scale]: labelScaleLookup) {
      auto label8 = static_cast<uint8_t>(label);
      write_all(fd, &label8, sizeof(label8));
      write_all(fd, &scale, sizeof(scale)); //float
    }

    // per MRImage metadata
    for (const auto &mrImg: MRImages) {
      //MRTiledImage scale and mag label
      write_all(fd, &mrImg->scale, sizeof(mrImg->scale)); //float
      write_all(fd, &mrImg->offset.x, sizeof(float));
      write_all(fd, &mrImg->offset.y, sizeof(float));

      uint8_t magLabel = static_cast<uint8_t>(mrImg->magLabel);
      write_all(fd, &magLabel, sizeof(magLabel));

      uint8_t compIdx = static_cast<uint8_t>(mrImg->componentIndex);
      write_all(fd, &compIdx, sizeof(compIdx));

      //MRTiledImage bounds
      write_all(fd, &mrImg->bounds.x, sizeof(mrImg->bounds.x)); //float
      write_all(fd, &mrImg->bounds.y, sizeof(mrImg->bounds.y)); //float
      write_all(fd, &mrImg->bounds.width, sizeof(mrImg->bounds.width)); //float
      write_all(fd, &mrImg->bounds.height, sizeof(mrImg->bounds.height)); //float

      const auto &tiles = mrImg->liveTilesOrderedVec;
      uint16_t numTiles = static_cast<uint16_t>(tiles.size());
      write_all(fd, &numTiles, sizeof(numTiles));

      for (const auto &tileIndex: tiles) {
        int32_t x = static_cast<int32_t>(tileIndex.x);
        int32_t y = static_cast<int32_t>(tileIndex.y);

        write_all(fd, &x, sizeof(int32_t));
        write_all(fd, &y, sizeof(int32_t));
      }
    }

    ::close(fd);
    headerWritten = true;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

void MRTiledImageSet::read_slide_header() {
  Poco::Path dir(cwd);
  dir.makeDirectory(); // ensures trailing slash semantics
  dir.append("slide.pcHdr");

  std::string headerPath = dir.toString();
  std::cout << "reading " + headerPath << std::endl;
  int fd = ::open(headerPath.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error(
      std::string("header open failed: ") +
      headerPath + " : " + std::strerror(errno));
  }

  try {
    // ===============================
    // MRImageSet metadata
    // ===============================

    uint16_t numImages;
    read_all(fd, &numImages, sizeof(numImages));
    MRImages.reserve(numImages);

    // ===============================
    // MRTiledImageSet bounds
    // ===============================

    read_all(fd, &bounds.x, sizeof(float));
    read_all(fd, &bounds.y, sizeof(float));
    read_all(fd, &bounds.width, sizeof(float));
    read_all(fd, &bounds.height, sizeof(float));

    // ===============================
    // labelName (length-prefixed string)
    // ===============================

    uint32_t labelLen;
    read_all(fd, &labelLen, sizeof(labelLen));

    labelName.resize(labelLen);
    read_all(fd, labelName.data(), labelLen);


    // ===============================
    // AbCs
    // ===============================

    read_all(fd, &framesPerMillisecond,sizeof(float));

    uint16_t numFrames;
    read_all(fd, &numFrames, sizeof(numFrames));

    AbCs.reserve(numFrames);
    for (uint16_t i = 0; i < numFrames; ++i) {
      int32_t x, y;
      read_all(fd, &x, sizeof(int32_t));
      read_all(fd, &y, sizeof(int32_t));
      AbCs.emplace_back(x, y);
    }

    frameLabels.reserve(numFrames);
    for (uint16_t i = 0; i < numFrames; ++i) {
      uint8_t fl;
      read_all(fd, &fl, sizeof(fl));
      frameLabels.push_back(fl);
    }

    // ===============================
    // scale lookup
    // ===============================

    uint8_t slSize;
    read_all(fd, &slSize, sizeof(slSize));

    for (uint8_t i = 0; i < slSize; ++i) {
      uint8_t label;
      float scale;

      read_all(fd, &label, sizeof(label));
      read_all(fd, &scale, sizeof(scale));

      labelScaleLookup[label] = scale;
    }

    // ===============================
    // Per MRImage metadata
    // ===============================
    for (uint16_t i = 0; i < numImages; ++i) {
      auto mrImg = std::make_shared<MRTiledImage>(nullptr, tileSize);

      read_all(fd, &mrImg->scale, sizeof(float));
      read_all(fd, &mrImg->offset.x, sizeof(float));
      read_all(fd, &mrImg->offset.y, sizeof(float));

      uint8_t tmp;
      read_all(fd, &tmp, sizeof(uint8_t));
      mrImg->magLabel = tmp;
      read_all(fd, &tmp, sizeof(uint8_t));
      mrImg->componentIndex = tmp;
      Poco::Path cachePath(cwd);
      cachePath.makeDirectory();
      cachePath.setFileName(std::to_string(mrImg->componentIndex));
      cachePath.setExtension("pcRawLayer");

      std::cout << "file " << cachePath.toString() << std::endl;

      read_all(fd, &mrImg->bounds.x, sizeof(float));
      read_all(fd, &mrImg->bounds.y, sizeof(float));
      read_all(fd, &mrImg->bounds.width, sizeof(float));
      read_all(fd, &mrImg->bounds.height, sizeof(float));

      uint16_t numTiles;
      read_all(fd, &numTiles, sizeof(numTiles));

      mrImg->liveTilesOrderedVec.reserve(numTiles);

      for (uint16_t t = 0; t < numTiles; ++t) {
        int32_t x, y;
        read_all(fd, &x, sizeof(int32_t));
        read_all(fd, &y, sizeof(int32_t));
        mrImg->liveTilesOrderedVec.emplace_back(x, y);
      }
      mrImg->liveTiles.insert(mrImg->liveTilesOrderedVec.begin(), mrImg->liveTilesOrderedVec.end());

      auto logicSize = tileSize;
      while (logicSize < mrImg->bounds.width && logicSize < mrImg->bounds.height && log2(tileSize) - mrImg->level.size()
             >= 2) {
        auto level = std::make_shared<TiledImage>(mrImg, tileSize, logicSize, mrImg->level.size());
        mrImg->level.push_back(level);
        logicSize = 2 * logicSize;
      }

      mrImg->strCachePath = cachePath.toString();
      mrImg->uncache_from_disk();
      mrImg->MRImageSet = shared_from_this();
      MRImages.push_back(mrImg);
    }

    ::close(fd);
  } catch (...) {
    std::cout << "read failed" << std::endl;
    ::close(fd);
    throw;
  }
}


std::vector<Point2i> MRTiledImageSet::poly_annotation_from_time_interval(
    long msTimeStart, long msTimeEnd,
    long &startFrameIdx, long &endFrameIdx) const
{
  const auto& ts = frameTimeStamps;

  if (ts.empty()) {
    startFrameIdx = endFrameIdx = -1;
    return {};
  }

  // ---- startFrameIdx: last index with ts[i] <= msTimeStart ----
  auto itStart = std::upper_bound(ts.begin(), ts.end(), msTimeStart);

  if (itStart == ts.begin()) {
    startFrameIdx = 0;  // all timestamps > start → clamp to first
  } else {
    startFrameIdx = static_cast<long>(std::distance(ts.begin(), itStart) - 1);
  }

  // ---- endFrameIdx: first index with ts[i] >= msTimeEnd ----
  auto itEnd = std::lower_bound(ts.begin(), ts.end(), msTimeEnd);

  if (itEnd == ts.end()) {
    endFrameIdx = static_cast<long>(ts.size() - 1);  // all timestamps < end → clamp to last
  } else {
    endFrameIdx = static_cast<long>(std::distance(ts.begin(), itEnd));
  }

  return poly_annotations_from_frame_interval(startFrameIdx, endFrameIdx);
}

std::vector<Point2i> MRTiledImageSet::poly_annotations_from_frame_interval(long startFrameIdx, long endFrameIdx) const {
  if (startFrameIdx <= endFrameIdx) {
    std::vector<std::vector<Point2i> > frameBoundaries;
    frameBoundaries.reserve(endFrameIdx - startFrameIdx + 1);

    for (long i = startFrameIdx; i <= endFrameIdx; ++i) {
      auto res = generate_frame_vertices(AbCs[i], frameLabels[i]);
      if (!res.empty()) {
        frameBoundaries.push_back(res);
      }
    }

    if (!frameBoundaries.empty()) {
      return pathCam::poly_union_envelope::union_boundary_then_chord_simplify_CW(frameBoundaries);
    }
  }
  return {};
}


void MRTiledImage::insertMat(cv::Mat &image_in, cv::Rect_<float> box) {
  bounds = bounds | box;

  //This assumes that the # of levels won't change after adding a new image,
  //which isn't going to be necessarily true.
  for (unsigned int i = 0; i < level.size(); i++) {
    level[i]->insertMat(image_in, box);
    cv::resize(image_in, image_in, cv::Size(image_in.cols / 2, image_in.rows / 2));
  }
}

void MRTiledImageSet::update_bounds() {
  auto minX = bounds.x;
  auto minY = bounds.y;

  auto maxX = minX + bounds.width;
  auto maxY = minY + bounds.height;

  for (const auto &image: MRImages) {
    if (image->suspended) { continue; }

    auto imageMinX = (image->bounds.x + image->offset.x) * image->scale;
    auto imageMinY = (image->bounds.y + image->offset.y) * image->scale;

    minX = fmin(minX, imageMinX);
    minY = fmin(minY, imageMinY);

    auto imageMaxX = imageMinX + image->scale * image->bounds.width;
    auto imageMaxY = imageMinY + image->scale * image->bounds.height;

    maxX = max(maxX, imageMaxX);
    maxY = max(maxY, imageMaxY);
  }

  bounds.x = minX;
  bounds.y = minY;

  bounds.width = maxX - minX;
  bounds.height = maxY - minY;
}

void MRTiledImageSet::detach() {
  Poco::FastMutex::ScopedLock lock(mutex);
  completed = true;

  MRImages.erase(
    std::remove_if(MRImages.begin(), MRImages.end(),
                   [&](const std::shared_ptr<MRTiledImage> &mrImg) {
                     for (const auto &tileIndex: mrImg->liveTiles) {
                       auto tileObj = mrImg->level[0]->getTile(tileIndex);
                       tileObj->owner = nullptr;
                     }

                     return mrImg->suspended;
                   }),
    MRImages.end());
}
