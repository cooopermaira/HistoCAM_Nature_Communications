//
//  MRTiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/11/24.
//

#include "pathCam.h"
#include <fcntl.h>
#include <sys/stat.h>

// using namespace pathCam;

//STATIC HELPER FUNCTIONS
int preallocate_file(int fd, off_t length) {
#ifdef __APPLE__
  fstore_t store = {};
  store.fst_flags = F_ALLOCATECONTIG; // try contiguous first
  store.fst_posmode = F_PEOFPOSMODE;
  store.fst_offset = 0;
  store.fst_length = length;

  int rc = fcntl(fd, F_PREALLOCATE, &store);
  if (rc == -1) {
    // fallback to non-contiguous
    store.fst_flags = F_ALLOCATEALL;
    rc = fcntl(fd, F_PREALLOCATE, &store);
  }

  if (rc != -1) {
    rc = ftruncate(fd, length);
  }

  return rc;
#else
  return posix_fallocate(fd, 0, length);
#endif
}

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
  return result; //could be empty, idk. better check return value just sayin //who tf made this comment?
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

void MRTiledImage::save_to_disk(const std::string &dir) {
  auto ppath = Poco::Path(dir);
  ppath.makeDirectory();
  ppath.setFileName(std::to_string(componentIndex));
  ppath.setExtension("png");

  Point2i minIdx(INT_MAX, INT_MAX);
  Point2i maxIdx(INT_MIN, INT_MIN);

  for (auto &tileIdx: liveTiles) {
    minIdx.x = std::min(minIdx.x, tileIdx.x);
    minIdx.y = std::min(minIdx.y, tileIdx.y);
    maxIdx.x = std::max(maxIdx.x, tileIdx.x);
    maxIdx.y = std::max(maxIdx.y, tileIdx.y);
  }

  Size size(tile_size * (maxIdx.x - minIdx.x + 1), tile_size * (maxIdx.y - minIdx.y + 1));
  Mat data(size,CV_8UC4, Scalar(0,0,0,0));

  for (auto &tileIdx: liveTiles) {
    auto tileObj = get_base_tile(tileIdx);
    assert(tileObj);
    assert(!tileObj->image.empty());

    Rect roi(tile_size * (tileIdx.x - minIdx.x), tile_size * (tileIdx.y - minIdx.y), tile_size, tile_size);
    tileObj->image.copyTo(data(roi));
  }

  imwrite(ppath.toString(),data);
}

void MRTiledImage::extract_akaze() {
  Point2i minIdx(INT_MAX, INT_MAX);
  Point2i maxIdx(INT_MIN, INT_MIN);

  for (auto &tileIdx: liveTiles) {
    minIdx.x = std::min(minIdx.x, tileIdx.x);
    minIdx.y = std::min(minIdx.y, tileIdx.y);
    maxIdx.x = std::max(maxIdx.x, tileIdx.x);
    maxIdx.y = std::max(maxIdx.y, tileIdx.y);
  }

  akazeMinIdx = minIdx;

  Size size(tile_size * (maxIdx.x - minIdx.x + 1), tile_size * (maxIdx.y - minIdx.y + 1));
  Mat data(size,CV_8UC1, Scalar(0));
  Mat mask(size,CV_8UC1, Scalar(0));

  for (auto &tileIdx: liveTiles) {
    auto tileObj = get_base_tile(tileIdx);
    assert(tileObj);
    assert(!tileObj->image.empty());

    Rect roi(tile_size * (tileIdx.x - minIdx.x), tile_size * (tileIdx.y - minIdx.y), tile_size, tile_size);
    auto dataROI = data(roi);
    auto maskROI = mask(roi);

    maskROI.setTo(255);
    cvtColor(tileObj->image, dataROI, COLOR_BGRA2GRAY);
  }

  for (auto s: /*{1.f, 0.5f, 0.1f}*/{0.5, 0.25, 0.1}) {
    akaze.push_back(buildFeatures(data, s, mask, true));
  }
}

MRTiledImage::MRTiledImage(StreamCam *parent, int _tile_size) : parent(parent), tile_size(_tile_size),
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
      int rc = ::preallocate_file(fd, static_cast<off_t>(totalBytes));
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
  } else if (!_keepInMemory) {
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

    write_all(fd, &framesPerMillisecond, sizeof(framesPerMillisecond)); //float
    // AbCs
    assert(frameComponentMembership.size() == AbCs.size());
    auto numFrames = static_cast<uint16_t>(AbCs.size());
    write_all(fd, &numFrames, sizeof(numFrames));
    for (const auto &abc: AbCs) {
      int32_t x = static_cast<int32_t>(abc.x);
      int32_t y = static_cast<int32_t>(abc.y);

      write_all(fd, &x, sizeof(int32_t));
      write_all(fd, &y, sizeof(int32_t));
    }
    for (const auto &fl: frameComponentMembership) {
      auto fl8 = static_cast<int8_t>(fl);
      write_all(fd, &fl8, sizeof(fl8));
    }

    for (const auto &ts : frameTimeStamps) {
      int64_t t = static_cast<int64_t>(ts);
      write_all(fd, &t, sizeof(t));
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

void MRTiledImageSet::load_slide() {
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

    read_all(fd, &framesPerMillisecond, sizeof(float));

    uint16_t numFrames;
    read_all(fd, &numFrames, sizeof(numFrames));

    AbCs.reserve(numFrames);
    for (uint16_t i = 0; i < numFrames; ++i) {
      int32_t x, y;
      read_all(fd, &x, sizeof(int32_t));
      read_all(fd, &y, sizeof(int32_t));
      AbCs.emplace_back(x, y);
    }

    frameComponentMembership.reserve(numFrames);
    for (uint16_t i = 0; i < numFrames; ++i) {
      int8_t fl;
      read_all(fd, &fl, sizeof(fl));
      frameComponentMembership.push_back(fl);
    }

    frameTimeStamps.reserve(numFrames);
    for (uint16_t i = 0; i < numFrames; ++i) {
      uint64_t ts;
      read_all(fd, &ts,sizeof(ts));
      frameTimeStamps.push_back(ts);
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

std::vector<Point2i> MRTiledImageSet::frame_centers_from_time_interval(long msTimeStart, long msTimeEnd, long &startFrameIdx, long &endFrameIdx) const {
  const auto &ts = frameTimeStamps;

  if (ts.empty()) {
    startFrameIdx = endFrameIdx = -1;
    return {};
  }

  // ---- startFrameIdx: last index with ts[i] <= msTimeStart ----
  auto itStart = std::upper_bound(ts.begin(), ts.end(), msTimeStart);

  if (itStart == ts.begin()) {
    startFrameIdx = 0; // all timestamps > start → clamp to first
  } else {
    startFrameIdx = std::distance(ts.begin(), itStart) - 1;
  }

  // ---- endFrameIdx: first index with ts[i] >= msTimeEnd ----
  auto itEnd = std::lower_bound(ts.begin(), ts.end(), msTimeEnd);

  if (itEnd == ts.end()) {
    endFrameIdx = static_cast<long>(ts.size() - 1); // all timestamps < end → clamp to last
  } else {
    endFrameIdx = std::distance(ts.begin(), itEnd);
  }

  return frame_centers_from_frame_interval(startFrameIdx, endFrameIdx);
}


std::vector<Point2i> MRTiledImageSet::poly_annotation_from_time_interval(
  long msTimeStart, long msTimeEnd,
  long &startFrameIdx, long &endFrameIdx) const {
  const auto &ts = frameTimeStamps;

  if (ts.empty()) {
    startFrameIdx = endFrameIdx = -1;
    return {};
  }

  // ---- startFrameIdx: last index with ts[i] <= msTimeStart ----
  auto itStart = std::upper_bound(ts.begin(), ts.end(), msTimeStart);

  if (itStart == ts.begin()) {
    startFrameIdx = 0; // all timestamps > start → clamp to first
  } else {
    startFrameIdx = static_cast<long>(std::distance(ts.begin(), itStart) - 1);
  }

  // ---- endFrameIdx: first index with ts[i] >= msTimeEnd ----
  auto itEnd = std::lower_bound(ts.begin(), ts.end(), msTimeEnd);

  if (itEnd == ts.end()) {
    endFrameIdx = static_cast<long>(ts.size() - 1); // all timestamps < end → clamp to last
  } else {
    endFrameIdx = static_cast<long>(std::distance(ts.begin(), itEnd));
  }

  return poly_annotations_from_frame_interval(startFrameIdx, endFrameIdx);
}


std::vector<Point2i> MRTiledImageSet::frame_centers_from_frame_interval(long startFrameIdx, long endFrameIdx) const {

  if (startFrameIdx <= endFrameIdx) {

    std::vector<Point2i> frameCenters;
    frameCenters.reserve(endFrameIdx - startFrameIdx + 1);

    Point2i center(MRTiledImageSet::frameWidth / 2, MRTiledImageSet::frameHeight / 2);

    for (long i = startFrameIdx; i <= endFrameIdx; ++i) {
      if (frameComponentMembership[i] < 0){continue;}
      auto mrImg = get_mrImg_by_comp_idx(frameComponentMembership[i]);
      auto coords = mrImg->scale * (Point2i(mrImg->offset) + AbCs[i] + center);
      frameCenters.push_back(coords);
    }

    return frameCenters;
  }
  return {};
}


std::vector<Point2i> MRTiledImageSet::poly_annotations_from_frame_interval(long startFrameIdx, long endFrameIdx) const {

  if (startFrameIdx <= endFrameIdx) {

    std::vector<std::vector<Point2i> > frameBoundaries;
    frameBoundaries.reserve(endFrameIdx - startFrameIdx + 1);

    for (long i = startFrameIdx; i <= endFrameIdx; ++i) {
      if (frameComponentMembership[i] < 0){continue;}
      auto mrImg = get_mrImg_by_comp_idx(frameComponentMembership[i]);
      auto label = mrImg->magLabel;
      auto coords = mrImg->scale * (mrImg->offset + Point2f(AbCs[i]));
      auto res = generate_frame_vertices(coords, label);
      if (!res.empty()) {
        frameBoundaries.push_back(res);
      }
    }

    if (!frameBoundaries.empty()) {
      return poly_union_envelope::union_boundary_then_chord_simplify_CW(frameBoundaries);
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

// void MRTiledImageSet::correct_alignment() {
//   for (auto &mrImg : MRImages) {
//     if (mrImg->suspended){continue;}
//     mrImg->extract_akaze();
//   }
// }

void MRTiledImageSet::correct_alignment()
{
    if (MRImages.empty()) return;

    // Step 1: extract features + reset
    for (auto& mrImg : MRImages)
    {
        if (mrImg->suspended) continue;
        mrImg->extract_akaze();
        mrImg->aligned = false;
    }

    const int anchorIdx = 0;
    MRImages[anchorIdx]->aligned = true;

    // Order by closeness to anchor scale (only affects iteration order)
    std::vector<int> order;
    for (int i = 0; i < (int)MRImages.size(); ++i)
    {
        if (i == anchorIdx) continue;
        if (MRImages[i]->suspended) continue;
        order.push_back(i);
    }

    std::sort(order.begin(), order.end(),
              [&](int a, int b)
              {
                  return std::abs(MRImages[a]->scale - MRImages[anchorIdx]->scale) <
                         std::abs(MRImages[b]->scale - MRImages[anchorIdx]->scale);
              });

    // 🔥 Iterate until no more progress (important!)
    bool progress = true;

    while (progress)
    {
        progress = false;

        for (int idx : order)
        {
            auto& root = MRImages[idx];
            if (root->aligned) continue;

            // --- build candidate list (aligned + different magLabel) ---
            std::vector<std::pair<double,int>> candidates;

            for (int j = 0; j < (int)MRImages.size(); ++j)
            {
                if (!MRImages[j]->aligned) continue;
                if (MRImages[j]->suspended) continue;
                if (MRImages[j]->magLabel == root->magLabel) continue;

                double d = std::abs(MRImages[j]->scale - root->scale);
                candidates.emplace_back(d, j);
            }

            if (candidates.empty()) continue;

            std::sort(candidates.begin(), candidates.end());

            // --- try candidates in order of likelihood ---
            const int MAX_TRIES = 4;
            bool alignedHere = false;

            for (int k = 0; k < std::min((int)candidates.size(), MAX_TRIES); ++k)
            {
                int j = candidates[k].second;
                auto& target = MRImages[j];

                HomographyResultM result =
                    findHomographyAKAZE_allScalePairs(target->akaze, root->akaze);

                if (!result.valid) continue;
                if (result.inliers < 20) continue;
                if (result.H.empty() || result.H.rows != 3 || result.H.cols != 3) continue;

                const double h00 = result.H.at<double>(0, 0);
                const double h11 = result.H.at<double>(1, 1);
                const double h01 = result.H.at<double>(0, 1);
                const double h10 = result.H.at<double>(1, 0);
                const double tx  = result.H.at<double>(0, 2);
                const double ty  = result.H.at<double>(1, 2);

                if (std::abs(h01) > 0.1 || std::abs(h10) > 0.1) continue;

                const double sx = h00;
                const double sy = h11;
                const double s  = 0.5 * (sx + sy);

                if (!std::isfinite(s) || std::abs(s) < 1e-8) continue;

                // --- origin conversion ---
                const cv::Point2f O_target(
                    target->akazeMinIdx.x * target->tile_size,
                    target->akazeMinIdx.y * target->tile_size
                );

                const cv::Point2f O_root(
                    root->akazeMinIdx.x * root->tile_size,
                    root->akazeMinIdx.y * root->tile_size
                );

                // --- your correct transform ---
                const double newScale = target->scale / s;

                cv::Point2f newOffset;
                newOffset.x = static_cast<float>(sx * (O_target.x + target->offset.x) - tx - O_root.x);
                newOffset.y = static_cast<float>(sy * (O_target.y + target->offset.y) - ty - O_root.y);

                if (!std::isfinite(newScale) ||
                    !std::isfinite(newOffset.x) ||
                    !std::isfinite(newOffset.y))
                {
                    continue;
                }

                std::cout << "alignment refinement: layer " << root->componentIndex
                          << "  scale " << root->scale << " -> " << newScale
                          << "  offset (" << root->offset.x << "," << root->offset.y
                          << ") -> (" << newOffset.x << "," << newOffset.y << ")"
                          << " anchored to layer " << target->componentIndex
                          << std::endl;

                root->set_scale(newScale);
                root->set_offset(newOffset);
                root->aligned = true;

                alignedHere = true;
                progress = true;
                break;
            }

            if (!alignedHere)
            {
                // optional debug
                // std::cout << "layer " << root->componentIndex << " failed this pass\n";
            }
        }
    }

    // Final report
    for (int i = 0; i < (int)MRImages.size(); i++)
    {
        if (!MRImages[i]->aligned)
        {
            std::cout << "Layer " << MRImages[i]->componentIndex
                      << " failed to align." << std::endl;
        }
    }
}

void MRTiledImageSet::generate_nav_paths() {
  if (MRImages.empty() || frameComponentMembership.empty() || frameTimeStamps.empty() || AbCs.empty()) {
    return;
  }
  navPaths.clear();

  NavigationPath currentNavPath;
  int ii = -1;
  std::shared_ptr<MRTiledImage> mrImg = nullptr;
  while (!mrImg) {
    ++ii;
    mrImg = get_mrImg_by_comp_idx(frameComponentMembership[ii]);
  }
  currentNavPath.magLabel = mrImg->magLabel; //assumes the first frame was valid. there will undoubtedly be a case where that isnt true eventually;
  currentNavPath.startTime = frameTimeStamps[ii];
  currentNavPath.startFrame = ii;
  currentNavPath.componentIndex = frameComponentMembership[ii];

  for (int i = 1 + ii; i < frameComponentMembership.size(); ++i) {
    if (frameComponentMembership[i] < 0){continue;}

    auto mrImg = get_mrImg_by_comp_idx(frameComponentMembership[i]);
    if (mrImg->magLabel != currentNavPath.magLabel) {
      currentNavPath.endTime = frameTimeStamps[i - 1];
      currentNavPath.endFrame = i - 1;
      currentNavPath.frameCenters = frame_centers_from_frame_interval(currentNavPath.startFrame,currentNavPath.endFrame);
      currentNavPath.distancePerFrame = currentNavPath.calc_dist_per_frame();
      navPaths.push_back(currentNavPath);

      currentNavPath = NavigationPath{};
      currentNavPath.magLabel = mrImg->magLabel;
      currentNavPath.startFrame = i;
      currentNavPath.startTime = frameTimeStamps[i];
      currentNavPath.componentIndex = frameComponentMembership[i];
    }
  }

  currentNavPath.endTime = frameTimeStamps.back();
  currentNavPath.endFrame = frameComponentMembership.size() - 1;
  currentNavPath.frameCenters = frame_centers_from_frame_interval(currentNavPath.startFrame,currentNavPath.endFrame);
  currentNavPath.distancePerFrame = currentNavPath.calc_dist_per_frame();
  navPaths.push_back(currentNavPath);
}
