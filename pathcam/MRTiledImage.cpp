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

void MRTiledImage::cache_to_disk(const std::string &_cwd) {
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


      const size_t tileSize = parent->tileSize;
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

        if (!tileObj->buf) {
          throw std::runtime_error("tileObj->buf is null");
        }

        write_all(fd, tileObj->buf, bytesPerTile);
        cudaFree(tileObj->buf);
        tileObj->buf = nullptr;
        tileObj->image.release();
        if (tileObj->destroyPreferredObj) {
          tileObj->destroyPreferredObj(tileObj->preferredObj);
        }
        tileObj->usingPreferred = false;
        tileObj->preferredObj = nullptr;
        tileObj->destroyPreferredObj = nullptr;
      }


      ::close(fd);
    } catch (...) {
      ::close(fd);
      // Optional: remove partially-written file if you consider it invalid
      // ::unlink(outPath.c_str());
      throw;
    }
    cachedToDisk = true;
  } else {
    assert(!liveTilesOrderedVec.empty());
    for (auto &tileIndex: liveTilesOrderedVec) {
      auto tileObj = level[0]->getTile(tileIndex);
      if (!tileObj) {
        throw std::runtime_error("getTile returned null");
      }

      Poco::FastMutex::ScopedLock lock(tileObj->mutex);

      if (!tileObj->buf) {
        throw std::runtime_error("tileObj->buf is null");
      }

      cudaFree(tileObj->buf);
      tileObj->buf = nullptr;
      tileObj->image.release();
      if (tileObj->destroyPreferredObj) {
        tileObj->destroyPreferredObj(tileObj->preferredObj);
      }
      tileObj->usingPreferred = false;
      tileObj->preferredObj = nullptr;
      tileObj->destroyPreferredObj = nullptr;
    }
  }

  auto parentSP = level[0]->parent.lock();
  for (int i = 1; i < level.size() - 3; ++i) {
    level[i].reset(new TiledImage(parentSP, tile_size, tile_size * (1 << i), i));
  }
  cudaDeviceSynchronize();
  inMemory = false;
}

void MRTiledImage::uncache_from_disk() {
  assert(!strCachePath.empty());
  Poco::File pCacheFile(strCachePath);
  assert(pCacheFile.exists() && pCacheFile.isFile());


  try {
    const size_t tileSize = parent->tileSize;
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

      cudaError_t cerr = cudaDeviceSynchronize();
      if (cerr != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") +
                                 cudaGetErrorString(cerr));
      }

      int cpuDevice = cudaCpuDeviceId;
      int gpuDevice = 0;
      cudaGetDevice(&gpuDevice);

      for (const auto &tileIndex: liveTilesOrderedVec) {
        auto tileObj = level[0]->getTile(tileIndex);
        if (!tileObj) throw std::runtime_error("getTile returned null");


        Poco::FastMutex::ScopedLock lock(tileObj->mutex);

        // Ensure the managed buffer exists (ideally you allocate these once and keep them)
        if (!tileObj->buf) {
          char *p = nullptr;
          cerr = cudaMallocManaged(&p, bytesPerTile, cudaMemAttachGlobal);
          if (cerr != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMallocManaged failed: ") +
                                     cudaGetErrorString(cerr));
          }
          tileObj->buf = p;
        }

        // Pull pages to CPU to avoid faulting mid-read (optional but often helps)
        // cerr = cudaMemPrefetchAsync(tileObj->buf, bytesPerTile, cpuDevice, 0);
        // if (cerr != cudaSuccess) {
        //   throw std::runtime_error(std::string("cudaMemPrefetchAsync->CPU failed: ") +
        //                            cudaGetErrorString(cerr));
        // }
        cerr = cudaStreamSynchronize(0);
        if (cerr != cudaSuccess) {
          throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                                   cudaGetErrorString(cerr));
        }

        // Read directly into the managed buffer
        read_all(fd, tileObj->buf, bytesPerTile);
        tileObj->image = cuda::GpuMat(tileSize, tileSize, CV_8UC4, tileObj->buf);
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

// void MRTiledImage::cache_to_disk(std::string _cwd) {
//   Poco::File cwd(_cwd);
//   assert(cwd.exists() && cwd.isDirectory());
//
//   Poco::Path cachePath(_cwd);
//   cachePath.makeDirectory();
//   cachePath.setFileName(std::to_string(componentIndex));
//   cachePath.setExtension("pcRawLayer");
//
//   int fd = open(cachePath.toString().c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
//   if (fd == -1) {
//     throw std::runtime_error("file create failed");
//   }
//
//   assert(liveTilesOrderedVec.empty());
//   liveTilesOrderedVec = std::vector(liveTiles.begin(),liveTiles.end());
//
//   try {
//     for (const auto& tileIndex : liveTilesOrderedVec) {
//       auto tileObj = level[0]-> getTile(tileIndex);
//       // assert(!tileObj->owner); //if this function is run outside normal use case, asserting this only limits functionality;
//       tileObj->buf
//     }
//   }catch (...) {
//
//   }
//
//   close(fd);
// }


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

    maxX = max(double(maxX), imageMaxX);
    maxY = max(double(maxY), imageMaxY);
  }

  bounds.x = minX;
  bounds.y = minY;

  bounds.width = maxX - minX;
  bounds.height = maxY - minY;
}

void MRTiledImageSet::detach() {
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
