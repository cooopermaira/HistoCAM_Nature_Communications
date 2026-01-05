//
//  util.h
//  pathCam
//
//  Created by Brian on 3/16/23.
//

#ifndef util_h
#define util_h

#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <vector>
#include <tuple>
#include <stdexcept>

namespace std {
  template <>
  struct hash<cv::Point2i> {
    std::size_t operator()(const cv::Point2i& p) const noexcept {
      return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
    }
  };
}

namespace pathCam {

  using namespace nvinfer1;
  struct ScaleResult {
    double scale = 0.0;
    double response = -1.0;
    cv::Point2d shift{0.0, 0.0};
    double shiftNorm = 0.0;
    double shiftNormDiag = 0.0;
    cv::Size cropSize;
    bool valid = false;
  };

  class nvLogger : public ILogger {
    void log(Severity s, const char *msg) noexcept override {
      if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
  };

  extern nvLogger nvloger;

  static std::vector<char> readFile(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
      std::cerr << "Open failed: " << p << "\n";
      std::exit(1);
    }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
  }

  inline void catch_ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh,
                float lowestScale, bool scaleUp) {
    // // Debug logging for SIFT extraction parameters
    // std::cout << "=== SIFT EXTRACTION DEBUG ===" << std::endl;
    // std::cout << "numOctaves: " << numOctaves << std::endl;
    // std::cout << "initBlur: " << initBlur << std::endl;
    // std::cout << "thresh: " << thresh << std::endl;
    // std::cout << "lowestScale: " << lowestScale << std::endl;
    // std::cout << "scaleUp: " << (scaleUp ? "true" : "false") << std::endl;
    //
    // // Debug logging for image properties
    // std::cout << "Image width: " << img.width << std::endl;
    // std::cout << "Image height: " << img.height << std::endl;
    // std::cout << "Image pitch: " << img.pitch << std::endl;
    // std::cout << "Image data pointer: " << (void*)img.d_data << std::endl;
    // std::cout << "Image host data pointer: " << (void*)img.h_data << std::endl;
    //
    // // Parameter validation
    // bool validParams = true;
    // if (numOctaves <= 0 || numOctaves > 8) {
    //   std::cout << "ERROR: Invalid numOctaves (" << numOctaves << "), should be 1-8" << std::endl;
    //   validParams = false;
    // }
    // if (initBlur <= 0.0 || initBlur > 5.0) {
    //   std::cout << "ERROR: Invalid initBlur (" << initBlur << "), should be 0.0-5.0" << std::endl;
    //   validParams = false;
    // }
    // if (thresh < 0.0 || thresh > 10.0) {
    //   std::cout << "ERROR: Invalid thresh (" << thresh << "), should be 0.0-10.0" << std::endl;
    //   validParams = false;
    // }
    // if (lowestScale <= 0.0 || lowestScale > 2.0) {
    //   std::cout << "ERROR: Invalid lowestScale (" << lowestScale << "), should be 0.0-2.0" << std::endl;
    //   validParams = false;
    // }
    // if (img.width <= 0 || img.height <= 0) {
    //   std::cout << "ERROR: Invalid image dimensions (" << img.width << "x" << img.height << ")" << std::endl;
    //   validParams = false;
    // }
    // if (img.d_data == nullptr) {
    //   std::cout << "ERROR: Null GPU data pointer" << std::endl;
    //   validParams = false;
    // }
    //
    // if (!validParams) {
    //   std::cout << "ABORTING: Invalid parameters detected" << std::endl;
    //   return;
    // }
    //
    // std::cout << "All parameters appear valid, proceeding with ExtractSift..." << std::endl;
    //
    // // Additional GPU memory checks
    // size_t imageSize = img.width * img.height * sizeof(float);
    // std::cout << "Image size in bytes: " << imageSize << " (" << (imageSize/1024/1024) << " MB)" << std::endl;
    //
    // // Check if image dimensions are within reasonable CUDA limits
    // if (img.width > 8192 || img.height > 8192) {
    //   std::cout << "WARNING: Very large image dimensions may cause CUDA kernel issues" << std::endl;
    // }
    //
    // // Check GPU memory availability before SIFT extraction
    // size_t free_mem, total_mem;
    // if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
    //   std::cout << "GPU Memory - Free: " << (free_mem/1024/1024) << " MB, Total: " << (total_mem/1024/1024) << " MB" << std::endl;
    //   if (imageSize * 10 > free_mem) { // SIFT needs ~10x image size for processing
    //     std::cout << "ERROR: Insufficient GPU memory for SIFT processing" << std::endl;
    //   }
    // } else {
    //   std::cout << "WARNING: Could not check GPU memory status" << std::endl;
    // }
    
    if (0 != ExtractSift(siftData,img,numOctaves,initBlur,thresh,lowestScale,scaleUp)) {
      int k = 0;
    }
  }

  struct PointComparator {
    bool operator()(const cv::Point2i& lhs, const cv::Point2i& rhs) const {
      if (lhs.x != rhs.x) {
        return lhs.x < rhs.x; // Compare x-coordinates
      }
      return lhs.y < rhs.y;     // Compare y-coordinates if x-coordinates are equal
    }
  };

  class Vec2 {
  public:
    double x, y;

    Vec2() : x(0.0), y(0.0) {};

    Vec2(double x, double y) : x(x), y(y) {};

    std::string toString() {
      std::stringstream ss;
      ss << "(" << x << "," << y << ")";
      return ss.str();
    }
  };

  class Bbox {
  public:
    double min_x, min_y, max_x, max_y;

    Bbox(double min_x = std::numeric_limits<double>::infinity(),
         double min_y = std::numeric_limits<double>::infinity(),
         double max_x = -std::numeric_limits<double>::infinity(),
         double max_y = -std::numeric_limits<double>::infinity()) :
        min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y) {};

    bool intersect(Bbox b) {
      return (min_x <= b.max_x && max_x >= b.min_x) &&
             (min_y <= b.max_y && max_y >= b.min_y);
    }

    cv::Rect as_cvRect() {
      return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
    }

    double area(Bbox b) {
      double Omin_x = fmax(min_x, b.min_x);
      double Omin_y = fmax(min_y, b.min_y);
      double Omax_x = fmin(max_x, b.max_x);
      double Omax_y = fmin(max_y, b.max_y);

      //Assuming both boxes are the same size
      double one_area = (max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y);
      double Oarea = (Omax_x - Omin_x) * (Omax_x - Omin_x) + (Omax_y - Omin_y) * (Omax_y - Omin_y);

      return Oarea / one_area;
    }


    std::string toString() {
      std::stringstream ss;
      ss << "[" << min_x << "," << min_y << "," << max_x << "," << max_y << "]";
      return ss.str();
    }

  };


  template <typename T, typename Hash = std::hash<T>>
  class UniqueQueue {
  private:
    std::queue<T> q;                  // To store elements in FIFO order
    std::unordered_set<T, Hash> seen;        // To track unique elements

  public:

    // Push an element into the queue if it is not already present
    void push(const T& value) {
      if (seen.find(value) == seen.end()) { // Check for uniqueness
        q.push(value);                   // Add to the queue
        seen.insert(value);              // Mark as seen
      }
    }

    // Pop an element from the front of the queue
    void pop() {
      if (!q.empty()) {
        T value = q.front();
        q.pop();                         // Remove from the queue
        seen.erase(value);               // Remove from the set
      }
    }

    // Get the front element of the queue
    T front() const {
      if (!q.empty()) {
        return q.front();
      }
      throw std::runtime_error("Queue is empty!");
    }

    // Check if the queue is empty
    bool empty() const {
      return q.empty();
    }

    // Get the size of the queue
    size_t size() const {
      return q.size();
    }
  };

  struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
      std::size_t h1 = std::hash<T1>()(p.first);  // Hash the first element
      std::size_t h2 = std::hash<T2>()(p.second); // Hash the second element
      return h1 ^ (h2 << 1);                      // Combine the two hashes
    }
  };

  struct TupleHash {
    template <typename T>
    static void hash_combine(std::size_t& seed, const T& value) {
      seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    std::size_t operator()(const std::tuple<int, int, unsigned>& key) const {
      std::size_t seed = 0;
      hash_combine(seed, std::get<0>(key));
      hash_combine(seed, std::get<1>(key));
      hash_combine(seed, std::get<2>(key));
      return seed;
    }
  };



  class ThreadQueue {
  private:
    Poco::ThreadPool *pool;
    std::queue<Poco::Runnable *> jobQueue;

  public:
    ThreadQueue(int min_threads, int max_threads) {
      pool = new Poco::ThreadPool(min_threads, max_threads, 60, POCO_THREAD_STACK_SIZE);
    }

    void run_jobs(std::vector<Poco::Runnable *> jobs) {
      for (unsigned int i = 0; i < jobs.size(); i++) {
        jobQueue.push(jobs[i]);
      }

      while (!jobQueue.empty()) {
        if (pool->available() > 0) {
          pool->start(*jobQueue.front());
          jobQueue.pop();
        } else {
          Poco::Thread::sleep(100);
        }
      }

      pool->joinAll();
    }


  };

}

#endif /* util_h */
