#ifndef IMAGE
#define IMAGE

#include "pathCam.h"

namespace pathCam {

  using Poco::MemoryPool;

  class Image {
  public:
    unsigned int width, height;
    std::atomic<unsigned int> reference_count;
    unsigned long index;
    enum {
      _NOLABEL = 0, _2X, _4X, _10x, _20x, _40X, _UNKNOWN, _BAD_FILE, _LOWFEAT, _UNDEREXP, _OVEREXP, _LENS_CHANGE
    };
    unsigned int label;
    int vertexId;
    Vec2 absoluteCoords;

    Poco::FastMutex buffer_mutex;

    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> keypointsMultilevel;
    cv::Mat descriptors;

    Image(MemoryPool *mempool = 0);

    ~Image();

    void set_memory_pool(MemoryPool *mempool_in) {
      mempool = mempool_in;
    }

    void set_disk_file(Poco::Path _image_file) {
      image_file = _image_file;
    }

    void load_raw_from_disk();

    void increment_smart_pointer() { reference_count++; }

    void copy_in(void *buffer) {
      buffer_mutex.lock();
      allocate_memory_RAW();
      memcpy(raw_buffer, buffer, width * height);
      reference_count++;
      buffer_mutex.unlock();
    }

    bool is_mostly_black();

    bool is_mostly_white(cv::Mat ROI);

    bool is_2x();

    void extract_features();

    void find_label();

    bool is_good();

    double check_blur();

    float debayer(int x, int y);

    std::string get_label() {
      switch (label) {
        case _NOLABEL:
          return "No label";
          break;
        case _2X:
          return "2x";
          break;
        case _4X:
          return "4x";
          break;
        case _10x:
          return "10x";
          break;
        case _40X:
          return "40x";
          break;
        case _UNKNOWN:
          return "Unknown";
          break;
        case _LOWFEAT:
          return "Low Features";
          break;
        case _UNDEREXP:
          return "Under exposed";
          break;
        case _OVEREXP:
          return "Over exposed";
          break;
        case _BAD_FILE:
          return "Bad File";
          break;
        case _LENS_CHANGE:
          return "Lens Change";
      }


    }

    inline char *get_Raw() { return raw_buffer; }

    inline Poco::Path get_ImageFile() { return image_file; }

    void create_reg_image(double reg_scale, double reg_crop, bool convert = true, int interpolation = cv::INTER_LINEAR,
                          bool real = false);

    inline cv::Mat get_reg_image() { return reg_image; }

    inline bool in_memory() { return (raw_buffer != 0); }

    inline double get_reg_scale() { return reg_scale; }

    inline void release_reg_image() { reg_image.release(); }

    inline void free_memory_RAW(bool force = false) {
      buffer_mutex.lock();
      if (raw_buffer != 0) {
        reference_count--;
        if (force || reference_count == 0) {
          if (mempool) {
            mempool->release(raw_buffer);
          } else {
            delete[] raw_buffer;
          }
          raw_buffer = 0;
        }
      }
      buffer_mutex.unlock();
    }


    Poco::Path image_file;
  private:

    MemoryPool *mempool;

    //already protected by mutex in calling function
    inline void allocate_memory_RAW() {
      if (raw_buffer == 0) {
        if (mempool) {
          raw_buffer = reinterpret_cast<char *>(mempool->get());
        } else { raw_buffer = new char[width * height]; }
      }
    }


    char *raw_buffer;
    cv::Mat reg_image;
    double reg_scale;
    double reg_crop;
    double variance;


  };
}

#endif // IMAGE
