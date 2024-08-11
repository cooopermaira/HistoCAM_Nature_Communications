#ifndef IMAGE
#define IMAGE

#include "pathCam.h"

namespace pathCam {

  using Poco::MemoryPool;
  class RegInfo;
  class StreamCam;
  class Image {
  public:
    RegInfo* regInfo;
    double blurVariance;
    unsigned int width, height;
    unsigned int scope_radius;
    unsigned int component_membership;
    std::atomic<unsigned int> reference_count;
    unsigned long index;
    double reg_scale_initial,reg_scale_full;
    double reg_crop_initial,reg_crop_full;
    enum {
      _NOLABEL = 0, _2X, _4X, _10x, _20x, _40X, _UNKNOWN, _BAD_FILE, _LOWFEAT, _UNDEREXP, _OVEREXP, _LENS_CHANGE
    };
    unsigned int label;
    int vertexId;
    Vec2 absoluteCoords;

    Poco::FastMutex buffer_mutex;

    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> keypointsMultilevel;
    std::vector<cv::KeyPoint> keypointsFull;
    cv::Mat descriptors;
    cv::Mat descriptorsMultilevel;
    cv::Mat descriptorsFull;
    cv::Mat readyImage;

    Image(unsigned int width, unsigned int height,unsigned int scope_radius, MemoryPool* mempool = 0);

    ~Image();

    void set_memory_pool(MemoryPool *mempool_in) {
      mempool = mempool_in;
    }

    void set_disk_file(Poco::Path _image_file) {
      image_file = _image_file;
    }

    void build_whitebalance_Mat(StreamCam* parent);

    void load_raw_from_disk();

    cv::Mat full_image_asMat();

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

    bool is_4x();

    void extract_features();

    bool decide_label_and_blur();

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
label = _NOLABEL;
      return "No label";

    }

    inline char *get_Raw() { return raw_buffer; }

    inline Poco::Path get_ImageFile() { return image_file; }

    void create_reg_image(double reg_scale, double reg_crop, bool convert = true, int interpolation = cv::INTER_LINEAR,
                          bool real = false);

    inline cv::Mat get_reg_image() { return reg_image; }

    inline bool in_memory() { return (raw_buffer != 0); }

    inline double get_reg_scale() { return reg_scale_initial; }

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



  };
}

#endif // IMAGE
