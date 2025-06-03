#ifndef IMAGE
#define IMAGE

#include "pathCam.h"

namespace pathCam {

  using Poco::MemoryPool;
  class RegInfo;
  class StreamCam;
  class Image {
  public:
    StreamCam* parent;
    RegInfo* regInfo;

    unsigned long index;

    unsigned int width, height;
    unsigned int scope_radius;
    unsigned int component_membership;
    std::atomic<unsigned int> reference_count;

    float reg_full_scale;
    double blurVariance;
    double reg_scale_initial,reg_scale_full;
    double reg_crop_initial,reg_crop_full;
    enum {
      _NOLABEL = 0, _2X, _4X, _10X, _20X, _40X, _UNKNOWN, _BAD_FILE, _LOWFEAT, _UNDEREXP, _OVEREXP, _LENS_CHANGE
    };
    unsigned int label;
    int vertexId;
    Vec2 absoluteCoords;

    Poco::FastMutex buffer_mutex;
    std::mutex cudaBufferMutex;
    std::condition_variable cudaBufferConVar;
    bool cudaBufferReady;

    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> keypointsMultilevel;
    std::vector<cv::KeyPoint> keypointsFull;
    cv::Mat descriptors;
    cv::Mat descriptorsMultilevel;
    cv::Mat descriptorsFull;

#ifdef HAVE_OPENCV_CUDAFEATURES2D
    cv::cuda::GpuMat SIFTDescriptors;
    cv::cuda::GpuMat SIFTKeypoints;
    SiftData siftData;
#endif


    Image(unsigned int width, unsigned int height,unsigned int scope_radius, MemoryPool* mempool = 0);

    ~Image();

    void set_memory_pool(MemoryPool *mempool_in) {
      mempool = mempool_in;
    }

    void set_disk_file(Poco::Path _image_file) {
      image_file = _image_file;
    }

    void correct_registration(std::vector<unsigned long> adjacentVerts);

    void load_raw_from_disk();

    void manually_set_label();

    cv::Mat full_image_asMat();

    void increment_smart_pointer() { reference_count++; }

    void copy_in(void *buffer) {
      buffer_mutex.lock();
      allocate_memory_RAW();
      memcpy(raw_buffer, buffer, width * height);
      reference_count++;
      buffer_mutex.unlock();
    }

#ifdef HAVE_OPENCV_CUDAARITHM
    bool move_buffer_to_gpu(int device);
#endif

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

    std::string static get_label(unsigned int _label) {
      switch (_label) {
        case _NOLABEL:
          return "No label";
          break;
        case _2X:
          return "2x";
          break;
        case _4X:
          return "4x";
          break;
        case _10X:
          return "10x";
          break;
        case _20X:
          return "20x";
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

      return "No label";

    }

    void mark_too_dark() {
      label = _UNDEREXP;
      free_memory_RAW();
    }

    inline char *get_Raw() {
      assert(raw_buffer);
      return raw_buffer;
    }

    inline char *get_raw_cuda() {
      assert(raw_buffer_cuda);
      return raw_buffer_cuda;
    }

    inline Poco::Path get_ImageFile() { return image_file; }

    void create_reg_image(double reg_scale, double reg_crop, bool convert = true, int interpolation = cv::INTER_LINEAR,
                          bool real = false);

    inline cv::Mat get_reg_image() { return reg_image; }


    inline bool in_memory() { return (raw_buffer != 0); }

    inline double get_reg_scale() { return reg_scale_initial; }

    inline void release_reg_image() { reg_image.release(); }

    void free_memory_RAW(bool force = false);

    void free_memory_CUDA();

    void write_to_path();
    
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
#ifdef HAVE_OPENCV_CUDAARITHM
    char *raw_buffer_cuda;
#endif
    cv::Mat reg_image;
    cv::Mat reg_image_uncropped;



  };
}

#endif // IMAGE
