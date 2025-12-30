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
    RegInfo* regInfo = nullptr;

    long index;
    double blurTime = 0;

    int width, height;
    int scope_radius;
    std::atomic<unsigned int> reference_count;

    float reg_full_scale;
    float motionBlur;
    int blurPatch = 1024;

    enum {
      _NOLABEL = 0, _2X, _4X, _10X, _20X, _40X, _UNKNOWN, _BAD_FILE, _LOWFEAT, _UNDEREXP, _OVEREXP, _MOTION_BLUR
    };
    unsigned int label;
    int vertexId;
    cv::Point2i absoluteCoords;
    cv::Point2f debugInitialGuess;

    Poco::FastMutex buffer_mutex;
    std::mutex cudaBufferMutex, blurMutex;
    std::condition_variable cudaBufferConVar, blurConVar;
    bool cudaBufferReady;
    bool blurSet = false;

    bool subsequentMatchLaunched = false;

    std::vector<cv::KeyPoint> keypoints,keypointsImageSpace;
    cv::Mat descriptors;

    static cv::cuda::GpuMat hannWindow, blurMask;
#ifdef HAVE_OPENCV_CUDAFEATURES2D
    cv::cuda::GpuMat SIFTDescriptors;
    cv::cuda::GpuMat SIFTKeypoints;
    SiftData siftData;
    bool siftInitialized = false;
#endif


    Image(unsigned int width, unsigned int height,unsigned int scope_radius, MemoryPool* mempool = 0);

    ~Image();

    void extract_sift(int numPts, int octaves, float initBlur, float thresh,
                      float lowestScale, float ambiguity, bool async, cv::cuda::GpuMat &buffer);

    void set_memory_pool(MemoryPool *mempool_in) {
      mempool = mempool_in;
    }

    void set_disk_file(Poco::Path _image_file) {
      image_file = _image_file;
    }

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
    void prepare_blur_check_statics();
    bool move_buffer_to_gpu(int _device, bool _freeHostBuffer = false);
#endif

    bool is_mostly_black();

    bool is_mostly_white(cv::Mat ROI);

    bool is_2x();

    bool is_4x();

    bool decide_label_and_blur();

    void find_label();

    bool is_good();

    void check_blur_async(const cv::Mat &img = cv::Mat(), bool submitForInference = true);

    static cv::Point2f compute_sharpness(cv::Mat &_img);

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
        case _MOTION_BLUR:
          return "Motion blur";
      }

      return "No label";

    }

    void mark_too_dark() {
      label = _UNDEREXP;
    }

    inline char *get_Raw() {
      //assert(raw_buffer);
      return raw_buffer;
    }

    char *get_raw_cuda();

    inline Poco::Path get_ImageFile() { return image_file; }

    void create_reg_image(double reg_scale, double reg_crop, bool convert = true, int interpolation = cv::INTER_LINEAR,
                          bool real = false);

    inline cv::Mat get_reg_image() { return reg_image; }


    inline bool in_memory() { return (raw_buffer != 0); }

    double get_reg_scale() const;

    inline void release_reg_image() { reg_image.release(); }

    void free_memory_RAW(bool force = false);

    void free_memory_cuda();

    void write_to_path(bool _profile = false);
    
    Poco::Path image_file;

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
    cv::Mat blurDFT;




  };
}

#endif // IMAGE
