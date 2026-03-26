#ifndef IMAGE
#define IMAGE

#include "pathCam.h"


namespace pathCam {

  using Poco::MemoryPool;
  class RegInfo;
  class StreamCam;
  class Match;
  class Image {
  public:
    StreamCam* parent;
    RegInfo* regInfo = nullptr;

    long index;
    long timeStamp = -1;
    double blurTime = 0;
    int loadCount = 0;

    // v DEBUG v
    int matchCount = 0;
    long profileTime = 0;
    // ^ DEBUG ^

    int width, height;
    int scope_radius;
    std::atomic<unsigned int> reference_count;

    float reg_full_scale;
    float motionBlur;
    static inline int blurPatch = 1024;

    enum {
      _NOLABEL = 0, _2X, _4X, _10X, _20X, _40X, /*_UNKNOWN,*/ _BAD_FILE, _LOWFEAT, _UNDEREXP, _OVEREXP, _MOTION_BLUR
    };
    unsigned int label;
    int vertexId;
    cv::Point2i absoluteCoords;

    Poco::FastMutex buffer_mutex,matchesMutex;
    std::mutex cudaBufferMutex, blurMutex;
    std::condition_variable cudaBufferConVar, blurConVar;
    bool cudaBufferReady;
    bool blurSet = false;
    bool hasBeenInMemory = false;
    bool labelObserved = false;

    bool subsequentMatchLaunched = false;


    std::unordered_set<cv::Point2i> ownedTiles;
    std::vector<std::shared_ptr<Match>> matches;
    std::vector<cv::KeyPoint> keypoints,keypointsImageSpace;
    std::vector<Features> akazeFeatures;
    cv::Mat descriptors;

    static cv::cuda::GpuMat hannWindow, blurMask;


    static std::vector<float> valid_scales_for_label(unsigned int label) {
      switch (label) {
        case _2X:
          return {1.0, 0.5, 0.2, 0.1, 0.05};
        case _4X:
          return {2,1,0.4,0.2,0.1};
        case _10X:
          return {5,2.5,1,0.5,0.25};
        case _20X:
          return {10,5,2,1,0.5};
        case _40X:
          return {20,10,4,2,1};
        default:
          throw std::runtime_error("unknown component mag label");
      }
    }

    Image(unsigned int width, unsigned int height,unsigned int scope_radius, MemoryPool* mempool = 0);

    ~Image();

    void set_memory_pool(MemoryPool *mempool_in) {
      mempool = mempool_in;
    }

    void set_disk_file(Poco::Path _image_file) {
      image_file = _image_file;
    }

    unsigned int get_label() const;

    void load_raw_from_disk(bool _alertDoubleLoad = true);

    void manually_set_label();

    cv::Mat full_image_asMat();

    void increment_smart_pointer() { ++reference_count; }

    void copy_in(void *buffer) {
      Poco::FastMutex::ScopedLock lock(buffer_mutex);
      allocate_memory_RAW();
      memcpy(raw_buffer, buffer, width * height);
      ++reference_count;
    }

    static Features buildFeatures(
    const cv::Mat& src,
    float s,
    cv::Ptr<cv::AKAZE> akaze)
    {
      Features f;
      f.scale = s;

      cv::resize(src, f.image, cv::Size(), s, s, cv::INTER_AREA);

      if (!f.image.empty())
        akaze->detectAndCompute(f.image, cv::noArray(), f.kp, f.desc);

      return f;
    }

    static float get_mpp(unsigned int _label) {
      switch (_label) {
        case _2X:
          return 3.45f;
        case _4X:
          return 1.73f;
        case _10X:
          return 0.69f;
        case _20X:
          return 0.35;
        default:
          return 0;
      }
    }

#ifdef HAVE_OPENCV_CUDAARITHM
    void prepare_blur_check_statics() const;
    static void cleanup_blur_check_statics();
    bool move_buffer_to_gpu(int _device, bool _freeHostBuffer = false);
#endif

    bool is_mostly_black();

    bool is_mostly_white(cv::Mat ROI);

    bool is_2x();

    bool is_4x();

    void set_observed_label(const std::string& _label);

    void find_label();

    bool is_good();

    void check_blur_async(const cv::Mat &img = cv::Mat(), bool submitForInference = true);


    float debayer(int x, int y);

    std::string static get_label(unsigned int _label) {
      switch (_label) {
        case _NOLABEL:
          return "Unk";
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
        // case _UNKNOWN:
        //   return "Unknown";
        //   break;
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
    void allocate_memory_RAW();


    char *raw_buffer;
#ifdef HAVE_OPENCV_CUDAARITHM
    char *raw_buffer_cuda;
#endif
    cv::Mat reg_image;
    cv::Mat blurDFT;




  };
}

#endif // IMAGE
