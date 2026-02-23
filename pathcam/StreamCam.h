//
//  StreamCam.h
//  pathCamLib
//
//  Created by cooper maira on 11/24/23.
//

#ifndef StreamCam_h
#define StreamCam_h

#include <stdio.h>

#include "MetricComposite.h"
#include "AccessSAM.h"
#include "pathCam.h"
#include "MRTiledImage.h"

using Poco::MemoryPool;
using Poco::Path;
using Poco::Logger;

class MRTiledImage;


namespace pathCam {
  class Composite;

  class CompositeVoronoi;

  class CompositeManager;

  class QManager;

  class JobQueue;

  class DiskReader;

  class InferenceManager;

  class PostProcessManager;

  class SiftFeatureMatcher;

  class FeatureTrackGenerator;

  class BundleAdjustmentIntegrator;

  class AccessSAM;


  class StreamCam : public BatchCam {

    friend class FeaturesRunnable;

    friend class MatchRunnable;

    friend class DiskReader;

    friend class DiskStreamer;

    friend class Loader;

    friend class SpinLoader;

    friend class QManager;

    friend class RegManager;

    friend class Composite;

    friend class CompositeVoronoi;

    friend class CompositeManager;

    friend class LoaderLogicRunnable;

    friend class RegistrationRunnable;

    friend class SingleMatchRunnable;

    friend class XCompRunnable;

    friend class ImageToTileCopyRunnable;

    friend class RebuildRunnable;

    friend class RegInfo;

  private:
    struct compareCompositeQSort{
      bool operator()(const std::vector<RegInfo*> lhs, const std::vector<RegInfo*> rhs){
        return lhs.at(0)->index > rhs.at(0)->index;
      };
    };

    std::priority_queue<std::vector<RegInfo*>, std::deque<std::vector<RegInfo*>>, compareCompositeQSort > compositeBatch;


  public:

    // v DEBUG v
    std::vector<int> matchablesIncremented,matchablesDecremented;
    StreamCam(Poco::Util::LayeredConfiguration::Ptr config);

    ~StreamCam();

    Poco::RWLock image_mutex;
    Poco::RWLock reg_results_mutex;
    Poco::RWLock resize_mmatch_mutex;
    Poco::FastMutex resize_buffer_mutex;
    Poco::FastMutex buffer_mutex;
    Poco::FastMutex compositeQ_mutex;
    Poco::FastMutex component_mutex;
    Poco::FastMutex lastFrameMutex;
    Poco::FastMutex scaleRepoMutex;
    Poco::FastMutex inferenceQMutex;
    Poco::FastMutex siftQMutex;
    Poco::FastMutex pixelDistanceMutex;
    Poco::FastMutex pyramidQMutex;
    Poco::FastMutex blurMutex;
    Poco::FastMutex CudaSiftGlobalUseMutex;
    Poco::FastMutex previousSlidesMutex;

    Poco::Path givenWorkingDirectory;

    long cudaSiftTime = 0;
    long captureTimeMS = 0;

    std::string inputFileOverride;
    std::string currentSlideLabel;
    int currentSlideIndex;
    int numSlides = 0;

    cv::Rect_<float> lastFrame;
    int lastComponentIndex;
    std::string lastLabel;
    bool showAsCircle;
    float lastScale;

    bool microscopeInput;
    bool recordingMode = false;

    bool opencvWithCuda = false;
    int compositorCudaDevice;
    int siftCudaDevice = -1;

    cuda::Stream cvCompositeStream;

    int siftWindow = 1024;
    int siftPoints = 60000;

    int lastActiveComponent = 0;
    int maxTilesPerBatch = 512;
    int maxMatchesPerPull = 50;
    int minPixelDistanceBetweenFrames;
    std::vector<Vec2> lastAcceptedCoords;
    Image* lastViewedFrame = nullptr;


    std::vector<double> labelScales = { -10.0, 1.0, 0.5, 0.2, 0.1, 0.05 };

    std::map<int, std::pair<double, Point2f>> scaleRepo;
    std::map<std::tuple<int,int,int>,int> tileCoordToClass;

    cuda::GpuMat flat_field2X;
    cuda::GpuMat flat_field4X;
    cuda::GpuMat flat_field10X;
    cuda::GpuMat flat_field20X;
    Mat circleMask;
    Mat regCircleMask;

    std::shared_ptr<MRTiledImageSet> MRImageSet;
    std::vector<std::shared_ptr<MRTiledImageSet>> previousSlides;

    std::vector<std::pair<std::string, double>> debugImageBlurWithNames;
    std::vector<double> debugImageBlur;
    CompositeManager *cm;
    QManager *qm;
    DiskReader *dr;
    InferenceManager *im;
    PostProcessManager* ppm;
    SiftFeatureMatcher* sfm;
    std::shared_ptr<AccessSAM> as;
    std::shared_ptr<JobQueue>JobQ;
    std::shared_ptr<JobQueue>jqSecondary;
    FeatureTrackGenerator *ftg;

    //std::vector < double > variancesForDebug;
    //std::vector<CompositeVoronoi *> composites;
    std::vector<std::shared_ptr<Composite>> composites;
    std::vector<bool> visited;

    std::queue<std::tuple<unsigned long, cv::Size, unsigned int> > newComponentQ;
    //UniqueQueue<std::pair<Point2i,unsigned int>,PairHash> tileEmbedQ;
    UniqueQueue<std::tuple<int,int,unsigned>,TupleHash>tileEmbedQ;
    std::deque<std::vector<std::pair<Image*,Image*>>> siftMatchQueue;
    std::queue<Image*> siftDataQueue;
    std::queue<std::string> disk_image;
    std::queue<char *> buffer;
    std::queue<Image *> spin_image_buffer;
    std::queue<std::pair<Point2i,unsigned>> pyramidBuilderQ;

    int windowWidth = 3;
    int maxIndex = -1;

    std::atomic<bool> compositing = true;
    std::atomic<bool> tileEmbeddingComplete = false;
    std::atomic<int> components = 0;
    std::atomic<int> diskCount = 0;
    std::atomic<int> loaderCount = 0;
    std::atomic<int> matchableCount = 0;
    std::atomic<int> regCount = 0;


    std::vector<DataObserver *> observers;

    Poco::Thread disk_thread, Q_thread, composite_thread, postprocessor_thread,inference_thread;

    Poco::Event inferenceWait;
    Poco::Event compositeWait;
    Poco::Event cacheAlert;

    ICudaEngine *blurEngine = nullptr;
    IExecutionContext *blurCtx = nullptr;
    cudaStream_t blurStream{};
    char *blurInputs = nullptr;
    float *blurOutputs = nullptr;
    bool outstandingBlurInference = false;
    std::queue<Image*> blurMeticQ;
    std::vector<Image*> blurImagesInProcess;

    // v DEBUG v
    // void increment_match_counter(bool trueForUpFalserDown,long imgIdx);

    bool run() override;

    bool core_run();

    bool spin_run();

    void Q_blur_metric(Image* image);

    void launch_blur_metric();

    void receive_blur_metric();

    void process_blur_Q();

    void load_blur_engine();

    void clean_up_blur_engine() const;

    std::pair<Image*,bool> get_most_recent_resolved_frame(Image* _fromImage, bool _acceptRoot);

    std::vector<std::pair<Image *,Rect>> get_overlapping_frames(Rect _regionInComponentSpace, int _componentIndex);

#ifdef HAVE_OPENCV_CUDAARITHM
    int GPU_select_cuda_device(int _priority = 0);

    // void align_and_rebuild();

    void load_delaunay_images_to_GPU(int _componentIndex);

    void push_SIFT_matches(std::vector<std::pair<Image*,Image*>>& _newOverlaps, Image *_image);
#endif
    Point2f get_AbC_relative_from_relative(unsigned int _srcCompIdx, Point2f _srcAbC, unsigned int _requestedCompIdx);

    // void push_pyramid_builder_Q(Point2i _index, unsigned _componentIndex);

    bool has_flatfield(int label);

    void mark_neighbors_as_underexposed(unsigned long index);

    std::string get_flatfield_path(int label, bool &ffAlreadySet);

    cuda::GpuMat get_flatfield(int label);

    void set_flatfield(int label, const cuda::GpuMat& ffGpu);

    void update_last_frame(cv::Rect_<float> _rectInScale1Space, bool showAsCircle, int _component_index,
                           std::string _label, float _scale);

    void get_last_frame(cv::Rect_<float> &_rectInBaseSpace, bool &showAsCircle, int &_lastComponentIndex,
                        std::string &_magLabel, float &_lastScale);

    void pass_image(Image *, unsigned long _image_index = 0, bool saveImg = false);

    bool sufficient_distance(pathCam::Vec2 _coordsInQuestion, int _componentIdx);

    void set_match(unsigned long _image_idx, unsigned long _prev_idx, Match *_m, bool _invert = true);

    void clear_buffer(int _image_idx);

    void set_scale_and_offset(unsigned int component_index, double scale, Point2f offset) {
      scaleRepoMutex.lock();
      scaleRepo[component_index] = {scale, offset};
      scaleRepoMutex.unlock();
    }

    bool get_scale_and_offset(unsigned int component_index, double &_scale, Point2f &_offset);


    // FRONT END INTERACTION
    std::shared_ptr<MRTiledImageSet> get_MRimage_reference();

    void add_observer(DataObserver *new_observer) {
      observers.push_back(new_observer);
    }

    void update_observers();

    void notify_observers();

    void cleanup_and_reset();

    bool create_mag_label_to_scale_lookup(std::unordered_map<int,float>& _lookup);

    void set_input_file(const std::string &_path){inputFileOverride = _path;}

    std::string set_slide_label(std::string name = "");

    std::string get_slide_label() const{return currentSlideLabel;}

    std::string get_slide_label(int slideIdx);

    Poco::Path make_working_directory() const;

    int get_num_slides() const {return numSlides;}

    std::vector<std::string> get_preconfig_anno_labels() const {
      std::vector<std::string> res(preconfiguredAnnoLabels.size());
      for (int i = 0; i < preconfiguredAnnoLabels.size(); ++i) {
        res[i] = preconfiguredAnnoLabels[i].name;
      }
      return res;
    }

    bool segment_with_SAM(std::vector<Point3f> &_clicks, int _segID, int _slideIdx);

    int increment_and_get_components() { return components++; }

    void add_image(Image *image, long index);

    void add_registration(RegInfo* regInfo);

    Image* get_image_ref(unsigned long);

    std::vector<Image *> get_image_ref(const std::vector<long int> &);

    std::vector<RegInfo*> get_reg_ref(const std::vector<unsigned long>&);

    RegInfo* get_reg_ref(long image_idx);

    std::vector<Image *> get_component_image_refs(unsigned long component);

    std::vector<RegInfo*> get_Q_front(bool _pop);

    std::vector<std::tuple<int,int,unsigned int>> get_tile_embed_Q_front();

    void get_sift_data_Q_front(std::vector<Image *> &_images);

    std::vector<std::pair<Image*,Image*>> get_sift_match_Q_front(std::vector<Image *> &_images);

    void push_tile_embed_Q(std::vector<Point2i>& _tiles, unsigned int _componentIndex);

    Image *get_Q_front_Spin();

    bool compositeQ_empty();

    void push_compositeQ(RegInfo* _regInfo);

    void add_new_component(unsigned long image_index, cv::Size image_size, unsigned int component_index);

    int add_new_component_Q(unsigned long image_index, cv::Size image_size);

    void run_agg_classify();




  };

}
#endif /* StreamCam_hpp */
