//
//  StreamCam.h
//  pathCamLib
//
//  Created by cooper maira on 11/24/23.
//

#ifndef StreamCam_h
#define StreamCam_h

#include <stdio.h>

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
    StreamCam(Poco::Util::LayeredConfiguration::Ptr config);

    ~StreamCam() { delete buffer_mutex, delete image_mutex; }

    Poco::RWLock *image_mutex;
    Poco::RWLock *reg_results_mutex;
    Poco::RWLock *resize_mmatch_mutex;
    Poco::FastMutex *resize_buffer_mutex;
    Poco::FastMutex *buffer_mutex;
    Poco::FastMutex *compositeQ_mutex;
    Poco::FastMutex *component_mutex;
    Poco::FastMutex *lastFrameMutex;
    Poco::FastMutex *scaleRepoMutex;
    Poco::FastMutex *inferenceQMutex;
    Poco::FastMutex *siftQMutex;
    Poco::FastMutex *pixelDistanceMutex;

    cv::Rect_<float> lastFrame;
    int lastComponentIndex;
    std::string lastLabel;
    bool showAsCircle;

    bool microscopeInput;
    bool recordingMode = false;

    bool opencvWithCuda = false;
    int compositorCudaDevice;
    int siftCudaDevice = -1;

    int lastActiveComponent = 0;
    int maxTilesPerBatch = 512;
    int maxMatchesPerPull = 50;
    int minPixelDistanceBetweenFrames;
    std::vector<Vec2> lastAcceptedCoords;


    std::vector<double> labelScales = { -10.0, 1.0, 0.5, 0.2, 0.1, 0.05 };

    std::map<int, std::pair<double, Point2f>> scaleRepo;
    std::map<std::tuple<int,int,int>,int> tileCoordToClass;

    Mat flat_field2X;
    Mat flat_field4X;
    Mat flat_field10X;
    Mat flat_field20X;
    Mat circleMask;
    Mat regCircleMask;

    std::shared_ptr<MRTiledImageSet> MRimage;

    std::vector<std::pair<std::string, double>> debugImageBlurWithNames;
    std::vector<double> debugImageBlur;
    CompositeManager *cm;
    QManager *qm;
    DiskReader *dr;
    InferenceManager *im;
    PostProcessManager* ppm;
    SiftFeatureMatcher* sfm;
    AccessSAM *as;
    JobQueue *JobQ;

    //std::vector < double > variancesForDebug;
    std::vector<CompositeVoronoi *> composites;
    std::vector<bool> visited;

    std::queue<std::tuple<unsigned long, cv::Size, unsigned int> > newComponentQ;
    //UniqueQueue<std::pair<Point2i,unsigned int>,PairHash> tileEmbedQ;
    UniqueQueue<std::tuple<int,int,unsigned>,TupleHash>tileEmbedQ;
    std::deque<std::vector<std::pair<Image*,Image*>>> siftMatchQueue;
    std::queue<Image*> siftDataQueue;
    std::queue<std::string> disk_image;
    std::queue<char *> buffer;
    std::queue<Image *> spin_image_buffer;

    int windowWidth = 3;
    int maxIndex = -1;

    std::atomic<bool> compositing = true;
    std::atomic<bool> tileEmbeddingComplete = false;
    std::atomic<unsigned int> components = 0;
    std::atomic<unsigned int> diskCount = 0;
    std::atomic<unsigned int> loaderCount = 0;
    std::atomic<int> matchableCount = 0;
    std::atomic<unsigned int> regCount = 0;

    std::vector<DataObserver *> observers;

    Poco::Thread disk_thread, Q_thread, composite_thread, postprocessor_thread,inference_thread;

    Poco::Event inferenceWait;
    Poco::Event compositeWait;

    bool run();

    bool spin_run();

#ifdef HAVE_OPENCV_CUDAARITHM
    int GPU_select_cuda_device(int _priority = 0);

    void align_and_rebuild();

    void load_delaunay_images_to_GPU(int _componentIndex);

    void push_SIFT_matches(std::vector<std::pair<Image*,Image*>>& _newOverlaps, Image *_image);
#endif
    Point2f get_AbC_relative_from_relative(unsigned int _srcCompIdx, Point2f _srcAbC, unsigned int _requestedCompIdx);

    bool has_flatfield(int label);

    void mark_neighbors_as_underexposed(unsigned long index);

    std::string get_flatfield(int label);

    void update_last_frame(cv::Rect_<float> _rectInScale1Space, bool showAsCircle, int _component_index,
                           std::string _label);

    void get_last_frame(cv::Rect_<float> &_rectInScale1Space, bool &showAsCircle, int &_lastComponentIndex,
                        std::string &_magLabel);

    void pass_image(Image *, unsigned long _image_index = 0, bool saveImg = false);

    bool sufficient_distance(pathCam::Vec2 _coordsInQuestion, int _componentIdx);

    void set_match(unsigned long _image_idx, unsigned long _prev_idx, Match *_m, bool _invert = true);

    void set_scale_and_offset(unsigned int component_index, double scale, Point2f offset) {
      scaleRepoMutex->lock();
      scaleRepo[component_index] = {scale, offset};
      scaleRepoMutex->unlock();
    }

    bool get_scale_and_offset(unsigned int component_index, double &_scale, Point2f &_offset);

    std::shared_ptr<MRTiledImageSet> get_image_reference();

    void add_observer(DataObserver *new_observer) {
      observers.push_back(new_observer);
    }

    void update_observers();

    void notify_observers();

    bool segment_with_SAM(std::vector<Point3f> &_clicks, int _segID);

    unsigned int increment_and_get_components() { return components++; }

    void add_image(Image *image, unsigned long index);

    void add_registration(RegInfo* regInfo);

    Image* get_image_ref(unsigned long);

    std::vector<Image *> get_image_ref(const std::vector<unsigned long>&) const;

    std::vector<RegInfo*> get_reg_ref(const std::vector<unsigned long>&);

    RegInfo* get_reg_ref(unsigned long image_idx);

    std::vector<Image *> get_component_image_refs(unsigned long component);

    std::vector<RegInfo*> get_Q_front();

    std::vector<std::tuple<int,int,unsigned int>> get_tile_embed_Q_front();

    void get_sift_data_Q_front(std::vector<Image *> &_images);

    std::vector<std::pair<Image*,Image*>> get_sift_match_Q_front(std::vector<Image *> &_images);

    void push_tile_embed_Q(std::vector<Point2i>& _tiles, unsigned int _componentIndex);

    Image *get_Q_front_Spin();

    bool compositeQ_empty();

    void push_compositeQ(RegInfo* index);

    void add_new_component(unsigned long image_index, cv::Size image_size, unsigned int component_index);

    void add_new_component_Q(unsigned long image_index, cv::Size image_size);

    void run_agg_classify();


  };

}
#endif /* StreamCam_hpp */
