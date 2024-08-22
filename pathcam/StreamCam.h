//
//  StreamCam.h
//  pathCamLib
//
//  Created by cooper maira on 11/24/23.
//

#ifndef StreamCam_h
#define StreamCam_h

#include <stdio.h>
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

    cv::Rect_<float> lastFrame;
    bool showAsCircle;

    std::map<unsigned int, std::pair<double, Point2f>> scaleRepo;

    Mat flat_field2X;
    Mat flat_field4X;

    std::shared_ptr<MRTiledImageSet> MRimage;

    std::vector<std::pair<std::string, double>> debugImageBlurWithNames;
    std::vector<double> debugImageBlur;
    CompositeManager *cm;
    QManager *qm;
    DiskReader *dr;

    bool run();

    bool spin_run();

    void update_last_frame(cv::Rect_<float> _rectInScale1Space, bool showAsCircle);

    void get_last_frame(cv::Rect_<float> &_rectInScale1Space, bool &showAsCircle);

    void pass_image(Image *, unsigned long _image_index = 0);

    void set_match(unsigned long image_idx, unsigned long prev_idx, Match *m);

    void set_scale_and_offset(unsigned int component_index, double scale, Point2f offset) {
      scaleRepoMutex->lock();
      scaleRepo[component_index] = {scale, offset};
      scaleRepoMutex->unlock();
    }

    bool get_scale_and_offset(unsigned int component_index, double &_scale, Point2f &_offset);

    bool microscopeInput;

    std::shared_ptr<MRTiledImageSet> get_image_reference();

    void add_observer(DataObserver *new_observer) {
      observers.push_back(new_observer);
    }

    void update_observers() {
      for (unsigned int i = 0; i < observers.size(); i++) {
        MRimage->update_bounds();
        observers[i]->update();
      }
    }

  protected:
    unsigned int increment_and_get_components() { return components++; }

    void add_image(Image *image, unsigned long index);

    void add_registration(RegInfo* regInfo);

    bool get_registration(unsigned long image_idx, RegInfo* res);

    RegInfo* get_registration(unsigned long image_idx);

    JobQueue *JobQ;

    Image *get_image_ref(unsigned long int);

    std::vector<Image *> get_image_refs(std::vector<unsigned long int>);

    std::vector<Image *> get_component_image_refs(unsigned long component);

    std::vector<RegInfo*> get_Q_front();

    Image *get_Q_front_Spin();

    bool compositeQ_empty();

    void push_compositeQ(RegInfo* index);

    //void reg_spanning_tree(unsigned int root_idx, Vec2 offset);
    void add_new_component(unsigned long image_index, cv::Size image_size, unsigned int component_index);

    void add_new_component_Q(unsigned long image_index, cv::Size image_size);

    //std::vector < double > variancesForDebug;
    std::vector<CompositeVoronoi *> composites;
    std::vector<bool> visited;

    std::queue<std::tuple<unsigned long, cv::Size, unsigned int> > newComponentQ;
    std::queue<std::string> disk_image;
    std::queue<char *> buffer;
    std::queue<Image *> spin_image_buffer;

    std::atomic<bool> compositing = true;
    std::atomic<unsigned int> components = 0;
    std::atomic<unsigned int> diskCount = 0;
    std::atomic<unsigned int> loaderCount = 0;
    std::atomic<unsigned int> matchableCount = 0;
    std::atomic<unsigned int> regCount = 0;

    std::atomic<int> debugMatchSuspendThread = 0;

    pathCam::ConsecQ RegistrationConsecQ;

    std::vector<DataObserver *> observers;

    Poco::Thread disk_thread, Q_thread, composite_thread;

  };

}
#endif /* StreamCam_hpp */
