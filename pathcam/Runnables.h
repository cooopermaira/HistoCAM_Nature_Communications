//
//  Runnables.hpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#ifndef Runnables_h
#define Runnables_h

#include <stdio.h>
#include "pathCam.h"
#include "Poco/Runnable.h"

namespace pathCam {
  class JobQueue;

  class RunnableIntermediate : public Poco::Runnable {
  public:
    RunnableIntermediate(unsigned long image_index, int jobTypeFlag) : image_index(image_index),
                                                                       jobTypeFlag(jobTypeFlag), jobComplete(false),
                                                                       someoneWaitingOnJobCompleteEvent(false) {}

    Poco::Event jobComplete;
    bool someoneWaitingOnJobCompleteEvent;
    unsigned long sort_order = 0;
    int jobTypeFlag = 0;
    unsigned long image_index;
    int jobRefNumber = 0;
    std::atomic<bool> successful = false;

    void waitOnThisGuy();
  };

  class RebuildRunnable : public RunnableIntermediate {
  public:
    int dtVertex;
    unsigned long imageIndex;
    CompositeVoronoi *composite;
    CompositeManager *cm;
    Mat polyMaskOutput;
    std::vector<Point2i> rebuildTiles;

    RebuildRunnable(CompositeVoronoi *_composite, int _dtVertex, unsigned long _imageIndex,
                    std::vector<Point2i> _rebuildTiles, Mat _polyMaskOutput);

    virtual void run();
  };

  class DebayerRunnable : public pathCam::RunnableIntermediate {
  public:
    explicit DebayerRunnable(pathCam::Image *image, Mat flat_field, Poco::Path outfile, std::vector<double> *_blur,
                             std::vector<std::string> *_names, unsigned long _sort_order) : image(image),
                                                                                            flatfield(flat_field),
                                                                                            outfile(outfile),
                                                                                            RunnableIntermediate(
                                                                                                _sort_order, 0),
                                                                                            blur(_blur),
                                                                                            names(_names) {}

    Mat flatfield;
    std::vector<double> *blur;
    std::vector<std::string> *names;
    Poco::Path outfile;
    pathCam::Image *image;

    virtual void run();
  };


  class CompositeManager : public Poco::Runnable {
  private:
    StreamCam *parent;

  public:
    bool successful;

    std::atomic<int> rebuildJobsOutstanding = 0;
    Poco::Event rebuildJobsComplete;

    CompositeManager(StreamCam *parent);

    virtual void run();

    void align_new_comp();

    void perform_global_alignment();

    void check_render_info();

    void save_components_to_disk();

    void decrement_rebuild_jobs_outstanding();
  };

  class RegistrationRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    RegInfo *regInfo;
  public:
    RegistrationRunnable(StreamCam *parent, RegInfo *regInfo) :
        parent(parent),
        regInfo(regInfo),
        RunnableIntermediate(regInfo->index, 3) {};

    virtual void run();


    std::pair<bool, Vec2> trace_to_root(unsigned long index);
  };


  //loader class takes data from disk streamer/microscope and prepares matchable jobs
  class LoaderLogicRunnable : public RunnableIntermediate {
  private:
    bool additionalSiftReg;
    bool saveImg;
    StreamCam *parent;
    Image *image;

  public:

    LoaderLogicRunnable(StreamCam *parent, Image *image, unsigned long image_idx, bool additionalFullReg,
                        bool saveImg = false) : image(
        image), parent(parent), additionalSiftReg(additionalFullReg), saveImg(saveImg),
                                                RunnableIntermediate(image_idx, 1) {};

    virtual void run();
  };


  class InferenceManager : public Poco::Runnable {
  private:
    StreamCam *parent;
    Mat threeChannelPreallocated;
    torch::Device device;
    torch::jit::script::Module tileEncoderModel;
    PyObject* slideAggregator;
    PyObject *pModule;
    Poco::Thread thread;
    Poco::RunnableAdapter<InferenceManager> adapter;

  public:



    InferenceManager(StreamCam *parent);
    
    virtual void run();
    void run_slide_analysis();
    void initialize_aggregator();
    PyObject* tensorToList2(const torch::Tensor& tensor);
    void run_slide_aggregation();

    int minx;
    int miny;
    int cropedDim;
    int embedSize;

    bool aggregatorReady = false;

    torch::Tensor mean;
    torch::Tensor stddv;
    torch::Tensor tileEmbeds;

    std::map<Point2i,unsigned long,PointComparator> tileCoordToTensorIndex;

    Poco::Event aggregatorWait;
    Poco::FastMutex *aggregatorMutex;
  };


  class QManager : public Poco::Runnable {
  private:
    StreamCam *parent;

  public:
    QManager(StreamCam *parent);

    virtual void run();
  };

  class DiskReader : public Poco::Runnable {
  private:
    StreamCam *parent;

  public:

    DiskReader(StreamCam *parent);

    virtual void run();
  };

  class DiskStreamer : public RunnableIntermediate {
  private:
    StreamCam *parent;
    std::string imageFile;

  public:

    DiskStreamer(StreamCam *parent, std::string file, unsigned long sort_order);

    //DiskStreamer(StreamCam *parent);
    virtual void run();
  };


  class ImageToTileCopyRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    Image *image;
    unsigned int component_membership;
    Point2i tile;


  public:
    ImageToTileCopyRunnable(StreamCam *parent, Image *image, unsigned int component_membership,
                            Point2i tile, unsigned long sort_order) : RunnableIntermediate(sort_order, 0),
                                                                      parent(parent),
                                                                      image(image),
                                                                      tile(tile),
                                                                      component_membership(component_membership) {};

    virtual void run();
  };


  class SingleMatchRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    unsigned long image_idx1, image_idx2;
    unsigned int component_membership;
    int edgeNumber;
  public:

    SingleMatchRunnable(StreamCam *parent, unsigned long image_idx1, unsigned long image_idx2,
                        unsigned int component_membership, int edgeNumber, unsigned long sort_order);

    virtual void run();
  };


  class XCompRunnable : public RunnableIntermediate {
  public:
    XCompRunnable(StreamCam *parent, unsigned long image_idx, unsigned int componentMembershipSelf) : image_idx(
        image_idx),
                                                                                                      componentMembership(
                                                                                                          componentMembershipSelf),
                                                                                                      parent(parent),
                                                                                                      RunnableIntermediate(
                                                                                                          image_idx,
                                                                                                          0) {};
    StreamCam *parent;
    unsigned long image_idx;
    unsigned int componentMembership;

    virtual void run();

    bool match_to_images(Image *selfImage, std::vector<Image *> images);

    void extract_multilevel_keypoints(Image *image);
  };

  class ReverseMatchRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    unsigned long image_idx;
    unsigned long start_from_idx;

  public:
    ReverseMatchRunnable(StreamCam *parent, unsigned long image_idx, unsigned long start_from_idx);

    virtual void run();
  };

  class MatchRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    unsigned long image_idx;

  public:

    MatchRunnable(StreamCam *parent, unsigned long image_idx);

    virtual void run();
  };
}
#endif /* Runnables_h */
