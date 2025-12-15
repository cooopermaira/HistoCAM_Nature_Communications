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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void launch_drop_alpha_and_swap(char *dst, const char *src, int count);

void ensure1DHann(int w, int h, cuda::GpuMat &wx, cuda::GpuMat &wy, cudaStream_t stream = nullptr);

void launch_apply_hann_2d(cuda::GpuMat &wx, cuda::GpuMat &wy, cuda::GpuMat &win, cuda::GpuMat &magNorm,
                          cudaStream_t stream = nullptr);

void launch_CPS(const cuda::GpuMat &F,
                const cuda::GpuMat &G,
                cuda::GpuMat &CPS,
                float eps = 1e-9f,
                cudaStream_t stream = nullptr);

#ifdef __cplusplus
}
#endif

namespace pathCam {
  class JobQueue;

  class RunnableIntermediate : public Poco::Runnable {
  public:
    RunnableIntermediate(unsigned long image_index, int jobTypeFlag) : image_index(image_index),
                                                                       jobTypeFlag(jobTypeFlag), jobComplete(false),
                                                                       someoneWaitingOnJobCompleteEvent(false),
                                                                       unprocessed(true),
                                                                       precedingJobCount(0) {
    }

    Poco::Event jobComplete;
    bool someoneWaitingOnJobCompleteEvent;
    bool unprocessed;
    unsigned long sort_order = 0;
    unsigned long image_index;
    std::atomic<int> precedingJobCount;
    int jobTypeFlag = 0;
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
    explicit DebayerRunnable(pathCam::Image *image, Mat flat_field, Poco::Path outfile, std::vector<int> *_blur,
                             std::vector<std::string> *_names, unsigned long _sort_order) : image(image),
      flatfield(flat_field),
      outfile(outfile),
      RunnableIntermediate(
        _sort_order, 0),
      blur(_blur),
      names(_names) {
    }

    Mat flatfield;
    std::vector<int> *blur;
    std::vector<std::string> *names;
    Poco::Path outfile;
    pathCam::Image *image;

    virtual void run();
  };


  class PostProcessManager : public Poco::Runnable {
  public:
    PostProcessManager(StreamCam *_parent): parent(_parent) {
    }

    void run() override;

    StreamCam *parent;

    std::vector<PostProcessorBase *> postProcesses;
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

    void debug_termination_check();

    void save_components_to_disk();

    void push_remaining_tiles_for_inference();

    void decrement_rebuild_jobs_outstanding();

    void build_match_pairs(std::vector<std::pair<Image *, Image *> > &_imagePairsToMatch,
                           std::vector<RegInfo *> &_regs);

    cv::detail::MatchesInfo compute_matches_info(SiftData &_sift1, SiftData &_sift2, unsigned long _img1_idx,
                                                 unsigned long _img2_idx);

    void perform_SIFT_multires_bundle_adjustment();

    void submit_outstanding_jobs();
  };

  class RegistrationRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    RegInfo *regInfo;

  public:
    RegistrationRunnable(StreamCam *parent, RegInfo *regInfo) : parent(parent),
                                                                regInfo(regInfo),
                                                                RunnableIntermediate(regInfo->index, 3) {
    };

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
                                                  image), parent(parent), additionalSiftReg(additionalFullReg),
                                                saveImg(saveImg),
                                                RunnableIntermediate(image_idx, 1) {
    };

    virtual void run();

    void self_cancel(int label);
  };


  class InferenceManager : public Poco::Runnable {
  private:
    StreamCam *parent;
    Mat threeChannelPreallocated;
#ifdef HAVE_OPENCV_CUDAARITHM
    cuda::GpuMat threeChannelPrealGPU;
    char *bufferGPU_rcv;
    char *bufferGPU;
    char *bufferMemory;
#endif

    std::vector<std::vector<float> > tileEmbedVec;
    std::vector<std::vector<float> > coordsVec;
    std::vector<std::vector<float> > resultsVec;

    std::string embedFileOut;
    std::string coordsFileOut;
    std::string signalFileOut;

    std::string reportFileIn;
    std::string embedFileIn;
    std::string signalFileIn;

#if 0
    torch::Device device;
    torch::Tensor tileEmbeds;
#endif

    Poco::FastMutex tileEmbedMutex;

  public:
    explicit InferenceManager(StreamCam *parent);

    virtual void run();

    void run_agg_classify();

    Poco::Thread thread;
    std::vector<std::pair<int, int> > minShift;

    bool aggregatorReady = false;

    std::map<std::tuple<int, int, unsigned>, unsigned> tileCoordToTensorIndex;
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
                                                                      component_membership(component_membership) {
    };

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
        0) {
    };
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
  public:
    StreamCam *parent;
    unsigned long image_idx;


    MatchRunnable(StreamCam *parent, unsigned long image_idx);

    virtual void run();
  };
}
#endif /* Runnables_h */
