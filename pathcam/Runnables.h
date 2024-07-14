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
    RunnableIntermediate(unsigned long sort_order) : sort_order(sort_order), jobComplete(false),someoneWaitingOnJobCompleteEvent(false) {
    }
    Poco::Event jobComplete;
    bool someoneWaitingOnJobCompleteEvent;
    unsigned long sort_order = 0;
    void waitOnThisGuy();
  };

  class DebayerRunnable : public pathCam::RunnableIntermediate {
  public:
    explicit DebayerRunnable(pathCam::Image *image, Poco::Path outfile) : image(image), outfile(outfile),
                                                                          RunnableIntermediate(0) {}

    Poco::Path outfile;
    pathCam::Image *image;

    virtual void run();
  };

  class CompositeManager : public Poco::Runnable {
  private:
    StreamCam *parent;

  public:
    bool successful;

    CompositeManager(StreamCam *parent);

    virtual void run();

    void perform_global_alignment();

    void save_components_to_disk();
  };

  class RegistrationRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    unsigned long index;
  public:
    RegistrationRunnable(StreamCam *parent, unsigned long index, unsigned long sort_order) : parent(parent),
                                                                                             index(index),
                                                                                             RunnableIntermediate(
                                                                                                 sort_order) {
    };

    virtual void run();

    std::pair<bool, Vec2> trace_to_root(unsigned long index);
  };


  //loader class takes data from disk streamer/microscope and prepares matchable jobs
  class LoaderLogicRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    Image *image;

  public:
    bool successful;

    LoaderLogicRunnable(StreamCam *parent, Image *image, unsigned long sort_order) : image(image), parent(parent),
                                                                                     successful(false),
                                                                                     RunnableIntermediate(
                                                                                         sort_order) {
    };

    virtual void run();
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
    bool successful;

    DiskReader(StreamCam *parent);

    virtual void run();
  };

  class DiskStreamer : public RunnableIntermediate {
  private:
    StreamCam *parent;
    std::string imageFile;

  public:
    bool successful;

    DiskStreamer(StreamCam *parent, std::string file, unsigned long sort_order);

    //DiskStreamer(StreamCam *parent);
    virtual void run();
  };


  class ImageToTileCopyRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    Image* image;
    unsigned int component_membership;
    Point2i tile;


  public:
    ImageToTileCopyRunnable(StreamCam *parent, Image *image, unsigned int component_membership,
                            Point2i tile,unsigned long sort_order) : RunnableIntermediate(sort_order),
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
    XCompRunnable(StreamCam *parent, unsigned long image_idx, unsigned int componentMembershipSelf,
                  unsigned int componentMembershipMatchTo) : image_idx(image_idx),
                                                                                componentMembershipSelf(
                                                                                    componentMembershipSelf),
                                                                                componentMembershipMatchTo(
                                                                                    componentMembershipMatchTo),
                                                                                parent(parent),
                                                                                RunnableIntermediate(0) {};
    StreamCam *parent;
    unsigned long image_idx;
    unsigned int componentMembershipSelf;
    //grab this value from back() of compositeQ when job is launched
    unsigned int componentMembershipMatchTo;

    virtual void run();
    void extract_multilevel_keypoints(Image* image);
  };


  class MatchRunnable : public RunnableIntermediate {
  private:
    StreamCam *parent;
    unsigned long image_idx;

  public:
    bool successful;

    MatchRunnable(StreamCam *parent, unsigned long image_idx, unsigned long sort_order);

    virtual void run();
  };
}
#endif /* Runnables_h */
