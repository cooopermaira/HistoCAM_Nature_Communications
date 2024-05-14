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
        RunnableIntermediate(unsigned long sort_order) : sort_order(sort_order) {
        }

        bool operator < (const RunnableIntermediate& other) const {
            return sort_order > other.sort_order;
        }

        unsigned long sort_order = 0;
    };

    class CompositeManager : public Poco::Runnable {
    private:
        StreamCam *parent;

    public:
        bool successful;

        CompositeManager(StreamCam *parent);

        virtual void run();
    };

    class RegistrationRunnable : public RunnableIntermediate {
    private:
        StreamCam *parent;
        unsigned long index;
    public:
        RegistrationRunnable(StreamCam *parent, unsigned long index, unsigned long sort_order) : parent(parent),
            index(index), RunnableIntermediate(sort_order) {
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
            successful(true), RunnableIntermediate(sort_order) {
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
