//
//  StreamCam.cpp
//  pathCamLib
//
//  Created by cooper maira on 11/24/23.
//

#include "pathCam.h"

namespace pathCam {
    using Poco::AutoPtr;
    using Poco::Path;
    using Poco::Util::XMLConfiguration;
    using Poco::Util::LayeredConfiguration;
    using Poco::Logger;
    using Poco::LogStream;
    using Poco::Environment;
    using Poco::FileChannel;


    StreamCam::StreamCam(LayeredConfiguration::Ptr config): BatchCam(config), buffer_mutex(new Poco::FastMutex()),
                                                            image_mutex(new Poco::FastMutex()),
                                                            resize_mmatch_mutex(new Poco::FastMutex()),
                                                            compositeQ_mutex(new Poco::FastMutex()),
                                                            component_mutex(new Poco::FastMutex()),
                                                            resize_buffer_mutex(new Poco::FastMutex()) {
        JobQ = new JobQueue(15, 15);
        reg_results.resize(1, RegInfo(true, Vec2(0, 0), true, 0));
        reg_results[0].index = 0;
    }


    void StreamCam::set_match(unsigned long image_idx, unsigned long prev_idx) {
        resize_mmatch_mutex->lock();
        matchM.match[image_idx][prev_idx] = new Match(matchM.match[prev_idx][image_idx]);
        resize_mmatch_mutex->unlock();
    }


    bool StreamCam::spin_run() {
        Poco::Thread Q_thread, composite_thread;

        std::cout << "spin_run started " << std::endl;

        QManager *qm = new QManager(this);
        Q_thread.start(qm);

        CompositeManager *cm = new CompositeManager(this);
        composite_thread.start(cm);

        Q_thread.join();
        composite_thread.join();

        std::cout << "spin_run done" << std::endl;

        return true;
    }

    bool StreamCam::run() {
        //cv::namedWindow("display");
        //cv::namedWindow("display2");
        Poco::Thread stream_thread, Q_thread, composite_thread;

        auto ds = new DiskReader(this);
        //auto ds = new DiskStreamer(this);
        stream_thread.start(ds);
        //ds->run();
        auto start = std::chrono::high_resolution_clock::now();

        QManager *qm = new QManager(this);
        Q_thread.start(qm);

        CompositeManager *cm = new CompositeManager(this);
        composite_thread.start(cm);

        stream_thread.join();
        Q_thread.join();
        composite_thread.join();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << duration.count() << std::endl;
        return true;
    }

    void StreamCam::add_image(Image* image, unsigned long index) {
        image_mutex->lock();
        unsigned long size = images.size();
        if (index >= size) {
            images.resize(index + 100);
            resize_mmatch_mutex->lock();
            matchM.resize(index + 100);
            reg_results.resize(index + 100, RegInfo());
            visited.resize(index + 100, false);
            resize_mmatch_mutex->unlock();
        }
        images[index] = image;
        image_mutex->unlock();
    }

    unsigned long int StreamCam::add_image(Image *image) {
        unsigned long int index;
        image_mutex->lock();
        images.push_back(image);
        index = images.size() - 1;

        if (index % 100 == 0) {
            resize_mmatch_mutex->lock();
            matchM.resize(index + 100);
            reg_results.resize(index + 100, RegInfo());
            visited.resize(index + 100, false);
            resize_mmatch_mutex->unlock();
        }
        image_mutex->unlock();
        return index;
    }

    std::vector<Image *> StreamCam::get_image_refs(std::vector<unsigned long int> indexes) {
        std::vector<Image *> temp;

        image_mutex->lock();
        for (unsigned int i = 0; i < indexes.size(); i++) {
            temp.push_back(images[indexes[i]]);
        }
        image_mutex->unlock();

        return temp;
    }

    Image *StreamCam::get_image_ref(unsigned long int index) {
        Image *temp;
        image_mutex->lock();
        temp = images[index];
        image_mutex->unlock();
        return temp;
    }

    void StreamCam::add_new_component(unsigned long image_index, cv::Size image_size) {
        reg_results[image_index] = RegInfo(true, Vec2(0.0, 0.0), true, increment_and_get_components());
        reg_results[image_index].index = image_index;
        reg_results[image_index].resolved = true;
        reg_results[image_index].matchedTo = image_index;

        //delete these pointers when destroyed
        auto *temp = new CompositeVoronoi(this, image_size);
        temp->update(std::vector<RegInfo>{reg_results[image_index]});
        component_mutex->lock();
        composites.push_back(temp);
        component_mutex->unlock();
    }

    std::vector<RegInfo> StreamCam::get_Q_front() {
        compositeQ_mutex->lock();
        std::vector<RegInfo> temp = compositeBatch.front();
        compositeBatch.pop();
        compositeQ_mutex->unlock();
        return temp;
    }

    Image *StreamCam::get_Q_front_Spin() {
        resize_buffer_mutex->lock();
        Image *temp = spin_image_buffer.front();
        spin_image_buffer.pop();
        resize_buffer_mutex->unlock();
        return temp;
    }

    void StreamCam::pass_image(Image *image,unsigned long sort_order) {
        LoaderLogicRunnable *llr = new LoaderLogicRunnable(this, image, sort_order);
        loaderCount++;
        JobQ->add_runnable(llr);
    }

    void StreamCam::push_compositeQ(RegInfo index) {
        compositeQ_mutex->lock();
        if (compositeBatch.empty() || compositeBatch.back().size() >= 20) {
            compositeBatch.push(std::vector<RegInfo>());
        }
        compositeBatch.back().push_back(index);
        compositeQ_mutex->unlock();
    }

    bool StreamCam::compositeQ_empty() {
        compositeQ_mutex->lock();
        bool isEmpty = compositeBatch.size() == 0;
        compositeQ_mutex->unlock();
        return isEmpty;
    }
}
