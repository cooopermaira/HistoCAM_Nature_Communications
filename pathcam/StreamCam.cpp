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


  StreamCam::StreamCam(LayeredConfiguration::Ptr config) : BatchCam(config), buffer_mutex(new Poco::FastMutex()),
                                                           image_mutex(new Poco::RWLock()),
                                                           reg_results_mutex(new Poco::RWLock),
                                                           resize_mmatch_mutex(new Poco::RWLock()),
                                                           compositeQ_mutex(new Poco::FastMutex()),
                                                           component_mutex(new Poco::FastMutex()),
                                                           resize_buffer_mutex(new Poco::FastMutex()){
    MRimage.reset(new MRTiledImageSet());
    JobQ = new JobQueue(10, 10);
    reg_results.resize(1, RegInfo(true, Vec2(0, 0), true, 0));
    reg_results[0].index = 0;
    //variancesForDebug.resize(2266);

  }


  void StreamCam::set_match(unsigned long image_idx, unsigned long prev_idx) {
    resize_mmatch_mutex->readLock();
    matchM.match[image_idx][prev_idx] = new Match(matchM.match[prev_idx][image_idx]);
    resize_mmatch_mutex->unlock();
  }


  bool StreamCam::spin_run() {

    std::cout << "spin_run started " << std::endl;

    auto *qm = new QManager(this);
    Q_thread.start(qm);

    auto *cm = new CompositeManager(this);
    composite_thread.start(cm);

    Q_thread.join();
    composite_thread.join();

    std::cout << "spin_run done" << std::endl;

    return true;
  }

  bool StreamCam::run() {
    auto start = std::chrono::high_resolution_clock::now();

    auto dr = new DiskReader(this);
    disk_thread.start(dr);

    auto *qm = new QManager(this);
    Q_thread.start(qm);

    auto *cm = new CompositeManager(this);
    composite_thread.start(cm);

    disk_thread.join();
    Q_thread.join();
    composite_thread.join();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    return true;
  }

  void StreamCam::add_image(Image *image, unsigned long index) {
    image_mutex->writeLock();
    unsigned long size = images.size();
    if (index >= size) {
      images.resize(index + 100);

      resize_mmatch_mutex->writeLock();
      matchM.resize(index + 100);
      resize_mmatch_mutex->unlock();

      reg_results_mutex->writeLock();
      reg_results.resize(index + 100, RegInfo());
      reg_results_mutex->unlock();
    }
    images[index] = image;
    image_mutex->unlock();
  }

  void StreamCam::add_registration(pathCam::RegInfo regInfo) {
    reg_results_mutex->writeLock();
    auto index = regInfo.index;
    reg_results[index]=regInfo;
    reg_results_mutex->unlock();
    regCount++;
  }

  unsigned int StreamCam::get_last_active_component(unsigned long image_index) {
    for (unsigned long i = image_index; i > 0; i--) {
      if (reg_results[i-1].component_membership != reg_results[image_index].component_membership){
        return reg_results[i-1].component_membership;
      }
    }
  }

  std::vector<Image *> StreamCam::get_image_refs(std::vector<unsigned long int> indexes) {
    std::vector<Image *> temp;

    image_mutex->readLock();
    for (unsigned int i = 0; i < indexes.size(); i++) {
      if(images[indexes[i]] == nullptr){ continue; }
      temp.push_back(images[indexes[i]]);
    }
    image_mutex->unlock();

    return temp;
  }

  Image *StreamCam::get_image_ref(unsigned long int index) {
    Image *temp;
    image_mutex->readLock();
    temp = images[index];
    image_mutex->unlock();
    return temp;
  }

  void StreamCam::add_new_component(unsigned long image_index, cv::Size image_size) {
    auto component_index = increment_and_get_components();
    auto ri = RegInfo(true, Vec2(0.0, 0.0), true, component_index);
    ri.index = image_index;
    ri.resolved = true;
    ri.matchedTo = image_index; //this is a root image, it has no match
    add_registration(ri);


    //delete these pointers when destroyed
    auto *temp = new CompositeVoronoi(this, image_size, component_index);
    component_mutex->lock();
    composites.push_back(temp);
    if (composites.size() == 1) {
      temp->imagePyramid->set_scale(1);
      temp->imagePyramid->set_offset(Point2f(0, 0));
    } else {
      temp->imagePyramid->set_scale(0);
      temp->imagePyramid->set_offset(Point2f(0,0));
      auto xcm = new XCompRunnable(this,image_index,component_index);
      JobQ->add_runnable(xcm);
    }
    component_mutex->unlock();
    temp->update(std::vector<RegInfo>{reg_results[image_index]});
  }

  std::vector<Image *> StreamCam::get_component_image_refs(unsigned long component) {
    auto dm = composites[component]->delaunayMembers;
    std::vector<unsigned long> res(dm.size());
    int i = 0;
    for (auto [key,value] : dm){
      res[i] = dm[key];
      i++;
    }
    return get_image_refs(res);
  }

  std::vector<RegInfo> StreamCam::get_Q_front() {
    compositeQ_mutex->lock();
    std::vector<RegInfo> temp = compositeBatch.front();
    compositeBatch.pop();
    compositeQ_mutex->unlock();
    return temp;
  }

  Image* StreamCam::get_Q_front_Spin() {
    resize_buffer_mutex->lock();
    Image *temp = spin_image_buffer.front();
    spin_image_buffer.pop();
    resize_buffer_mutex->unlock();
    return temp;
  }

  void StreamCam::pass_image(Image *image, unsigned long _image_index) {
    LoaderLogicRunnable *llr = new LoaderLogicRunnable(this, image, _image_index);
    loaderCount++;
    JobQ->add_runnable(llr);
  }

  void StreamCam::push_compositeQ(RegInfo index) {
    compositeQ_mutex->lock();
    if (compositeBatch.empty() || compositeBatch.back().size() >= 1) {
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

  std::shared_ptr<MRTiledImageSet> StreamCam::get_image_reference() {
    return MRimage;
  }

  void RunnableIntermediate::waitOnThisGuy() {
    someoneWaitingOnJobCompleteEvent = true;
    if(!successful) {
      jobComplete.wait();
    }
    someoneWaitingOnJobCompleteEvent = false;
  }
}
