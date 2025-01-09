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
                                                           resize_buffer_mutex(new Poco::FastMutex()),
                                                           lastFrameMutex(new Poco::FastMutex()),
                                                           scaleRepoMutex(new Poco::FastMutex()),
                                                           cm(new CompositeManager(this)),
                                                           qm(new QManager(this)),
                                                           dr(new DiskReader(this)),
                                                           inferenceWait(false),
                                                           microscopeInput(true){
      //inferencing = false;
    if (inferencing) {
      inferenceQMutex = new Poco::FastMutex();
      im = new InferenceManager(this);
    }

    MRimage.reset(new MRTiledImageSet());
    JobQ = new JobQueue(10, 10);


    //lastFrame = Rect(0,0,image_width,image_height);
    circleMask = cv::Mat::zeros(image_height, image_width, CV_8U);
    cv::circle(circleMask, cv::Point(image_width / 2, image_height / 2), scope_radius, cv::Scalar(255),
               -1);
    regCircleMask = cv::Mat::zeros(image_height * scale_factor, image_width * scale_factor, CV_8U);
    cv::circle(regCircleMask, cv::Point(float(image_width / 2) * scale_factor, float(image_height / 2) * scale_factor),
               scope_radius, cv::Scalar(255),
               -1);

    if (flat_field_file_2x.getExtension() == "Raw") {
      char *buffer = new char[6464 * 4852];
      std::ifstream stream;
      stream.open(flat_field_file_2x.toString(), std::ios::binary);
      stream.read(buffer, 6464 * 4852);
      flat_field2X = cv::Mat(cv::Size(6464, 4852), CV_8U, buffer, Mat::AUTO_STEP);
      cvtColor(flat_field2X, flat_field2X, COLOR_BayerBG2BGR);
      delete buffer;
    } else {
      flat_field2X = cv::imread(flat_field_file_2x.toString());
    }

    flat_field2X.convertTo(flat_field2X, CV_32F);
    flat_field2X *= 1 / 170.0;

    if (flat_field_file_4x.getExtension() == "Raw") {
      char *buffer = new char[6464 * 4852];
      std::ifstream stream;
      stream.open(flat_field_file_4x.toString(), std::ios::binary);
      stream.read(buffer, 6464 * 4852);
      flat_field4X = cv::Mat(cv::Size(6464, 4852), CV_8U, buffer, Mat::AUTO_STEP);
      cvtColor(flat_field4X, flat_field4X, COLOR_BayerBG2BGR);
      delete buffer;
    } else {
      flat_field4X = cv::imread(flat_field_file_4x.toString());
    }

    flat_field4X.convertTo(flat_field4X, CV_32F);
    flat_field4X *= 1 / 170.0;

    if (flat_field_file_10x.getExtension() == "Raw") {
      char *buffer = new char[6464 * 4852];
      std::ifstream stream;
      stream.open(flat_field_file_10x.toString(), std::ios::binary);
      stream.read(buffer, 6464 * 4852);
      flat_field10X = cv::Mat(cv::Size(6464, 4852), CV_8U, buffer, Mat::AUTO_STEP);
      cvtColor(flat_field10X, flat_field10X, COLOR_BayerBG2BGR);
      delete buffer;
    } else {
      flat_field10X = cv::imread(flat_field_file_10x.toString());
    }
    flat_field10X.convertTo(flat_field10X, CV_32F);
    flat_field10X *= 1 / 170.0;

    if (flat_field_file_20x.getExtension() == "Raw") {
      char *buffer = new char[6464 * 4852];
      std::ifstream stream;
      stream.open(flat_field_file_20x.toString(), std::ios::binary);
      stream.read(buffer, 6464 * 4852);
      flat_field20X = cv::Mat(cv::Size(6464, 4852), CV_8U, buffer, Mat::AUTO_STEP);
      cvtColor(flat_field20X, flat_field20X, COLOR_BayerBG2BGR);
      delete buffer;
    } else {
      flat_field20X = cv::imread(flat_field_file_20x.toString());
    }
    flat_field20X.convertTo(flat_field20X, CV_32F);
    flat_field20X *= 1 / 170.0;

  }

  void StreamCam::update_last_frame(cv::Rect_<float> _rectInScale1Space, bool _showAsCircle) {
    lastFrameMutex->lock();
    lastFrame = _rectInScale1Space;
    showAsCircle = _showAsCircle;
    lastFrameMutex->unlock();
  }

  void StreamCam::get_last_frame(cv::Rect_<float> &_rectInScale1Space, bool &_showAsCircle) {
    lastFrameMutex->lock();
    _showAsCircle = showAsCircle;
    _rectInScale1Space = lastFrame;
    lastFrameMutex->unlock();
  }


  void StreamCam::set_match(unsigned long image_idx, unsigned long prev_idx, Match *m) {
    resize_mmatch_mutex->writeLock();
    matchM.match[image_idx][prev_idx] = new Match(m);
    resize_mmatch_mutex->unlock();
  }

  bool StreamCam::has_flatfield(int label) {
    if (label == Image::_2X || label == Image::_4X || label == Image::_10X || Image::_20X) {
      return true;
    }
    return false;
  }

  Mat StreamCam::get_flatfield(int label) {
    switch (label) {
      case Image::_2X:
        return flat_field2X;
      case Image::_4X:
        return flat_field4X;
      case Image::_10X:
        return flat_field10X;
      case Image::_20X:
        return flat_field20X;
    }
  }

  bool StreamCam::spin_run() {

    std::cout << "spin_run started " << std::endl;


    Q_thread.start(qm);
    composite_thread.start(cm);

    Q_thread.join();
    composite_thread.join();

    std::cout << "spin_run done" << std::endl;

    return true;
  }

  bool StreamCam::run() {
    auto start = std::chrono::high_resolution_clock::now();


    disk_thread.start(dr);
    Q_thread.start(qm);
    composite_thread.start(cm);
    if (inferencing) {
      inference_thread.start(im);
    }

    disk_thread.join();
    Q_thread.join();
    composite_thread.join();
    if (inferencing) {
      inference_thread.join();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    return true;
  }

  void StreamCam::add_image(Image *image, unsigned long index) {
    image_mutex->writeLock();
    reg_results_mutex->writeLock();
    unsigned long size = images.size();
    if (index >= size) {
      images.resize(index + 100);

      reg_results.resize(index + 100);

      resize_mmatch_mutex->writeLock();
      matchM.resize(index + 100);
      resize_mmatch_mutex->unlock();

    }

    reg_results[index] = new RegInfo(this, true, Vec2(0.0, 0.0), true, 0);
    reg_results_mutex->unlock();

    images[index] = image;
    image_mutex->unlock();
  }

  void StreamCam::add_registration(pathCam::RegInfo *regInfo) {
    reg_results_mutex->writeLock();
    auto index = regInfo->index;
    reg_results[index] = regInfo;
    reg_results_mutex->unlock();
    auto image = get_image_ref(index);
    image->regInfo = regInfo;
  }

  bool StreamCam::get_registration(unsigned long image_idx, RegInfo *res) {
    reg_results_mutex->readLock();

    if (reg_results.size() <= image_idx || reg_results[image_idx] == nullptr) {
      reg_results_mutex->unlock();
      return false;
    }

    *res = *reg_results[image_idx];
    reg_results_mutex->unlock();
    return true;
  }


  std::vector<Image *> StreamCam::get_image_refs(std::vector<unsigned long int> indexes) {
    std::vector<Image *> temp;

    image_mutex->readLock();
    for (unsigned int i = 0; i < indexes.size(); i++) {
      if (images[indexes[i]] == nullptr) { continue; }
      temp.push_back(images[indexes[i]]);
    }
    image_mutex->unlock();

    return temp;
  }

  RegInfo *StreamCam::get_registration(unsigned long image_idx) {
    RegInfo *temp;
    reg_results_mutex->readLock();
    temp = reg_results[image_idx];
    reg_results_mutex->unlock();
    return temp;
  }

  Image *StreamCam::get_image_ref(unsigned long int index) {
    Image *temp;
    image_mutex->readLock();
    temp = images[index];
    image_mutex->unlock();
    return temp;
  }

  void StreamCam::add_new_component_Q(unsigned long image_index, cv::Size image_size) {

    auto component_index = increment_and_get_components();
    if (component_index != 0) {
      //start job to find scale and offset
      auto xcm = new XCompRunnable(this, image_index, component_index);
      JobQ->add_runnable(xcm);
    }
    newComponentQ.push({image_index, image_size, component_index});

  }


  void StreamCam::add_new_component(unsigned long image_index, cv::Size image_size, unsigned int component_index) {
    auto ri = reg_results[image_index];
    ri->component_membership = component_index;
    ri->index = image_index;
    ri->matchedTo = image_index; //this is a root image, it has no match

    //delete these pointers when destroyed
    auto *temp = new CompositeVoronoi(this, image_size, component_index);
    component_mutex->lock();
    composites.push_back(temp);
    if (composites.size() == 1) {
      ri->set_abc(Vec2(0, 0), component_index, true);
      set_scale_and_offset(0, 1, Point2f(0, 0));
      temp->imagePyramid->set_scale(1);
      temp->imagePyramid->set_offset(Point2f(0, 0));
      temp->update(std::vector<RegInfo *>{reg_results[image_index]});
    } else {
      temp->store_new_info(ri);
      temp->imagePyramid->set_scale(0);
      temp->imagePyramid->set_offset(Point2f(0, 0));
    }
    component_mutex->unlock();

  }

  std::vector<Image *> StreamCam::get_component_image_refs(unsigned long component) {
    auto dm = composites[component]->delaunayMembers;
    std::vector<unsigned long> res(dm.size());
    int i = 0;
    for (auto [key, value]: dm) {
      res[i] = dm[key];
      i++;
    }
    return get_image_refs(res);
  }

  std::vector<RegInfo *> StreamCam::get_Q_front() {
    compositeQ_mutex->lock();
    std::vector<RegInfo *> temp = compositeBatch.top();
    compositeBatch.pop();
    compositeQ_mutex->unlock();
    return temp;
  }

  std::vector<std::pair<Point2i, unsigned int>> StreamCam::get_tile_embed_Q_front() {
    std::vector<std::pair<Point2i,unsigned int>> temp;

    inferenceQMutex->lock();
    if (!tileEmbedQ.empty()) {
      unsigned int component_index = tileEmbedQ.front().second;


      while (!tileEmbedQ.empty() && tileEmbedQ.front().second == component_index && temp.size() < 512) {
        temp.push_back(tileEmbedQ.front());
        tileEmbedQ.pop();
      }
      inferenceQMutex->unlock();
      return temp;

    } else {
      inferenceQMutex->unlock();
      return {};
    }

  }

  void StreamCam::push_tile_embed_Q(std::pair<Point2i, unsigned int> _tileSet) {
    inferenceQMutex->lock();
    tileEmbedQ.push(_tileSet);
    inferenceQMutex->unlock();
    inferenceWait.set();
  }

  Image *StreamCam::get_Q_front_Spin() {
    resize_buffer_mutex->lock();
    Image *temp = spin_image_buffer.front();
    spin_image_buffer.pop();
    resize_buffer_mutex->unlock();
    return temp;
  }

  void StreamCam::pass_image(Image *image, unsigned long _image_index, bool _saveImg) {
    LoaderLogicRunnable *llr = new LoaderLogicRunnable(this, image, _image_index, true, _saveImg);
    loaderCount++;
    JobQ->add_runnable(llr);
  }

  void StreamCam::push_compositeQ(RegInfo *index) {
    compositeQ_mutex->lock();
    compositeBatch.push({index});
    compositeQ_mutex->unlock();
  }

  bool StreamCam::get_scale_and_offset(unsigned int component_index, double &_scale, cv::Point2f &_offset) {
    scaleRepoMutex->lock();
    if (scaleRepo.find(component_index) == scaleRepo.end()) {
      scaleRepoMutex->unlock();
      return false;
    }

    auto res = scaleRepo[component_index];
    _scale = res.first;
    _offset = res.second;
    scaleRepoMutex->unlock();
    return true;
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
    if (!successful) {
      jobComplete.wait();
    }
    someoneWaitingOnJobCompleteEvent = false;
  }

}
