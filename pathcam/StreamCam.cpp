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
                                                           pixelDistanceMutex(new Poco::FastMutex()),
                                                           cm(new CompositeManager(this)),
                                                           qm(new QManager(this)),
                                                           dr(new DiskReader(this)),
                                                           inferenceWait(true),
                                                           compositeWait(true),
                                                           microscopeInput(true) {
    inferencing = false;
    if (inferencing) {
      inferenceQMutex = new Poco::FastMutex();
      im = new InferenceManager(this);
    }

    int threads = 10;

    minPixelDistanceBetweenFrames = 400;
    minPixelDistanceBetweenFrames = pow(minPixelDistanceBetweenFrames,2);

#ifdef HAVE_OPENCV_CUDAARITHM
      compositorCudaDevice = GPU_select_cuda_device(1);
#endif

    MRimage.reset(new MRTiledImageSet());
    JobQ = new JobQueue(threads, threads,windowWidth);
    JobQ->parent = this;


    //lastFrame = Rect(0,0,image_width,image_height);
    circleMask = cv::Mat::zeros(image_height, image_width, CV_8U);
    cv::circle(circleMask, cv::Point(image_width / 2, image_height / 2), scope_radius, cv::Scalar(255),
               -1);
    regCircleMask = cv::Mat::zeros(image_height * scale_factor, image_width * scale_factor, CV_8U);
    cv::circle(regCircleMask, cv::Point(float(image_width / 2) * scale_factor, float(image_height / 2) * scale_factor),
               scope_radius, cv::Scalar(255),
               -1);
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
      //im->thread.join();
      inference_thread.join();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    return true;
  }


  bool StreamCam::spin_run() {

    std::cout << "spin_run started " << std::endl;
    recordingMode = true;

    Q_thread.start(qm);
    composite_thread.start(cm);
    if (inferencing) {
      inference_thread.start(im);
    }

    Q_thread.join();
    composite_thread.join();
    if (inferencing) {
      //im->thread.join();
      inference_thread.join();
    }

    std::cout << "spin_run done" << std::endl;

    return true;
  }

  void StreamCam::update_last_frame(cv::Rect_<float> _rectInScale1Space, bool _showAsCircle, int _component_index,
                                    std::string _label) {
    lastFrameMutex->lock();
    lastFrame = _rectInScale1Space;
    lastComponentIndex = _component_index;
    showAsCircle = _showAsCircle;
    lastLabel = _label;
    lastFrameMutex->unlock();
  }

  void StreamCam::get_last_frame(cv::Rect_<float> &_rectInScale1Space, bool &_showAsCircle, int &_lastComponentIndex,
                                 std::string &_magLabel) {
    lastFrameMutex->lock();
    _lastComponentIndex = lastComponentIndex;
    _showAsCircle = showAsCircle;
    _rectInScale1Space = lastFrame;
    _magLabel = lastLabel;
    lastFrameMutex->unlock();
  }

#ifdef HAVE_OPENCV_CUDAARITHM

  int StreamCam::GPU_select_cuda_device(int _priority) {
    int device_count = cuda::getCudaEnabledDeviceCount();
    if (device_count == 0) {
      throw std::runtime_error("No CUDA devices found");
    }

    struct DeviceInfoStruct {
      int index;
      int major;
      int minor;
      float capability() const { return major + minor / 10.0f; }
    };

    std::vector<DeviceInfoStruct> devices;
    for (int i = 0; i < device_count; ++i) {
      cuda::DeviceInfo dev_info(i);
      devices.push_back({i, dev_info.majorVersion(), dev_info.minorVersion()});
    }

    // Sort devices by compute capability descending
    std::sort(devices.begin(), devices.end(), [](const auto& a, const auto& b) {
        return a.capability() > b.capability();
    });

    return devices[std::min(_priority,(int) devices.size() - 1)].index;

  }
#endif

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

  void StreamCam::mark_neighbors_as_underexposed(unsigned long _index) {
    std::vector<unsigned long> neighborhood;
    for (unsigned long i = max(0ul,_index - windowWidth); i < _index + windowWidth; i++) {
      neighborhood.push_back(i);
      JobQ->cancel_job(2,_index);
    }

    auto answer = get_image_refs(neighborhood);

    for (auto img : answer) {
      img->mark_too_dark();
    }
  }

  std::string StreamCam::get_flatfield(int label) {
    if (recordingMode) {
      switch (label) {
        case Image::_2X:
          return flat_field_file_2x_r.toString();
        case Image::_4X:
          return flat_field_file_4x_r.toString();
        case Image::_10X:
          return flat_field_file_10x_r.toString();
        case Image::_20X:
          return flat_field_file_20x_r.toString();
      }
    }else {
      switch (label) {
        case Image::_2X:
          return flat_field_file_2x.toString();
        case Image::_4X:
          return flat_field_file_4x.toString();
        case Image::_10X:
          return flat_field_file_10x.toString();
        case Image::_20X:
          return flat_field_file_20x.toString();
      }
    }
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
    ++maxIndex;
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


  std::vector<Image *> StreamCam::get_image_refs(std::vector<unsigned long int> indexes) {
    std::vector<Image *> temp;

    image_mutex->readLock();
    for (unsigned int i = 0; i < indexes.size(); i++) {
      if (images[indexes[i]]) {
        temp.push_back(images[indexes[i]]);
      }
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

    auto *temp = new CompositeVoronoi(this, image_size, component_index);
    component_mutex->lock();
    composites.push_back(temp);

    if (composites.size() == 1) {//first component added

      //set component mag level
      if(initialLabel == 0){
        temp->componentMagLabel = get_image_ref(image_index)->label;
      }else {
        temp->componentMagLabel = initialLabel;
        temp->get_flatfield();
      }

      set_scale_and_offset(0, 1, Point2f(0, 0));

      //set component scale and offset to be 1 and origin
      temp->imagePyramid->set_scale(1);
      temp->imagePyramid->set_offset(Point2f(0, 0));
      ri->set_abc(Vec2(0, 0), component_index, true);
    } else {

      temp->store_new_info(ri);
      temp->imagePyramid->set_scale(0);
      temp->imagePyramid->set_offset(Point2f(0, 0));

    }
    component_mutex->unlock();

  }

  //demo
  void StreamCam::run_agg_classify() {
    if(inferencing && classifying){
      im->run_agg_classify();
      std::this_thread::sleep_for(std::chrono::seconds(2));
      compositeWait.set();
    }
  }

  void StreamCam::update_observers()  {
    for (unsigned int i = 0; i < observers.size(); i++) {
      MRimage->update_bounds();
      observers[i]->notify_new_data();
      observers[i]->update();
    }
  }

  void StreamCam::notify_observers() {
    for (unsigned int i = 0; i < observers.size(); i++) {
      MRimage->update_bounds();
      observers[i]->notify_new_data();
    }
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

  std::vector<std::tuple<int, int, unsigned int>> StreamCam::get_tile_embed_Q_front() {
    std::vector<std::tuple<int, int, unsigned int>> temp;

    inferenceQMutex->lock();
    if (!tileEmbedQ.empty()) {
      unsigned int component_index = std::get<2>(tileEmbedQ.front());

      while (!tileEmbedQ.empty() && std::get<2>(tileEmbedQ.front()) == component_index && temp.size() < maxTilesPerBatch) {
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

  void StreamCam::push_tile_embed_Q(std::vector<Point2i> &_tiles, unsigned int _componentIndex) {
    if(inferencing) {
      inferenceQMutex->lock();
      for (auto tilePoint: _tiles) {
        auto tpl = std::tuple<int, int, unsigned>(tilePoint.x, tilePoint.y, _componentIndex);
        tileEmbedQ.push(tpl);
      }
      inferenceQMutex->unlock();
      inferenceWait.set();
    }
  }

  Image *StreamCam::get_Q_front_Spin() {
    resize_buffer_mutex->lock();
    Image *temp = spin_image_buffer.front();
    spin_image_buffer.pop();
    resize_buffer_mutex->unlock();
    return temp;
  }

  void StreamCam::pass_image(Image *image, unsigned long _image_index, bool _saveImg) {
    add_image(image,_image_index);
    LoaderLogicRunnable *llr = new LoaderLogicRunnable(this, image, _image_index, true, _saveImg);
    loaderCount++;
    JobQ->add_runnable(llr);
  }

  bool StreamCam::sufficient_distance(Vec2 _coordsInQuestion, int _componentIdx) {
    bool answer = false;

    pixelDistanceMutex->lock();

    if(lastAcceptedCoords.size() < _componentIdx + 1){
      lastAcceptedCoords.resize(_componentIdx + 1);
      lastAcceptedCoords[_componentIdx] = _coordsInQuestion;
      answer = true;
    }else{
      Vec2 v = lastAcceptedCoords[_componentIdx];
      if(pow(v.x - _coordsInQuestion.x,2) + pow(v.y - _coordsInQuestion.y,2) >= minPixelDistanceBetweenFrames){
        lastAcceptedCoords[_componentIdx] = _coordsInQuestion;
        answer = true;
      }
    }

    pixelDistanceMutex->unlock();
    return answer;
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
