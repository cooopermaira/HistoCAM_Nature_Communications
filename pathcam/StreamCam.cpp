//
//  StreamCam.cpp
//  pathCamLib
//
//  Created by cooper maira on 11/24/23.
//

#include "pathCam.h"


namespace pathCam {
  using Poco::Util::LayeredConfiguration;


  StreamCam::StreamCam(LayeredConfiguration::Ptr config) : BatchCam(config),
                                                           inferenceWait(Poco::Event::EVENT_AUTORESET),
                                                           compositeWait(Poco::Event::EVENT_AUTORESET),
                                                           cacheAlert(Poco::Event::EVENT_AUTORESET) {
    //inferencing = false;
    if (inferencing) {
      im = new InferenceManager(this);
    }

#ifdef PATHCAM_HAS_TENSORRT
    if (segmentWithSAM) {
      std::thread t([this]() {
        as = std::make_shared<AccessSAM>(this);
        as->load_model();
      });
      t.detach();
    }
#endif

    int threads = 2;

    JobQ = std::make_shared<JobQueue>(threads, threads, windowWidth);
    JobQ->parent = this;
    jqSecondary = std::make_shared<JobQueue>(2, 2, 0);

    MRTiledImageSet::frameHeight = image_height;
    MRTiledImageSet::frameWidth = image_width;
    MRTiledImageSet::scopeRadius = scope_radius * 0.7f;
    MRTiledImageSet::tileSize = tileSize;

    minPixelDistanceBetweenFrames = 500;
    minPixelDistanceBetweenFrames = pow(minPixelDistanceBetweenFrames, 2);


    // load_blur_engine();

#ifdef PATHCAM_OPENCV_CUDA
    compositorCudaDevice = GPU_select_cuda_device(1);
    //siftCudaDevice = GPU_select_cuda_device();
    siftCudaDevice = compositorCudaDevice;

    // cudaSetDevice(compositorCudaDevice);
#endif

    //lastFrame = Rect(0,0,image_width,image_height);
    circleMask = cv::Mat::zeros(image_height, image_width, CV_8U);
    cv::circle(circleMask, cv::Point(image_width / 2, image_height / 2), scope_radius, cv::Scalar(255),
               -1);

    //   cudaDeviceProp prop;
    //   cudaGetDeviceProperties(&prop, 0);
    //
    //   int uva, managed;
    //   cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, 0);
    //   cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, 0);
    //
    //   bool isUnifiedSystem = prop.integrated && uva && managed;
    //   int k = 0;
  }


  bool StreamCam::run() {
    auto start = std::chrono::high_resolution_clock::now();

    if (!MRImageSet) {
      MRImageSet = std::make_shared<MRTiledImageSet>();
    }
    MRImageSet->cwd = make_working_directory();
    MRImageSet->labelName = currentSlideLabel;


    auto dr1 = DiskReader(this);
    auto qm1 = QManager(this);
    auto cm1 = CompositeManager(this);

    compositing = true;

    disk_thread.start(dr1);
    Q_thread.start(qm1);
    composite_thread.start(cm1);

    //postprocessor_thread.start(ppm);
    if (inferencing) {
      inference_thread.start(*im);
    }

    disk_thread.join();
    Q_thread.join();
    composite_thread.join();
    //postprocessor_thread.join();
    if (inferencing) {
      //im->thread.join();
      inference_thread.join();
    }

    cleanup_and_reset();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "total runtime " << duration << std::endl;
    std::cout << std::endl << std::endl;


    return true;
  }


  bool StreamCam::spin_run() {
    std::cout << "spin_run started " << std::endl;

    if (!MRImageSet) {
      MRImageSet = std::make_shared<MRTiledImageSet>();
    }
    MRImageSet->cwd = make_working_directory();
    MRImageSet->labelName = currentSlideLabel;


    auto qm1 = QManager(this);
    auto cm1 = CompositeManager(this);

    recordingMode = true;
    compositing = true;

    Q_thread.start(qm1);
    composite_thread.start(cm1);
    //postprocessor_thread.start(ppm);
    if (inferencing) {
      inference_thread.start(*im);
    }

    Q_thread.join();
    composite_thread.join();
    //postprocessor_thread.join();
    if (inferencing) {
      //im->thread.join();
      inference_thread.join();
    }

    Poco::Path image_path = givenWorkingDirectory;
    image_path.append(MRImageSet->labelName);
    image_path.append("ts");
    image_path.setExtension("txt");

    write_image_timestamps(image_path.toString());

    cleanup_and_reset();

    std::cout << "spin_run done" << std::endl;

    return true;
  }

  std::string StreamCam::set_slide_label(std::string _name) {
    previousSlidesMutex.lock();
    currentSlideIndex = previousSlides.size();
    previousSlidesMutex.unlock();

    if (_name.empty()) {
      if (currentSlideLabel.empty()) {
        currentSlideLabel = std::to_string(currentSlideIndex);
      } else {
        currentSlideLabel = "";
      }
    } else {
      currentSlideLabel = _name;
    }
    return currentSlideLabel;
  }

  void StreamCam::write_image_timestamps(const std::string &filename) {
    image_mutex.readLock();

    std::ofstream out(filename);
    if (!out.is_open()) {
      std::cout << "timestamp write failed" << std::endl;
      return;
    }

    for (int i = 0; i <= maxIndex; ++i) {
      auto img = images[i];
      if (img) {
        out << img->index << " " << img->timeStamp << std::endl;
      }
    }
    out.close();
    image_mutex.unlock();
  }

  Poco::Path StreamCam::make_working_directory() const {
    assert(!currentSlideLabel.empty());

    // Build: <cwd>/<currentSlideLabel>/
    Poco::Path p = givenWorkingDirectory;
    p.makeDirectory();
    p.pushDirectory(currentSlideLabel);
    p.makeDirectory(); // ensures trailing slash; does NOT create on disk

    Poco::File dir(p);

    try {
      if (!dir.exists()) {
        // createDirectory() creates only the leaf; createDirectories() creates parents too.
        dir.createDirectories();
        std::cout << "created working directory " << p.toString() << std::endl;
      } else if (!dir.isDirectory()) {
        throw Poco::FileException("Path exists but is not a directory", p.toString());
      }
    } catch (const Poco::Exception &e) {
      std::cerr << "Failed to create working directory '" << p.toString()
          << "': " << e.displayText() << std::endl;
      throw; // or handle as you prefer
    }

    return p;
  }

  std::string StreamCam::get_slide_label(int slideIdx) {
    Poco::FastMutex::ScopedLock lock(previousSlidesMutex);
    if (previousSlides.size() <= slideIdx) { return ""; }
    return previousSlides[slideIdx]->labelName;
  }

  void StreamCam::update_last_frame(cv::Rect_<float> _rectInScale1Space, bool _showAsCircle, int _component_index,
                                    std::string _label, float _scale) {
    lastFrameMutex.lock();
    lastFrame = _rectInScale1Space;
    lastComponentIndex = _component_index;
    showAsCircle = _showAsCircle;
    lastLabel = _label;
    lastScale = _scale;
    lastFrameMutex.unlock();
  }

  void StreamCam::get_last_frame(cv::Rect_<float> &_rectInBaseSpace, bool &_showAsCircle, int &_lastComponentIndex,
                                 std::string &_magLabel, float &_lastScale) {
    lastFrameMutex.lock();
    _lastComponentIndex = lastComponentIndex;
    _showAsCircle = showAsCircle;
    _rectInBaseSpace = lastFrame;
    _lastScale = lastScale;
    _magLabel = lastLabel;
    lastFrameMutex.unlock();
  }


  std::pair<Image *, bool> StreamCam::get_most_recent_resolved_frame(Image *_fromImage, bool _acceptRoot) {
    Image *mostRecentResolved = nullptr;
    bool objChange = false;

    image_mutex.readLock();
    for (int i = _fromImage->index - 1; i >= 0; --i) {
      auto img = images[i];
      if (img && img->label == Image::_UNDEREXP) {
        objChange = true;
      }
      if (img && img->regInfo && img->regInfo->resolved) {
        if (img->regInfo->root && !_acceptRoot) { continue; }
        mostRecentResolved = img;
        break;
      }
    }
    image_mutex.unlock();

    return {mostRecentResolved, objChange};
  }

  std::vector<std::pair<Image *, Rect> > StreamCam::get_overlapping_frames(
    Rect _regionInComponentSpace, int _componentIndex) {
    std::vector<std::pair<Image *, Rect> > result;

    auto myTL = get_AbC_relative_from_relative(_componentIndex, _regionInComponentSpace.tl(), 0);
    Point p2 = _regionInComponentSpace.tl() + Point2i(image_width, image_height);
    auto myBR = get_AbC_relative_from_relative(_componentIndex, p2, 0);
    Rect myRect(myTL, myBR);

    image_mutex.readLock();
    for (auto &img: images) {
      if (!(img && img->regInfo && img->regInfo->component_membership >= 0)) { continue; }
      if (composites[img->regInfo->component_membership]->imagePyramid->scale == 0) { continue; };
      auto theirTL = get_AbC_relative_from_relative(img->regInfo->component_membership,
                                                    img->regInfo->absoluteCoords, 0);
      auto p = img->regInfo->absoluteCoords + Point2i(image_width, image_height);
      auto theirBR = get_AbC_relative_from_relative(img->regInfo->component_membership, p, 0);
      Rect theirRect(theirTL, theirBR);

      auto intersect = theirRect & myRect;
      if (!intersect.empty()) {
        result.push_back({img, intersect});
      }
    }
    image_mutex.unlock();

    return result;
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
    std::sort(devices.begin(), devices.end(), [](const auto &a, const auto &b) {
      return a.capability() > b.capability();
    });

    return devices[std::min(_priority, (int) devices.size() - 1)].index;
  }


  void StreamCam::load_delaunay_images_to_GPU(int _componentIndex) {
    //this is honestly unhinged to do this without at all checking if the space is available in memory or on the gpu
    //but for now were going with it TODO

    for (auto &img: composites[_componentIndex]->contributingImages) {
      if (!img->cudaBufferReady) {
        img->load_raw_from_disk();
        img->move_buffer_to_gpu(compositorCudaDevice, true);
      }
    }
  }

  // void StreamCam::push_SIFT_matches(std::vector<std::pair<Image *, Image *> > &_newOverlaps, Image *_image) {
  //   sfm->matchWorkOutstanding += (int) _newOverlaps.size();
  //   siftQMutex.lock();
  //   if (compositorCudaDevice != siftCudaDevice) {
  //     siftDataQueue.push(_image);
  //   }
  //
  //   if (_image->regInfo->root) {
  //     siftMatchQueue.push_front(_newOverlaps);
  //   } else {
  //     siftMatchQueue.push_back(_newOverlaps);
  //   }
  //   siftQMutex.unlock();
  // }

#endif

  // void StreamCam::push_pyramid_builder_Q(Point2i _index, unsigned _componentIndex) {
  //   pyramidQMutex.lock();
  //   pyramidBuilderQ.push({_index, _componentIndex});
  //   pyramidQMutex.unlock();
  // }

  void StreamCam::set_match(unsigned long _image_idx, unsigned long _prev_idx, Match *_m, bool _invert) {
    resize_mmatch_mutex.writeLock();
    if (_invert) {
      matchM.match[_image_idx][_prev_idx] = new Match(_m);
    } else {
      matchM.match[_prev_idx][_image_idx] = _m;
    }
    resize_mmatch_mutex.unlock();
  }

  void StreamCam::clear_buffer(int _image_idx) {
    auto img = get_image_ref(_image_idx);
    img->free_memory_RAW();
    img->free_memory_cuda();

  }

  Point2f StreamCam::get_AbC_relative_from_relative(unsigned int _srcCompIdx, Point2f _srcAbC,
                                                    unsigned int _dstCompIdx) {
    /*returns coordinates in dst component space given coordinates in src component space*/

    assert(composites.size() > max(_dstCompIdx,_srcCompIdx));

    auto srcImP = composites[_srcCompIdx]->imagePyramid;
    assert(srcImP);
    assert(srcImP->scale != 0);

    auto dstImP = composites[_dstCompIdx]->imagePyramid;
    assert(dstImP);
    assert(dstImP->scale != 0);

    //convert to base space
    auto pointInBaseSpace = srcImP->scale * (_srcAbC + srcImP->offset);

    //convert to dst space
    return pointInBaseSpace / dstImP->scale - dstImP->offset;
  }


  bool StreamCam::has_flatfield(int label) {
    if (label == Image::_2X || label == Image::_4X || label == Image::_10X || Image::_20X) {
      return true;
    }
    return false;
  }

  void StreamCam::mark_neighbors_as_underexposed(unsigned long _index) {
    std::vector<long> neighborhood;
    for (long i = max(0ul, _index - windowWidth); i <= _index + windowWidth; i++) {
      neighborhood.push_back(i);
      JobQ->cancel_job(2, i);
    }

    auto answer = get_image_ref(neighborhood);
    for (auto img: answer) {
      img->mark_too_dark();
    }

    JobQ->update_job_readiness(2, _index);
  }

  void StreamCam::set_flatfield(int label, const Mat &ff) {
    switch (label) {
      case Image::_2X:
        flat_field2X = ff;
        return;
      case Image::_4X:
        flat_field4X = ff;
        return;
      case Image::_10X:
        flat_field10X = ff;
        return;
      case Image::_20X:
        flat_field20X = ff;
        return;
      default:
        throw std::runtime_error("not recognized");
    }
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
      default:
        throw std::runtime_error("not recognized");
    }
  }

  std::string StreamCam::get_flatfield_path(int label, bool &ffAlreadySet) {
    if (recordingMode) {
      switch (label) {
        case Image::_2X:
          if (flat_field2X.empty()) { ffAlreadySet = false; }
          return flat_field_file_2x_r.toString();
        case Image::_4X:
          if (flat_field4X.empty()) { ffAlreadySet = false; }
          return flat_field_file_4x_r.toString();
        case Image::_10X:
          if (flat_field10X.empty()) { ffAlreadySet = false; }
          return flat_field_file_10x_r.toString();
        case Image::_20X:
          if (flat_field20X.empty()) { ffAlreadySet = false; }
          return flat_field_file_20x_r.toString();
      }
    } else {
      switch (label) {
        case Image::_2X:
          if (flat_field2X.empty()) { ffAlreadySet = false; }
          return flat_field_file_2x.toString();
        case Image::_4X:
          if (flat_field4X.empty()) { ffAlreadySet = false; }
          return flat_field_file_4x.toString();
        case Image::_10X:
          if (flat_field10X.empty()) { ffAlreadySet = false; }
          return flat_field_file_10x.toString();
        case Image::_20X:
          if (flat_field20X.empty()) { ffAlreadySet = false; }
          return flat_field_file_20x.toString();
      }
    }
  }


  void StreamCam::add_image(Image *image, long index) {
    if (index >= images.size()) {
      reg_results_mutex.writeLock();
      reg_results.resize(index + 100, nullptr);
      reg_results_mutex.unlock();

      // reg_results[index] = new RegInfo(this, true, {0.0, 0.0}, true, 0);

      image_mutex.writeLock();
      images.resize(index + 100);
      images[index] = image;
      ++maxIndex;
      image_mutex.unlock();
    } else {
      // reg_results_mutex.readLock();
      // reg_results[index] = new RegInfo(this, true, {0.0, 0.0}, true, 0);
      // reg_results_mutex.unlock();

      image_mutex.writeLock();
      images[index] = image;
      ++maxIndex;
      image_mutex.unlock();
    }
    //
    // image_mutex.writeLock();
    // reg_results_mutex.writeLock();
    // long size = images.size();
    // if (index >= size) {
    //   images.resize(index + 100);
    //
    //   reg_results.resize(index + 100);
    //
    //   // resize_mmatch_mutex.writeLock();
    //   // matchM.resize(index + 100);
    //   // resize_mmatch_mutex.unlock();
    // }
    //
    // reg_results[index] = new RegInfo(this, true, {0.0, 0.0}, true, 0);
    // reg_results_mutex.unlock();
    //
    // images[index] = image;
    // ++maxIndex;
    // image_mutex.unlock();
  }

  void StreamCam::add_registration(pathCam::RegInfo *regInfo) {
    reg_results_mutex.writeLock();
    auto index = regInfo->index;
    reg_results[index] = regInfo;
    reg_results_mutex.unlock();
    auto image = get_image_ref(index);
    image->regInfo = regInfo;
  }


  std::vector<Image *> StreamCam::get_image_ref(const std::vector<long int> &_indexes) {
    /*because images vector can be expanded, this gives access to the pointers within that vector under mutex lock.
    an empty vector of unsigned longs returns entire list of images*/
    std::vector<Image *> temp;

    image_mutex.readLock();
    if (_indexes.empty()) {
      long i = 0;
      while (images[i] && i <= maxIndex) {
        temp.push_back(images[i++]);
      }
    } else {
      for (int i = 0; i < _indexes.size(); i++) {
        if (images[_indexes[i]] && _indexes[i] <= maxIndex) {
          temp.push_back(images[_indexes[i]]);
        }
      }
    }
    image_mutex.unlock();

    return temp;
  }

  std::vector<RegInfo *> StreamCam::get_reg_ref(const std::vector<unsigned long> &_indexes) {
    std::vector<RegInfo *> temp;

    reg_results_mutex.writeLock();
    for (unsigned int i = 0; i < _indexes.size(); i++) {
      if (reg_results[_indexes[i]]) {
        temp.push_back(reg_results[_indexes[i]]);
      }
    }
    reg_results_mutex.unlock();

    return temp;
  };

  RegInfo *StreamCam::get_reg_ref(long image_idx) {
    RegInfo *temp;
    reg_results_mutex.readLock();
    temp = reg_results[image_idx];
    reg_results_mutex.unlock();
    return temp;
  }

  void StreamCam::set_reg_ref(RegInfo *ri) {
    reg_results_mutex.writeLock();
    reg_results[ri->index] = ri;
    reg_results_mutex.unlock();
  }


  Image *StreamCam::get_image_ref(unsigned long index) {
    Image *temp = nullptr;

    image_mutex.readLock();
    if (index < images.size()) {
      temp = images[index];
    }
    image_mutex.unlock();

    return temp;
  }

  int StreamCam::add_new_component_Q(unsigned long image_index, Size image_size) {
    Poco::Mutex::ScopedLock lock(componentQmutex);
    auto component_index = increment_and_get_components();
    newComponentQ.push({image_index, image_size, component_index});
    return component_index;
  }


  void StreamCam::add_new_component(unsigned long image_index, Size image_size, unsigned int component_index) {
    auto ri = reg_results[image_index];

    {
      Poco::Mutex::ScopedLock lock1(ri->rAccessMutex);
      Poco::RWLock::ScopedWriteLock lock2(component_mutex);

      ri->component_membership = component_index;
      ri->stayFixedDuringBundleAdjustment = true;
      ri->root = true;
      ri->matchedTo = ri->index;

      std::shared_ptr<Composite> component;
      if (CompositeType == _MetricComposite) {
        component = std::make_shared<MetricComposite>(this, image_size, component_index);
      } else if (CompositeType == _CompositeVoronoi) {
#ifdef PATHCAM_OPENCV_CUDA
        component = std::make_shared<CompositeVoronoi>(this, image_size, component_index);
#else
        component = std::make_shared<MetricComposite>(this, image_size, component_index);
#endif
      }
      component->joinedTo = component;

      composites.push_back(component);
      component->root = ri->image;

      if (composites.size() == 1) {
        ri->rootOfRoot = true;
        component->set_scale(1);
      } else {
        component->set_scale(0);
      }
      component->set_offset(Point2f(0, 0));
    }

    {
      std::lock_guard lock(ri->image->blurMutex);
      ri->image->motionBlur = 10000000;
      ri->image->blurSet = true;
    }

    ri->resolved = true;
    push_compositeQ(ri);
    ri->cast_requested_votes();
  }

  std::shared_ptr<Composite> StreamCam::joined_to_root(const std::shared_ptr<Composite> &query) const {
    if (!query) {
      throw std::runtime_error("null composite");
    }

    std::shared_ptr<Composite> ans = query;
    std::unordered_set<unsigned> visitedSet;

    while (ans->joinedTo != ans) {
      if (!visitedSet.insert(ans->componentIndex).second) {
        throw std::runtime_error("cycle detected");
      }
      ans = ans->joinedTo;
    }
    return ans;
  }

  //demo
  void StreamCam::run_agg_classify() {
    throw std::runtime_error("this function hasn't been refactored since removing torch");
    if (inferencing && classifying) {
      //im->run_agg_classify();
      std::this_thread::sleep_for(std::chrono::seconds(2));
      compositeWait.set();
    }
  }

  void StreamCam::update_observers() {
    for (unsigned int i = 0; i < observers.size(); i++) {
      MRImageSet->update_bounds();
      observers[i]->notify_new_data();
      observers[i]->update();
    }
  }

  void StreamCam::notify_observers() {
    if (MRImageSet) {
      MRImageSet->update_bounds();
    }
    for (int i = 0; i < observers.size(); i++) {
      observers[i]->notify_new_data();
    }
  }

  bool StreamCam::segment_with_SAM(std::vector<Point3f> &_clicks, int _segID, int _slideIdx) {
#ifdef PATHCAM_HAS_TENSORRT
    if (segmentWithSAM) {
      if (as) {
        if ((_clicks.end() - 2)->z == 4) {
          if (_clicks.back().z == 5) {
            std::vector<Point3f> inputClicks(_clicks.begin(), _clicks.end() - 2);
            Point2i fovUL((_clicks.end() - 2)->x, (_clicks.end() - 2)->y);
            Point2i fovLR(_clicks.back().x, _clicks.back().y);
            as->create_segmentation_coarse_to_fine(inputClicks, _segID, {fovUL, fovLR}, _slideIdx);
            return true;
          } else {
            throw std::runtime_error("one FOV point but not the other");
          }
        } else {
          as->create_segmentation(_clicks, _segID);
          return true;
        }
      }
    }
#endif
    return false;
  }


  std::vector<Image *> StreamCam::get_component_image_refs(unsigned long component) {
    auto dm = composites[component]->delaunayMembers;
    std::vector<long> res(dm.size());
    int i = 0;
    for (auto [key, value]: dm) {
      res[i] = dm[key];
      i++;
    }
    return get_image_ref(res);
  }

  std::vector<RegInfo *> StreamCam::get_Q_front(bool _pop) {
    compositeQ_mutex.lock();
    std::vector<RegInfo *> temp = compositeBatch.top();
    if (_pop) {
      compositeBatch.pop();
    }
    compositeQ_mutex.unlock();
    return temp;
  }

  std::vector<std::tuple<int, int, unsigned int> > StreamCam::get_tile_embed_Q_front() {
    std::vector<std::tuple<int, int, unsigned int> > temp;

    inferenceQMutex.lock();
    if (!tileEmbedQ.empty()) {
      unsigned int component_index = std::get<2>(tileEmbedQ.front());

      while (!tileEmbedQ.empty() && std::get<2>(tileEmbedQ.front()) == component_index && temp.size() <
             maxTilesPerBatch) {
        temp.push_back(tileEmbedQ.front());
        tileEmbedQ.pop();
      }
      inferenceQMutex.unlock();
      return temp;
    } else {
      inferenceQMutex.unlock();
      return {};
    }
  }


  // std::vector<std::pair<Image *, Image *> > StreamCam::get_sift_match_Q_front(std::vector<Image *> &_images) {
  //   std::vector<std::pair<Image *, Image *> > temp;
  //
  //   siftQMutex.lock();
  //
  //   //move all the buffers to my device (if necessary)
  //   get_sift_data_Q_front(_images);
  //
  //   //grab a bunch of matches from the Q to process
  //   if (!siftMatchQueue.empty()) {
  //     temp = siftMatchQueue.front();
  //     siftMatchQueue.pop_front();
  //   }
  //   siftQMutex.unlock();
  //   return temp;
  // }


  // void StreamCam::get_sift_data_Q_front(std::vector<Image *> &_images) {
  //   std::vector<Image *> temp;
  //
  //   //no need to check if compositor device is different from sft device, Q will be empty if same -> no need to move data
  //   while (!siftDataQueue.empty()) {
  //     auto img = siftDataQueue.front();
  //     _images.push_back(img);
  //     if (siftCudaDevice != compositorCudaDevice) {
  //       CHECK_CUDA(cudaMalloc((void **) &img->siftData.d_data, sizeof(SiftPoint) * img->siftData.numPts));
  //       CHECK_CUDA(
  //         cudaMemcpy(img->siftData.d_data, img->siftData.h_data, sizeof(SiftPoint) * img->siftData.numPts,
  //           cudaMemcpyHostToDevice));
  //     }
  //     siftDataQueue.pop();
  //   }
  // }


  void StreamCam::push_tile_embed_Q(std::vector<Point2i> &_tiles, unsigned int _componentIndex) {
    if (inferencing) {
      inferenceQMutex.lock();
      for (auto tilePoint: _tiles) {
        auto tpl = std::tuple<int, int, unsigned>(tilePoint.x, tilePoint.y, _componentIndex);
        tileEmbedQ.push(tpl);
      }
      inferenceQMutex.unlock();
      inferenceWait.set();
    }
  }

  Image *StreamCam::get_Q_front_Spin() {
    resize_buffer_mutex.lock();
    Image *temp = spin_image_buffer.front();
    spin_image_buffer.pop();
    resize_buffer_mutex.unlock();
    return temp;
  }

  void StreamCam::pass_image(Image *image, unsigned long _image_index, bool _saveImg) {
    add_image(image, _image_index);
    LoaderLogicRunnable *llr = new LoaderLogicRunnable(this, image, _image_index, true, _saveImg);
    loaderCount++;
    JobQ->add_runnable(llr);
  }

  bool StreamCam::sufficient_distance(Vec2 _coordsInQuestion, int _componentIdx) {
    bool answer = false;

    pixelDistanceMutex.lock();

    if (lastAcceptedCoords.size() < _componentIdx + 1) {
      lastAcceptedCoords.resize(_componentIdx + 1);
      lastAcceptedCoords[_componentIdx] = _coordsInQuestion;
      answer = true;
    } else {
      Vec2 v = lastAcceptedCoords[_componentIdx];
      if (pow(v.x - _coordsInQuestion.x, 2) + pow(v.y - _coordsInQuestion.y, 2) >=
          minPixelDistanceBetweenFrames) {
        lastAcceptedCoords[_componentIdx] = _coordsInQuestion;
        answer = true;
      }
    }

    pixelDistanceMutex.unlock();
    return answer;
  }

  void StreamCam::push_compositeQ(RegInfo *_regInfo) {
    /*_regInfo's component membership may have changed after being added to this Q. If it's already gone
     * through the Q and been added to a suspended composite, it needs to be re-added to this Q with now
     * corrected membership. This is the same behavior as adding it the first time. If it's still in the
     * Q, the membership has already been corrected by this point so just do nothing, the problem is solved
     * before it was noticed.
     */
    _regInfo->rAccessMutex.lock();
    if (_regInfo->inCompositeQ) {
      _regInfo->rAccessMutex.unlock();
      return;
    }
    _regInfo->inCompositeQ = true;
    _regInfo->rAccessMutex.unlock();

    compositeQ_mutex.lock();
    compositeBatch.push({_regInfo});
    _regInfo->queued = true;
    compositeQ_mutex.unlock();
  }

  bool StreamCam::get_scale_and_offset(unsigned int component_index, double &_scale, cv::Point2f &_offset) {
    scaleRepoMutex.lock();
    if (scaleRepo.find(component_index) == scaleRepo.end()) {
      scaleRepoMutex.unlock();
      return false;
    }

    auto res = scaleRepo[component_index];
    _scale = res.first;
    _offset = res.second;
    scaleRepoMutex.unlock();
    return true;
  }

  bool StreamCam::compositeQ_empty() {
    compositeQ_mutex.lock();
    bool isEmpty = compositeBatch.size() == 0;
    compositeQ_mutex.unlock();
    return isEmpty;
  }

  std::shared_ptr<MRTiledImageSet> StreamCam::get_MRimage_reference() {
    if (!MRImageSet) {
      MRImageSet = std::make_shared<MRTiledImageSet>();
      {
        Poco::FastMutex::ScopedLock lock(previousSlidesMutex);
        MRImageSet->index = previousSlides.size();
      }
    }
    return MRImageSet;
  }

  void RunnableIntermediate::waitOnThisGuy() {
    someoneWaitingOnJobCompleteEvent = true;
    if (!successful) {
      jobComplete.wait();
    }
    someoneWaitingOnJobCompleteEvent = false;
  }

  bool StreamCam::create_mag_label_to_scale_lookup(std::unordered_map<int, float> &_lookup) {
    std::vector<int> labels = {Image::_2X, Image::_4X, Image::_10X, Image::_20X, Image::_40X};
    for (auto comp: composites) {
      if (comp->get_scale() == 1) {
        assert(!comp->candidateScaleRatios.empty());
        for (int i = 0; i < labels.size(); ++i) {
          _lookup[labels[i]] = 1 / comp->candidateScaleRatios[i];
        }
        return true;
      }
    }
    return false;
  }

  void StreamCam::save_velocity_data() {
    auto p = images[0]->image_file.parent().parent();
    p.setFileName("velocities.txt");
    auto pstr = p.toString();
    std::ofstream out(pstr);
    if (!out.is_open()) {
      std::cout << "timestamp write failed" << std::endl;
      return;
    }

    for (int i = 0; i <= maxIndex; ++i) {
      auto ri = reg_results[i];
      if (!ri) { continue; }
      if (ri->matchedTo >= 0 && ri->matchedBy >= 0 && ri->image) {
        auto other = reg_results[ri->matchedBy];
        auto travel = ri->relativeCoords + other->relativeCoords;
        int val = travel.x * travel.x + travel.y * travel.y;

        out << ri->image->image_file.toString() << " " << val << std::endl;
      }
    }

    out.close();
  }

  void StreamCam::cleanup_and_reset() {

    for (auto comp: composites) {
      comp->correct_offset();
    }

    // save_velocity_data();


    std::vector<Point2i> AbCs(maxIndex + 1);
    std::vector<int> frameComponentMembership(maxIndex + 1);
    std::vector<long> frameTimeStamps(maxIndex + 1);

    for (auto &img: images) {
      if (img && img->index <= maxIndex) {
        if (img->regInfo && img->is_good()) {
          if (!img->regInfo->wasAligned) {
            // AbC wasn't aligned in bundle adjustment, recalculate based on relative coords
            auto abc = Point2f(
              -img->regInfo->relativeCoords + images[img->regInfo->matchedTo]->regInfo->absoluteCoords);
            // abc = get_AbC_relative_from_relative(img->regInfo->component_membership, abc, 0);
            AbCs[img->index] = abc;
          } else {
            // Abc was aligned during BA, trust its coords
            // AbCs[img->index] = get_AbC_relative_from_relative(img->regInfo->component_membership,
            //                                                   Point2f(img->regInfo->absoluteCoords), 0);
            AbCs[img->index] = img->regInfo->absoluteCoords;
          }
          frameComponentMembership[img->index] = img->regInfo->component_membership;
        } else {
          if (img->label == Image::_UNDEREXP || img->label == Image::_LOWFEAT) {
            if (img->index > 0) {
              AbCs[img->index] = AbCs[img->index - 1];
              frameComponentMembership[img->index] = frameComponentMembership[img->index - 1];
            } //else it just stays (0,0) because that's what it inits to.
          } else {
            std::cout << img->index << " " << Image::get_label(img->label) << "no reginfo but non black label" <<
                std::endl;
          }
        }
        frameTimeStamps[img->index] = img->timeStamp;
      }
    }

    std::unordered_map<int, float> labelScaleLookup;
    create_mag_label_to_scale_lookup(labelScaleLookup);

    for (auto &comp: composites) {
      if (comp->frameCount < 5) {
        comp->suspended = true;
        comp->imagePyramid->suspended = true;
      }
    }

    MRImageSet->labelScaleLookup = labelScaleLookup;
    MRImageSet->AbCs = AbCs;
    MRImageSet->frameComponentMembership = frameComponentMembership;
    MRImageSet->frameTimeStamps = frameTimeStamps;
    MRImageSet->framesPerMillisecond = float(maxIndex) / float(captureTimeMS);
    MRImageSet->captureTimeMS = captureTimeMS;


    //clean up all jobs (jobq 1 and 2)
    JobQ->pool->joinAll();
    for (int i = 0; i < JobQ->jobRefs.size(); ++i) {
      if (JobQ->jobRefs[i]) {
        delete JobQ->jobRefs[i];
      }
    }
    JobQ->jobRefs.clear();
    JobQ->cancelJob.clear();
    JobQ->jobsReadiness.clear();
    for (int i = 0; i < JobQ->jobRefsZeroFlag.size(); ++i) {
      if (JobQ->jobRefsZeroFlag[i]) {
        delete JobQ->jobRefsZeroFlag[i];
      }
    }
    JobQ->jobRefsZeroFlag.clear();

    jqSecondary->pool->joinAll();
    for (int i = 0; i < jqSecondary->jobRefs.size(); ++i) {
      if (jqSecondary->jobRefs[i]) {
        delete jqSecondary->jobRefs[i];
      }
    }
    jqSecondary->jobRefs.clear();
    jqSecondary->cancelJob.clear();
    jqSecondary->jobsReadiness.clear();
    for (int i = 0; i < jqSecondary->jobRefsZeroFlag.size(); ++i) {
      if (jqSecondary->jobRefsZeroFlag[i]) {
        delete jqSecondary->jobRefsZeroFlag[i];
      }
    }
    jqSecondary->jobRefsZeroFlag.clear();

    //clean up all reginfo
    for (int i = 0; i < reg_results.size(); ++i) {
      if (reg_results[i]) {
        delete reg_results[i];
      }
    }
    reg_results.clear();

    //clean up all images
    for (int i = 0; i < images.size(); ++i) {
      if (images[i]) {
        delete images[i];
      }
    }
    images.clear();

    //clean up all composites
    for (auto &c: composites) {
      c->joinedTo = nullptr;
      c->absorbedComponents.clear();
    }
    composites.clear();
    components = 0;
    maxIndex = -1;

    MRImageSet->generate_nav_paths();

    //store slide and reset slide member variable
    MRImageSet->detach();
    {
      Poco::FastMutex::ScopedLock lock(previousSlidesMutex);
      previousSlides.push_back(std::move(MRImageSet));
      ++numSlides;
    }
    cacheAlert.set();

    assert(set_slide_label().empty());
    pathcamReady = true;

    // auto start = std::chrono::high_resolution_clock::now();
    // previousSlides.back()->correct_alignment();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    //   std::chrono::high_resolution_clock::now() - start).count();
    // int k = 0;
  }

  StreamCam::~StreamCam() {
    // clean_up_blur_engine();

    // Image::cleanup_blur_check_statics();

    previousSlides.clear();
  }
}
