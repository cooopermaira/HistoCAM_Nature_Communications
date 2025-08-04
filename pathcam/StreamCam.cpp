//
//  StreamCam.cpp
//  pathCamLib
//
//  Created by cooper maira on 11/24/23.
//

#include "pathCam.h"
#include "utils.h"

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
                                                           siftQMutex(new Poco::FastMutex()),
                                                           cm(new CompositeManager(this)),
                                                           qm(new QManager(this)),
                                                           dr(new DiskReader(this)),
                                                           ppm(new PostProcessManager(this)),
                                                           sfm(new SiftFeatureMatcher(this)),
                                                           inferenceWait(true),
                                                           compositeWait(true),
                                                           microscopeInput(true) {
    //inferencing = false;

    if (inferencing) {
      inferenceQMutex = new Poco::FastMutex();
      im = new InferenceManager(this);
    }

    if (segmentWithSAM) {
      as = new AccessSAM(this);
    }

    int threads = 10;

    minPixelDistanceBetweenFrames = 500;
    minPixelDistanceBetweenFrames = pow(minPixelDistanceBetweenFrames, 2);

#ifdef HAVE_OPENCV_CUDAARITHM
    compositorCudaDevice = GPU_select_cuda_device(1);
    siftCudaDevice = GPU_select_cuda_device();

    // cudaSetDevice(compositorCudaDevice);
    // SpeedSam ss2("/home/max/pathcam/pathcam/SPEED-SAM-C-TENSORRT/model/SAM_encoder.engine","/home/max/pathcam/pathcam/SPEED-SAM-C-TENSORRT/model/SAM_mask_decoder.engine");
    //segmentWithPoint(ss2,"/home/max/pathcam/pathcam/SPEED-SAM-C-TENSORRT/assets/dogs.jpg","/home/max/pathcam/pathcam/SPEED-SAM-C-TENSORRT/assets/dogs_mask2.jpg");
#endif

    MRimage.reset(new MRTiledImageSet());
    JobQ = new JobQueue(threads, threads, windowWidth);
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
    postprocessor_thread.start(ppm);
    if (inferencing) {
      inference_thread.start(*im);
    }

    disk_thread.join();
    Q_thread.join();
    composite_thread.join();
    postprocessor_thread.join();
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
    postprocessor_thread.start(ppm);
    if (inferencing) {
      inference_thread.start(*im);
    }

    Q_thread.join();
    composite_thread.join();
    postprocessor_thread.join();
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
    std::sort(devices.begin(), devices.end(), [](const auto &a, const auto &b) {
      return a.capability() > b.capability();
    });

    return devices[std::min(_priority, (int) devices.size() - 1)].index;
  }

  void StreamCam::align_and_rebuild() {
    if (composites.empty()) { return; }

    std::thread([this]() { this->load_delaunay_images_to_GPU(0); }).detach();

    while (true) {
      if (sfm->tracksReady()) {
        //we're ready
        auto tracks = sfm->ftg->generateCurrentTracks(sfm->imagesProcessed);



        sfm->bai->optimizer->clear();
        sfm->bai->run_bundle_adjustment(tracks, sfm->imagesProcessed);


        double maxX = 0;
        double maxY = 0;
        for (auto &comp : composites) {
          if (comp->needsAlignment) {
            for (auto &img : comp->delaunayImages) {
              if (!img->regInfo->stayFixedDuringBundleAdjustment) {
                auto pv = sfm->bai->optimizer->poseVertex(img->index);
                if (img->regInfo->root && !img->regInfo->rootOfRoot) {

                  comp->set_scale(pv->t[2] / 10000);
                  Point2f coords(-pv->t[0], -pv->t[1]);

                  //this is where the offset is officially set for a component. This happens nowhere else.
                  comp->set_offset({0,0});
                  auto offset = get_AbC_relative_from_relative(0, coords, img->regInfo->component_membership);
                  comp->set_offset(offset);
                  composites[img->component_membership]->deduce_label();

                  Point2f pointInBaseSpace(-pv->t[0],-pv->t[1]);
                  auto val = img->debugInitialGuess - pointInBaseSpace;
                  val.x = abs(val.x);
                  val.y = abs(val.y);
                  if (val.x > 500 || val.y > 500) {
                    int k = 0;
                  }

                } else {

                  Point2f pointInBaseSpace(-pv->t[0],-pv->t[1]);
                  auto val = img->debugInitialGuess - pointInBaseSpace;
                  val.x = abs(val.x);
                  val.y = abs(val.y);
                  if (val.x > 500 || val.y > 500) {
                    int k = 0;
                  }

                  img->regInfo->set_AbC_local_from_relative(0,pointInBaseSpace);

                  //debug
                  double diffx = abs(img->absoluteCoords.x - img->regInfo->absoluteCoords.x);
                  double diffy = abs(img->absoluteCoords.y - img->regInfo->absoluteCoords.y);
                  if (diffx > maxX) {
                    maxX = diffx;
                  }
                  if (diffy > maxY) {
                    maxY = diffy;
                  }

                  img->absoluteCoords.x = img->regInfo->absoluteCoords.x;
                  img->absoluteCoords.y = img->regInfo->absoluteCoords.y;

                }
                if (comp->componentIndex == 0) {
                  pv->t[2] = 10000;
                }
                //pv->fixed = true;
                img->regInfo->stayFixedDuringBundleAdjustment = true;
              }
            }
          }
        }

        // for (const auto &stat: sfm->bai->optimizer->batchStatistics()) {
        //   std::printf("iter: %2d, chi2: %.6f\n", stat.iteration + 1, stat.chi2);
        // }

        std::cout << maxX << " " << maxY << std::endl;

        break;
      }
    }

    cudaSetDevice(compositorCudaDevice);
    for (int i = 0; i < composites.size(); ++i) {
      if (i < composites.size() - 1 && composites[i+1]->needsAlignment) {
        //load next component while we rebuild this one
        int ii = i + 1;
        std::thread([this,ii]() { this->load_delaunay_images_to_GPU(ii); }).detach();
      }
      if (composites[i]->needsAlignment) {
        composites[i]->rebuild();
      }
    }

  }


  void StreamCam::load_delaunay_images_to_GPU(int _componentIndex) {
    //this is honestly unhinged to do this without at all checking if the space is available in memory or on the gpu
    //but for now were going with it TODO

    for (auto &img: composites[_componentIndex]->delaunayImages) {
      if (!img->cudaBufferReady) {
        img->load_raw_from_disk();
        img->move_buffer_to_gpu(compositorCudaDevice, true);
      }
    }
  }

  void StreamCam::push_SIFT_matches(std::vector<std::pair<Image *, Image *> > &_newOverlaps, Image *_image) {
    sfm->matchWorkOutstanding += (int) _newOverlaps.size();
    siftQMutex->lock();
    if (compositorCudaDevice != siftCudaDevice) {
      siftDataQueue.push(_image);
    }

    if (_image->regInfo->root) {
      siftMatchQueue.push_front(_newOverlaps);
    }else {
      siftMatchQueue.push_back(_newOverlaps);
    }
    siftQMutex->unlock();
  }

#endif

  void StreamCam::set_match(unsigned long image_idx, unsigned long prev_idx, Match *m) {
    resize_mmatch_mutex->writeLock();
    matchM.match[image_idx][prev_idx] = new Match(m);
    resize_mmatch_mutex->unlock();
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

    std::vector<unsigned long> neighborhood;
    for (unsigned long i = max(0ul, _index - windowWidth); i <= _index + windowWidth; i++) {
      neighborhood.push_back(i);
      JobQ->cancel_job(2, i);
    }

    auto answer = get_image_ref(neighborhood);

    // for (auto img: answer) {
    //   img->mark_too_dark();
    // }
    JobQ->queue_mutex->lock();
    JobQ->update_job_readiness(2, _index);
    JobQ->queue_mutex->unlock();
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
    } else {
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


  std::vector<Image *> StreamCam::get_image_ref(const std::vector<unsigned long int> &_indexes) const {
    /*because images vector can be expanded, this gives access to the pointers within that vector under mutex lock.
    an empty vector of unsigned longs returns entire list of images*/
    std::vector<Image *> temp;

    image_mutex->readLock();
    if (_indexes.empty()) {
      unsigned long i = 0;
      while (images[i] && i <= maxIndex) {
        temp.push_back(images[i++]);
      }
    }else {
      for (unsigned int i = 0; i < _indexes.size(); i++) {
        if (images[_indexes[i]] && _indexes[i] <= maxIndex) {
          temp.push_back(images[_indexes[i]]);
        }
      }
    }
    image_mutex->unlock();

    return temp;
  }

  std::vector<RegInfo *> StreamCam::get_reg_ref(const std::vector<unsigned long> &_indexes) {
    std::vector<RegInfo *> temp;

    reg_results_mutex->writeLock();
    for (unsigned int i = 0; i < _indexes.size(); i++) {
      if (reg_results[_indexes[i]]) {
        temp.push_back(reg_results[_indexes[i]]);
      }
    }
    reg_results_mutex->unlock();

    return temp;
  };

  RegInfo *StreamCam::get_reg_ref(unsigned long image_idx) {
    RegInfo *temp;
    reg_results_mutex->readLock();
    temp = reg_results[image_idx];
    reg_results_mutex->unlock();
    return temp;
  }

  Image *StreamCam::get_image_ref(unsigned long index) {
    Image *temp;
    image_mutex->readLock();
    temp = images[index];
    image_mutex->unlock();
    return temp;
  }

  void StreamCam::add_new_component_Q(unsigned long image_index, cv::Size image_size) {
    auto component_index = increment_and_get_components();
    // if (component_index != 0) {
    //   //start job to find scale and offset
    //   auto xcm = new XCompRunnable(this, image_index, component_index);
    //   JobQ->add_runnable(xcm);
    // }
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

    if (composites.size() == 1) {
      //first component added
      ri->rootOfRoot = true;
      //this becomes the base scale
      temp->set_scale(1);
    } else {
      //this is saying "unknown scale" - will be determined in align_and_rebuild
      temp->set_scale(0);
    }
    temp->set_offset(Point2f(0, 0));
    ri->set_abc(Vec2(0, 0), component_index, true);
    component_mutex->unlock();
  }

  //demo
  void StreamCam::run_agg_classify() {
    if (inferencing && classifying) {
      im->run_agg_classify();
      std::this_thread::sleep_for(std::chrono::seconds(2));
      compositeWait.set();
    }
  }

  void StreamCam::update_observers() {
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
    return get_image_ref(res);
  }

  std::vector<RegInfo *> StreamCam::get_Q_front() {
    compositeQ_mutex->lock();
    std::vector<RegInfo *> temp = compositeBatch.top();
    compositeBatch.pop();
    compositeQ_mutex->unlock();
    return temp;
  }

  std::vector<std::tuple<int, int, unsigned int> > StreamCam::get_tile_embed_Q_front() {
    std::vector<std::tuple<int, int, unsigned int> > temp;

    inferenceQMutex->lock();
    if (!tileEmbedQ.empty()) {
      unsigned int component_index = std::get<2>(tileEmbedQ.front());

      while (!tileEmbedQ.empty() && std::get<2>(tileEmbedQ.front()) == component_index && temp.size() <
             maxTilesPerBatch) {
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


  std::vector<std::pair<Image *, Image *> > StreamCam::get_sift_match_Q_front(std::vector<Image *> &_images) {
    std::vector<std::pair<Image *, Image *> > temp;

    siftQMutex->lock();

    //move all the buffers to my device (if necessary)
    get_sift_data_Q_front(_images);

    //grab a bunch of matches from the Q to process
    if (!siftMatchQueue.empty()) {
      temp = siftMatchQueue.front();
      siftMatchQueue.pop_front();
    }
    siftQMutex->unlock();
    return temp;
  }


  void StreamCam::get_sift_data_Q_front(std::vector<Image *> &_images) {
    std::vector<Image *> temp;

    //no need to check if compositor device is different from sft device, Q will be empty if same -> no need to move data
    while (!siftDataQueue.empty()) {
      auto img = siftDataQueue.front();
      _images.push_back(img);
      cudaMalloc((void **) &img->siftData.d_data, sizeof(SiftPoint) * img->siftData.numPts);
      cudaMemcpy(img->siftData.d_data, img->siftData.h_data, sizeof(SiftPoint) * img->siftData.numPts,
                 cudaMemcpyHostToDevice);
      siftDataQueue.pop();
    }
  }


  void StreamCam::push_tile_embed_Q(std::vector<Point2i> &_tiles, unsigned int _componentIndex) {
    if (inferencing) {
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
    add_image(image, _image_index);
    LoaderLogicRunnable *llr = new LoaderLogicRunnable(this, image, _image_index, true, _saveImg);
    loaderCount++;
    JobQ->add_runnable(llr);
  }

  bool StreamCam::sufficient_distance(Vec2 _coordsInQuestion, int _componentIdx) {
    bool answer = false;

    pixelDistanceMutex->lock();

    if (lastAcceptedCoords.size() < _componentIdx + 1) {
      lastAcceptedCoords.resize(_componentIdx + 1);
      lastAcceptedCoords[_componentIdx] = _coordsInQuestion;
      answer = true;
    } else {
      Vec2 v = lastAcceptedCoords[_componentIdx];
      if (pow(v.x - _coordsInQuestion.x, 2) + pow(v.y - _coordsInQuestion.y, 2) >= minPixelDistanceBetweenFrames) {
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
