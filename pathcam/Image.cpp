#include "pathCam.h"

using namespace cv;

namespace pathCam {
  cuda::GpuMat Image::hannWindow, Image::blurMask;
  static Ptr<cuda::Filter> g_gauss;
  static std::once_flag g_gauss_once;


  // inline void sortSiftDataByX(SiftData& sd) {
  //   assert(sd.h_data);
  //   assert(sd.numPts <= sd.maxPts);
  //
  //   std::sort(sd.h_data, sd.h_data + sd.numPts,
  //             [](const SiftPoint& a, const SiftPoint& b) {
  //               return a.xpos < b.xpos;
  //             });
  //
  //   cudaMemcpy(sd.d_data,
  //          sd.h_data,
  //          sd.numPts * sizeof(SiftPoint),
  //          cudaMemcpyHostToDevice);
  // }


  static std::mutex g_gauss_mtx;

  inline void blur_once(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst,
                        cv::cuda::Stream &s = cv::cuda::Stream::Null()) {
    //ensureGauss(src.type());
    std::lock_guard<std::mutex> lk(g_gauss_mtx);
    g_gauss->apply(src, dst, s);
  }

  void Image::cleanup_blur_check_statics() {
    static std::once_flag cleanup_flag;

    std::call_once(cleanup_flag, [] {

      if (!hannWindow.empty())
        hannWindow.release();

    g_gauss.release();  // cv::Ptr reset

    cudaDeviceSynchronize();
});
  }


  void Image::prepare_blur_check_statics() const {
    static std::once_flag flag;
    std::call_once(flag, [&] {

      Mat temp(blurPatch, blurPatch,CV_8U, Scalar(0));


      createHanningWindow(temp, Size(blurPatch, blurPatch),CV_32F);
      hannWindow.upload(temp);

      const cv::Size ksize{5, 5};
      const double sigma = 3;
      g_gauss = cv::cuda::createGaussianFilter(CV_32F, CV_32F, ksize, sigma, sigma,
                                               BORDER_DEFAULT);
    });
  }

  Image::Image(unsigned int width, unsigned int height, unsigned int scope_radius,
               MemoryPool *mempool) : width(width),
                                      height(
                                        height),
                                      scope_radius(
                                        scope_radius),
                                      label(
                                        _NOLABEL),
                                      mempool(
                                        mempool),
                                      raw_buffer(0), raw_buffer_cuda(nullptr),
                                      reference_count(
                                        0),
                                      image_file(
                                        Poco::Path()),
                                      motionBlur(
                                        10000),
                                      cudaBufferReady(false) {
    prepare_blur_check_statics();

  };

  Image::~Image() {
    free_memory_RAW(true);
    // if (regInfo) {
    //   delete regInfo;
    // }
  }

#ifdef HAVE_OPENCV_CUDAARITHM
  bool Image::move_buffer_to_gpu(int _device, bool _freeHostBuffer) {
    // if (raw_buffer_cuda) {
    //   std::lock_guard lock(cudaBufferMutex);
    //   cudaBufferReady = true;
    //   cudaBufferConVar.notify_one();
    //   return false;
    // }

    if (!raw_buffer) {
      return false;
    }
    CHECK_CUDA(cudaSetDevice(_device));
    size_t nBytes = height * width;
    CHECK_CUDA(cudaMalloc(&raw_buffer_cuda,nBytes));
    CHECK_CUDA(cudaMemcpy(raw_buffer_cuda,raw_buffer,nBytes,cudaMemcpyHostToDevice));

    {
      std::lock_guard lock(cudaBufferMutex);
      cudaBufferReady = true;
      cudaBufferConVar.notify_one();
    }

    //release from system memory, keep on gpu only
    if (_freeHostBuffer) {
      free_memory_RAW();
    }

    return true;
  }
#endif

  void Image::free_memory_cuda() {
    if (raw_buffer_cuda != nullptr) {
      cudaFree(raw_buffer_cuda);
    }
    raw_buffer_cuda = nullptr;
    cudaBufferReady = false;
  }


  bool Image::is_mostly_black() {
    float threshold_value = 20.f;
    int checkPoints = 40;
    float countBlack = 0;
    float tooBlack = 0.4 * float(checkPoints);
    for (int i = 0; i < checkPoints; i++) {
      int x = width / 2 + (scope_radius - 400.f) * cos(float(i) / float(checkPoints) * 2.f * 3.14f);
      int y = height / 2 + (scope_radius - 400.f) * sin(float(i) / float(checkPoints) * 2.f * 3.14f);
      float val = debayer(x, y);
      if (val < threshold_value) { countBlack++; }
      if (countBlack > tooBlack) {
        return true;
      }
    }
    return false;
  }


  void Image::check_blur_async(const Mat &img, bool submitForInference) {
     auto start = std::chrono::high_resolution_clock::now();
    Mat grayHost;
    if (!img.empty()) {
      assert(img.rows == blurPatch && img.cols == blurPatch && img.channels() == 1);
      grayHost = img.clone();
    } else {
      if (!in_memory()) {
        throw std::runtime_error("Image not in memory during blur check");
      }
      const Mat raw(Size(width, height), CV_8U, raw_buffer);
      const Rect roi(width / 2 - blurPatch / 2, height / 2 - blurPatch / 2, blurPatch, blurPatch);

      cvtColor(raw(roi), grayHost, COLOR_BayerBG2GRAY);
    }
    cuda::Stream s;
    if (grayHost.depth() != CV_32F) {
      grayHost.convertTo(grayHost,CV_32F);
    }
    cuda::GpuMat gray(grayHost.rows, grayHost.cols,CV_32FC1, grayHost.data);
    cuda::multiply(gray, hannWindow, gray, 1, -1, s);

    //compute the log magnitude of the dft
    char *buff;
    cudaMallocManaged(&buff, blurPatch * blurPatch * sizeof(float));

    cuda::GpuMat planes[] = {gray, cuda::GpuMat(gray.rows, gray.cols,CV_32F, Scalar(0))};
    cuda::GpuMat complexI, mag, rcv(blurPatch, blurPatch,CV_32FC1, buff);
    cuda::merge(planes, 2, complexI, s);
    cuda::dft(complexI, complexI, Size(gray.cols, gray.rows), 0, s);

    cuda::split(complexI, planes, s);
    cuda::magnitude(planes[0], planes[1], mag, s);
    cuda::add(Scalar(1e-6), mag, mag, noArray(), -1, s);
    cuda::log(mag, mag, s);

    // blur the dft so noise doesnt interfere so bad. blur_once is quagmire because the box filter isnt thread safe

    blur_once(mag, mag, s);
    //cuda::normalize(mag, rcv, 0, 255, NORM_MINMAX,CV_8U, noArray(), s);
    cuda::normalize(mag, rcv, 0, 1, NORM_MINMAX,CV_32F, noArray(), s);

    s.waitForCompletion();

    blurDFT = Mat(blurPatch, blurPatch,CV_32FC1, buff);
    resize(blurDFT, blurDFT, Size(128, 128));
    cudaFree(buff);

    if (submitForInference) {
      parent->Q_blur_metric(this);
    }

    blurTime = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
  }


  void Image::write_to_path(bool _profile) {
    auto start = std::chrono::high_resolution_clock::now();

    std::ofstream file(image_file.toString(), std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);

    file.write(get_Raw(), width * height);

    if (_profile) {
      blurTime = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - start).count();
    }
  }






  bool Image::is_mostly_white(Mat ROI) {
    unsigned int threshold_value = 225;
    Mat thresholded;
    threshold(ROI, thresholded, threshold_value, 255, THRESH_BINARY);
    unsigned int total_pixels = (unsigned int) ROI.total();
    unsigned int white_pixels = countNonZero(thresholded);
    return white_pixels > (total_pixels / 2);
  }

  float Image::debayer(int x, int y) {
    //Assuming RGGB
    float red = (uint8_t) raw_buffer[y * width + x];
    float green = (uint8_t) raw_buffer[(y + 1) * width + x];
    green += (uint8_t) raw_buffer[(y - 1) * width + x];
    green += (uint8_t) raw_buffer[(y) * width + x + 1];
    green += (uint8_t) raw_buffer[(y) * width + x - 1];
    green /= 4.0;
    float blue = (uint8_t) raw_buffer[(y + 1) * width + x + 1];
    blue += (uint8_t) raw_buffer[(y + 1) * width + x - 1];
    blue += (uint8_t) raw_buffer[(y - 1) * width + x - 1];
    blue += (uint8_t) raw_buffer[(y - 1) * width + x + 1];
    blue /= 4.0;
    return 0.30 * red + 0.59 * green + 0.11 * blue;
  }

  bool Image::is_4x() {
    if (!in_memory()) {
      throw std::invalid_argument("Image not in memory during 4x check");
    }
    buffer_mutex.lock();
    auto corner1 = Rect(0, 0, 64, 64);
    auto corner2 = Rect(width - 64, 0, 64, 64);
    auto corner3 = Rect(width - 64, height - 64, 64, 64);
    auto corner4 = Rect(0, height - 64, 64, 64);

    Mat image = cv::Mat(height, width, CV_8U, get_Raw(), Mat::AUTO_STEP);
    Mat roi1, roi2, roi3, roi4;

    cvtColor(image(corner1), roi1, COLOR_BayerBG2GRAY);
    cvtColor(image(corner2), roi2, COLOR_BayerBG2GRAY);
    cvtColor(image(corner3), roi3, COLOR_BayerBG2GRAY);
    cvtColor(image(corner4), roi4, COLOR_BayerBG2GRAY);

    auto centerRect = Rect(width / 2 - 128, height / 2 - 128, 256, 256);
    Mat centerRoi;
    cvtColor(image(centerRect), centerRoi, COLOR_BayerBG2GRAY);
    buffer_mutex.unlock();

    auto val = cv::mean(roi1);
    val += cv::mean(roi2);
    val += cv::mean(roi3);
    val += cv::mean(roi4);
    val /= 4.0;


    auto difference = (cv::mean(centerRoi) - val)[0];
    return difference >= 120.0;
  }

  bool Image::is_2x() {
    //return true;
    if (!in_memory()) {
      throw std::invalid_argument("Image not in memory during 2x check");
    }
    buffer_mutex.lock();
    float center = 0.f;
    int steps = 20;
    int radius_of_test = 1000; //pixels
    for (int i = 0; i < steps; i++) {
      //creates an X of samples across 2x image viewport
      int j = 2 * i * radius_of_test / (steps - 1) - radius_of_test;
      center += debayer(width / 2 + j, height / 2 + j);
      center += debayer(width / 2 + j, height / 2 - j);
    }
    center /= (2 * steps);
    float outside2X = debayer(width * 0.9, height * 0.9);
    buffer_mutex.unlock();
    return (center - outside2X) >= 150;
  }


  bool Image::is_good() {
    //return label == _2X;
    if (label == _UNDEREXP || label == _LOWFEAT) {
      return false;
    }
    return true;
  }

  void Image::find_label() {
    /*
        if (is_mostly_black()) {
          label = _UNDEREXP;
          return;
        }*/

    manually_set_label();
    //label = _10X;
    //label = _4X;
    //label = _2X;
    return;
    if (is_2x()) {
      label = _2X;
      return;
    }

    if (is_4x()) {
      label = _4X;
      return;
    }

    label = _NOLABEL;
    return;
  }

  void Image::set_observed_label(const std::string &_label) {
    if (_label == "02") {
      label = _2X;
      labelObserved = true;
    }else if (_label == "04") {
      label = _4X;
      labelObserved = true;
    }else if (_label == "10") {
      label = _10X;
      labelObserved = true;
    }else if (_label == "20") {
      label = _20X;
      labelObserved = true;
    }else if (_label == "40") {
      label = _40X;
      labelObserved = true;
      std::cout<<"WARNING 40X"<<std::endl;
    }
  }

  void Image::manually_set_label() {
    //    if (index < 649) {
    //      label = Image::_2X;
    //    } else if ((index >= 649 && index < 754) || (index >= 832 && index < 1103)) {
    //      label = Image::_4X;
    //    } else if ((index >= 754 && index < 832) || (index >= 1103 && index < 1365)) {
    //      label = Image::_10X;
    //    } else {
    //      label = Image::_20X;
    //    }

    ////2_20new
    //    if(index < 413){
    //      label = Image::_2X;
    //    }else if(index >= 413 && index <777) {
    //      label = _4X;
    //    }else if(index >= 777 && index < 1358){
    //      label = Image::_10X;
    //    }else{
    //      label = Image::_20X;
    //    }

    //afb
    if (index < 141) {
      label = Image::_4X;
    } else if (index >= 141 && index < 315) {
      label = _10X;
    } else if (index >= 315 && index < 1358) {
      label = Image::_40X;
    }
  }

  cv::Mat Image::full_image_asMat() {
    load_raw_from_disk();
    Mat image_Mat = cv::Mat(width, height, CV_8U, get_Raw(), Mat::AUTO_STEP);
    cvtColor(image_Mat, image_Mat, COLOR_BayerBG2BGR);

    free_memory_RAW();
    return image_Mat;
  }

  char *Image::get_raw_cuda() {
    //assert(raw_buffer_cuda);
    if (parent && parent->unifiedMemory) {
      //assert(raw_buffer);
      return raw_buffer;
    }
    return raw_buffer_cuda;
  }

  void Image::create_reg_image(double _reg_scale, double _reg_crop, bool convert, int interpolation, bool real) {
    bool release = false;
    buffer_mutex.lock();

    if (!reg_image.empty()) {
      buffer_mutex.unlock();
      return;
    }

    if (raw_buffer == 0) {
      buffer_mutex.unlock();
      load_raw_from_disk();
      buffer_mutex.lock();
      release = true;
    }


    Size image_size = Size(width, height);

    Mat temp = cv::Mat(image_size, CV_8UC1, raw_buffer, Mat::AUTO_STEP);
    reg_image = temp.clone();
    buffer_mutex.unlock();


    if (release) {
      free_memory_RAW();
    }


    if (convert) {
      //      if(flatfield_first){
      //        cvtColor(reg_image,reg_image,COLOR_BayerBG2BGR);
      //        divide(reg_image, flatfield, reg_image, 1, CV_8U);
      //        cvtColor(reg_image,reg_image,COLOR_BGR2GRAY);
      //      }else {
      cvtColor(reg_image, reg_image, COLOR_BayerBG2GRAY);
      //}
    }
    if (real) {
      reg_image.convertTo(reg_image, CV_32FC1);
    }
    if (_reg_scale != 1.0) {
      image_size = Size(image_size.width * _reg_scale, image_size.height * _reg_scale);
      cv::resize(reg_image, reg_image, image_size);
    }
    if (_reg_crop != 1.0) {
      Size old_image_size = image_size;
      image_size = Size(image_size.width * _reg_crop, image_size.height * _reg_crop);
      Rect myROI((old_image_size.width / 2) - (image_size.width / 2),
                 (old_image_size.height / 2) - image_size.height / 2,
                 image_size.width, image_size.height);
      reg_image = reg_image(myROI);
    }
    int k = 0;
  };

  double Image::get_reg_scale() const {
    return parent->scale_factor;
  }

  void Image::allocate_memory_RAW() {
    if (!raw_buffer) {
      // if (mempool) {
      //   raw_buffer = reinterpret_cast<char *>(mempool->get());
      // } else {
      //   if (parent && parent->unifiedMemory) {
      cudaMallocManaged(&raw_buffer,width * height);
      //   }else {
      //     raw_buffer = new char[width * height];
      //   }
      // }
    }
  }

  unsigned int Image::get_label() const {
    if (label == _NOLABEL) {
      if (regInfo && regInfo->component_membership >= 0) {
        if (parent) {
          return parent->composites[regInfo->component_membership]->componentMagLabel;
        }
      }
    }
    return label;
  }

  void Image::load_raw_from_disk(bool _alertDoubleLoad) {
    buffer_mutex.lock();
    if (!raw_buffer) {
      if (hasBeenInMemory && _alertDoubleLoad) {
        std::cout<<"image "<<index<<" double load"<<std::endl;
      }

      ++loadCount;

      if (image_file.toString() != "") {
        std::ifstream stream;
        stream.open(image_file.toString(), std::ios::binary);
        allocate_memory_RAW();
        stream.read(raw_buffer, width * height);
        stream.close();
        hasBeenInMemory = true;
      } else {
        std::cerr << "Loading from disk with no path\n";
        buffer_mutex.unlock();
        return;
      }
    }
    ++reference_count;
    buffer_mutex.unlock();
  }


  void Image::free_memory_RAW(bool force) {
    buffer_mutex.lock();
    if (raw_buffer != nullptr) {
      --reference_count;
      if (force || reference_count == 0) {
        // if (mempool) {
        //   mempool->release(raw_buffer);
        // } else {
        if (parent && parent->unifiedMemory) {
          cudaFree(raw_buffer);
        }else {
          free(raw_buffer);
        }
        // }
        // cudaFree(raw_buffer);
        raw_buffer = nullptr;
        reference_count = 0;
      }
    }
    buffer_mutex.unlock();
  }

/*
  void Image::extract_sift(int numPts, int octaves, float initBlur, float thresh,
                           float lowestScale, cuda::GpuMat &buffer, bool siftWindow, float *tempSpace) {
    if ((siftInitialized && siftWindow) || (siftFullInitialized && !siftWindow)){return;}

    Rect roi((width - buffer.cols)/2, (height - buffer.rows)/2,buffer.cols,buffer.rows);

    buffer_mutex.lock();
    assert(get_Raw());

    cuda::GpuMat raw(height, width, CV_8UC1, get_Raw(),width);
    raw(roi).convertTo(buffer,CV_32F);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      printf("convertTo launch error: %s\n", cudaGetErrorString(e));
      abort();
    }

    // 2) Force completion so any illegal access in OpenCV shows up HERE
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("convertTo sync error: %s\n", cudaGetErrorString(e));
      abort();
    }


    buffer_mutex.unlock();

    CudaImage cImgRaw;
    cImgRaw.Allocate(buffer.cols, buffer.rows, buffer.step / sizeof(float), false,
                         reinterpret_cast<float *>(buffer.data), nullptr);

    if (siftWindow) {
      assert(buffer.rows == parent->siftWindow && buffer.cols == parent->siftWindow);

      InitSiftData(siftData, numPts, true, true);

      parent->CudaSiftGlobalUseMutex.lock();
      auto start = std::chrono::high_resolution_clock::now();
      ExtractSift(siftData,cImgRaw,octaves,initBlur,thresh,lowestScale,false,tempSpace);
      parent->cudaSiftTime += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
      parent->CudaSiftGlobalUseMutex.unlock();

      assert(siftData.numPts > 0);
      sortSiftDataByX(siftData);
      siftInitialized = true;
    }else {
      assert(buffer.rows == height && buffer.cols == width);
      InitSiftData(siftDataFull,numPts,true,true);

      parent->CudaSiftGlobalUseMutex.lock();
      catch_ExtractSift(siftDataFull,cImgRaw,octaves,initBlur,thresh,lowestScale,false);
      parent->CudaSiftGlobalUseMutex.unlock();

      assert(siftDataFull.numPts > 0);
      siftFullInitialized = true;
    }
  }
  */
}
