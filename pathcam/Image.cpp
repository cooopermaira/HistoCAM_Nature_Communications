#include "pathCam.h"

using namespace cv;

namespace pathCam {
  cuda::GpuMat Image::hannWindow, Image::cornerRad;
  static Ptr<cuda::Filter> g_gauss;
  static std::once_flag g_gauss_once;

  inline void ensureGauss(int type) {
    std::call_once(g_gauss_once, [type]{
        const cv::Size ksize{5,5};
        const double sigma = 3;
        g_gauss = cv::cuda::createGaussianFilter(type, type, ksize, sigma, sigma,
                                                 cv::BORDER_DEFAULT);
    });
  }
  static std::mutex g_gauss_mtx;
  inline void blur_once(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                     cv::cuda::Stream& s = cv::cuda::Stream::Null())
  {
    ensureGauss(src.type());
    std::lock_guard<std::mutex> lk(g_gauss_mtx);
    g_gauss->apply(src, dst, s);
  }

  Image::Image(unsigned int width, unsigned int height, unsigned int scope_radius, MemoryPool *mempool) : width(width),
                                                                                                          height(
                                                                                                              height),
                                                                                                          scope_radius(
                                                                                                              scope_radius),
                                                                                                          label(
                                                                                                              _NOLABEL),
                                                                                                          mempool(
                                                                                                              mempool),
                                                                                                          raw_buffer(0),raw_buffer_cuda(nullptr),
                                                                                                          reference_count(
                                                                                                              0),
                                                                                                          image_file(
                                                                                                              Poco::Path()),
                                                                                                          motionBlur(
                                                                                                              0),
  cudaBufferReady(false) {
    if (hannWindow.empty()) {
      Mat temp;
      createHanningWindow(temp,Size(512,512),CV_32F);
      hannWindow.upload(temp);
    }
    if (cornerRad.empty()) {
      Mat temp(512,512,CV_8U,Scalar(0));
      circle(temp,Point(0,0),80,Scalar(255),1);
      circle(temp,Point(0,512),80,Scalar(255),1);
      circle(temp,Point(512,0),80,Scalar(255),1);
      circle(temp,Point(512,512),80,Scalar(255),1);
      cornerRad.upload(temp);
    }
  };

  Image::~Image() {
    free_memory_RAW(true);
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
    size_t nBytes = parent->image_height * parent->image_width;
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


  void Image::free_memory_RAW(bool force) {

    buffer_mutex.lock();
    if (raw_buffer != nullptr) {
      reference_count--;
      if (force || reference_count == 0) {
        if (mempool) {
          mempool->release(raw_buffer);
        } else {
          delete[] raw_buffer;
        }
        raw_buffer = nullptr;
      }
    }
    buffer_mutex.unlock();
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

  int Image::check_blur(bool _unifiedMemory) {
    if (!in_memory()) {
      throw std::runtime_error("Image not in memory during blur check");
    }
    cuda::Stream s;

    cuda::GpuMat raw;
    if (_unifiedMemory) {
      //no copy, pretty dope
      raw = cuda::GpuMat(Size(width, height), CV_8U, raw_buffer);
    }else {
      Mat temp(Size(width, height), CV_8U, raw_buffer);
      raw.upload(temp);
    }

    //grab a 512 window in the center to debayer. smart placement of this window would be an improvement
    int roiSize = 512;
    Rect roi(width/2 - roiSize/2, height/2 - roiSize/2, roiSize, roiSize);
    cuda::GpuMat gray;

    //debayer and multiply by hanning window. if you dont, bright lines will corrupt borders and f up min max calc
    cuda::cvtColor(raw(roi),gray,COLOR_BayerBG2GRAY,0,s);
    gray.convertTo(gray,CV_32F);
    cuda::multiply(gray,hannWindow,gray,1,-1,s);

    //compute the log magnitude of the dft
    cuda::GpuMat planes[] = {gray,cuda::GpuMat(gray.rows,gray.cols,CV_32F,Scalar(0))};
    cuda::GpuMat complexI, mag;
    cuda::merge(planes,2,complexI,s);
    cuda::dft(complexI, complexI, Size(gray.cols,gray.rows),0,s);

    cuda::split(complexI,planes,s);
    cuda::magnitude(planes[0],planes[1],mag,s);
    cuda::add(Scalar(1e-6),mag,mag,noArray(),-1,s);
    cuda::log(mag,mag,s);

    // blur the dft so noise doesnt interfere so bad. blur_once is quagmire because the box filter isnt thread safe
    blur_once(mag,mag,s);
    cuda::normalize(mag,mag,0,255,NORM_MINMAX,CV_8U,noArray(),s);

    s.waitForCompletion();

    // //debug for viewing normalized 8U dft
    //Mat temp;
    //mag.download(temp);

    //calculate min max around specific region of dft. this region is uniform if clear and wavy if blurred
    double maxval,minval;
    cuda::minMax(mag,&minval,&maxval,cornerRad);
    motionBlur = int(maxval) - int(minval);
    return motionBlur;
  }



  void Image::write_to_path(bool _profile) {
    std::fstream file;

    auto start = std::chrono::high_resolution_clock::now();

    file = std::fstream(image_file.toString(), std::ios::out | std::ios::binary);
    if (file.fail()) {
      throw new std::exception;
    }
    file.write(get_Raw(), width * height);

    if (_profile) {
      writeTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    }
  }



  void Image::correct_registration(std::vector<unsigned long> adjacentVerts) {
    if (adjacentVerts.size() > 0) {
      //pulling parent reference from odd place, could be passed as parameter
      auto parent = regInfo->parent;
      for (int i = 0; i < adjacentVerts.size(); i++) {
        parent->composites[component_membership]->matchableCount++;
        parent->composites[component_membership]->matchedEdges.resize(adjacentVerts.size(), {-1, -1});
        auto sm = new SingleMatchRunnable(parent, index, adjacentVerts[i], component_membership, i, 0);
        parent->JobQ->add_runnable(sm);
      }
      //wait until these jobs have completed
      while (parent->composites[component_membership]->matchableCount > 0) {
        Poco::Thread::sleep(40);
      }
      auto adjustedAbC = Vec2(0, 0);
      std::vector<Vec2> calcedAbC, calcedOffset, calcedReg;
      std::vector<unsigned long> adjacentVertsKeep;
      int count = 0;
      for (auto i: parent->composites[component_membership]->matchedEdges) {
        if (i.first > -1) {
          auto pwr = parent->matchM.match[i.first][i.second];
          auto theirReg = parent->get_reg_ref(i.second);
          adjustedAbC.x += pwr->t_x + theirReg->absoluteCoords.x;
          adjustedAbC.y += pwr->t_y + theirReg->absoluteCoords.y;

          //debug vectors
          adjacentVertsKeep.push_back(i.second);
          calcedReg.push_back(Vec2(theirReg->absoluteCoords.x, theirReg->absoluteCoords.y));
          calcedOffset.push_back(Vec2(pwr->t_x, pwr->t_y));
          calcedAbC.push_back(Vec2(pwr->t_x + theirReg->absoluteCoords.x, pwr->t_y + theirReg->absoluteCoords.y));
          count++;
        }
      }
      if (count == 0) { return; }
      adjustedAbC.x /= double(count);
      adjustedAbC.y /= double(count);
      if (adjacentVerts.size() > 24) {
        int k = 0;
      }
      if (abs(adjustedAbC.x - regInfo->absoluteCoords.x) > 50 || abs(adjustedAbC.y - regInfo->absoluteCoords.y) > 100) {
        int k = 0;
      }
      parent->composites[component_membership]->matchedEdges.clear();
      regInfo->set_abc(adjustedAbC, component_membership, false);
    } else {
      regInfo->attempt_absolute_reg(false);
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
    if (label == _UNDEREXP) {
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

    label = _UNKNOWN;
    return;
  }

  bool Image::decide_label_and_blur() {
    //this function will return false if image fails PRELIMINARY blur test. Image might still be blurry
    if (!in_memory()) {
      throw std::invalid_argument("image not in memory during decide_label_and_blur()");
    }



    find_label();

    if (label == _2X || label == _4X) {
      return motionBlur > 500.0;
    }

    return true;
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
      cv::Rect myROI((old_image_size.width / 2) - (image_size.width / 2),
                     (old_image_size.height / 2) - image_size.height / 2,
                     image_size.width, image_size.height);
      reg_image = reg_image(myROI);
    }
    int k = 0;

  };

  void Image::load_raw_from_disk() {
    buffer_mutex.lock();
    if (raw_buffer == 0) {
      if (image_file.toString() != "") {
        std::ifstream stream;
        stream.open(image_file.toString(), std::ios::binary);
        allocate_memory_RAW();
        stream.read(raw_buffer, width * height);
        stream.close();
      } else {
        std::cerr << "Loading from disk with no path\n";
        buffer_mutex.unlock();
        return;
      }
    }
    reference_count++;
    buffer_mutex.unlock();
  }

}

