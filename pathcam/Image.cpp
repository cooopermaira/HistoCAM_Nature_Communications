#include "pathCam.h"

using namespace cv;

namespace pathCam {
  static cv::Mat filterAnomaliesRelMean(const cv::Mat& img_f32, double theta) {
    CV_Assert(img_f32.type() == CV_32F);
    cv::Mat out = img_f32.clone();
    const int H = img_f32.rows, W = img_f32.cols;
    auto pix = [&](int y,int x){ return img_f32.at<float>(y,x); };
    for (int y = 1; y < H-1; ++y) {
      for (int x = 1; x < W-1; ++x) {
        float sum = 0.f;
        int n = 0;
        for (int dy=-1; dy<=1; ++dy) {
          for (int dx=-1; dx<=1; ++dx) {
            if (dx||dy) {
              sum += pix(y+dy,x+dx);
              ++n;
            }
          }
        }
        float mu = sum / std::max(n,1);
        float p  = pix(y,x);
        // avoid division by tiny mu: if mu≈0, only consider large absolute deviation
        bool is_anom = (std::abs(mu) > 1e-6f) ? (std::abs(p - mu) / std::abs(mu) > theta)
                                              : (std::abs(p) > 0.f); // crude but safe
        if (is_anom) out.at<float>(y,x) = mu;
      }
    }
    return out;
  }

  static float meanRelativeDrop(const cv::Mat& num, const cv::Mat& den, float eps=1e-6f) {
    CV_Assert(num.type() == CV_32F && den.type() == CV_32F);
    double sum = 0.0; size_t cnt = 0;
    for (int y=0; y<num.rows; ++y) {
      const float* np = num.ptr<float>(y);
      const float* dp = den.ptr<float>(y);
      for (int x=0; x<num.cols; ++x) {
        float d = dp[x];
        if (d > eps) { sum += (double)np[x] / (double)d; ++cnt; }
      }
    }
    return cnt ? (float)(sum / (double)cnt) : 0.f;
  }

  static std::pair<float,float> maskedAbsGradPercentiles(const cv::Mat& grad_f32, const cv::Mat& mask_u8,
                                                       double pL, double pU) {
    CV_Assert(grad_f32.type() == CV_32F && mask_u8.type() == CV_8U);
    std::vector<float> vals; vals.reserve(grad_f32.rows*grad_f32.cols/10);
    for (int y=0; y<grad_f32.rows; ++y) {
      const float* gptr = grad_f32.ptr<float>(y);
      const uchar* mptr = mask_u8.ptr<uchar>(y);
      for (int x=0; x<grad_f32.cols; ++x) if (mptr[x]) vals.push_back(std::abs(gptr[x]));
    }
    if (vals.empty()) return {0.f, 0.f};
    std::sort(vals.begin(), vals.end());
    auto pick = [&](double p)->float {
      if (vals.empty()) return 0.f;
      double idx = (p/100.0) * (vals.size()-1);
      size_t i0 = (size_t)std::floor(idx);
      size_t i1 = std::min(i0+1, vals.size()-1);
      double t = idx - i0;
      return (float)((1.0-t)*vals[i0] + t*vals[i1]);
    };
    return {pick(pL), pick(pU)};
  }

  // Keep only values whose abs lies within [pL,pU] percentiles; returns filtered gradient (others zeroed)
  static cv::Mat filterGradByPercentiles(const cv::Mat& grad_f32, const cv::Mat& mask_u8,
                                         double pL, double pU) {
    auto [lo, hi] = maskedAbsGradPercentiles(grad_f32, mask_u8, pL, pU);
    cv::Mat out = cv::Mat::zeros(grad_f32.size(), CV_32F);
    if (hi <= 0.f) return out;
    for (int y=0; y<grad_f32.rows; ++y) {
      const float* gp = grad_f32.ptr<float>(y);
      const uchar* mp = mask_u8.ptr<uchar>(y);
      float*       op = out.ptr<float>(y);
      for (int x=0; x<grad_f32.cols; ++x) {
        if (!mp[x]) continue;
        float a = std::abs(gp[x]);
        if (a >= lo && a <= hi) op[x] = a; // store magnitude (paper multiplies mask; magnitude is fine)
      }
    }
    return out;
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
                                                                                                          raw_buffer(0),
                                                                                                          reference_count(
                                                                                                              0),
                                                                                                          image_file(
                                                                                                              Poco::Path()),
                                                                                                          blurVariance(
                                                                                                              1000),
  cudaBufferReady(false) {};

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

  Point2f Image::compute_sharpness(Mat &_img) {
    CV_Assert(_img.type() == CV_8U);
    Mat img32;

    _img.convertTo(img32,CV_32F);
    auto denoised = filterAnomaliesRelMean(img32,0.5);

    Mat mask = (_img > 0) & (_img < 255);
    assert(mask.type() == CV_8U);

    Mat gx, gy;
    Sobel(denoised,gx,CV_32F,1,0,5);
    Sobel(denoised,gy,CV_32F,0,1,5);

    Mat sbx = filterGradByPercentiles(gx,mask,98.5,99.5);
    Mat sby = filterGradByPercentiles(gy,mask,98.5,99.5);

    Mat blur_f32;
    GaussianBlur(denoised,blur_f32,Size(5,5),1,1);

    Mat bgx,bgy;
    Sobel(blur_f32,bgx,CV_32F,1,0,5);
    Sobel(blur_f32,bgy,CV_32F,0,1,5);

    Mat GBx = filterGradByPercentiles(bgx,mask,98.5,99.5);
    Mat GBy = filterGradByPercentiles(bgy,mask,98.5,99.5);

    Mat dx,dy;
    subtract(sbx,GBx,dx,noArray(),CV_32F);
    subtract(sby,GBy,dy,noArray(),CV_32F);

    float mx = meanRelativeDrop(dx, sbx);
    float my = meanRelativeDrop(dy, sby);

    Point2f r;
    r.x = 100.f * std::max(0.f, std::min(mx, 1.f)); // clamp to [0,100] if desired
    r.y = 100.f * std::max(0.f, std::min(my, 1.f));
    return r;
  }


  double Image::check_blur() {
    if (!in_memory()) {
      throw std::runtime_error("Image not in memory during blur check");
    }

    cv::Mat temp = cv::Mat(Size(width, height), CV_8UC1, raw_buffer, Mat::AUTO_STEP);
    int steps = 6;
    int radius = 2190;
    for (int i = 0; i < steps; i++) {
      int xloc = width / 2 + (radius - 640) * cos(float(i) / float(steps) * 2.f * 3.14f);
      int yloc = height / 2 + (radius - 640) * sin(float(i) / float(steps) * 2.f * 3.14f);
      xloc += xloc % 2;
      yloc += yloc % 2;
      cv::Rect rectROI(xloc - 64, yloc - 64, 128, 128);
      Mat ROI = temp(rectROI).clone();
      cvtColor(ROI, ROI, COLOR_BayerBG2GRAY);

      auto val = compute_sharpness(ROI);
      float p = min(val.x,val.y);
      blurVariance = min(blurVariance,p);


      //imwrite("/media/max/Data/blur_test/roi.png",ROI);
      int k = 0;
    }

    return blurVariance;
  }

  void Image::write_to_path() {
    std::fstream file;
    file = std::fstream(image_file.toString(), std::ios::out | std::ios::binary);
    if (file.fail()) {
      throw new std::exception;
    }
    file.write(get_Raw(), width * height);
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

    if (check_blur() < 200) {
      return false;
    }

    find_label();

    if (label == _2X || label == _4X) {
      return blurVariance > 500.0;
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

