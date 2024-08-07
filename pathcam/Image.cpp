#include "pathCam.h"

using namespace cv;

namespace pathCam {


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
                                                                                                              0) {};

  Image::~Image() {
    free_memory_RAW(true);
  }

  bool Image::is_mostly_black() {
    float threshold_value = 20.f;
    int checkPoints = 40;
    float countBlack = 0;
    float tooBlack = 0.2 * float(checkPoints);
    for (int i = 0; i < checkPoints; i++) {
      int x = width / 2 + (scope_radius - 200.f) * cos(float(i) / float(checkPoints) * 2.f * 3.14f);
      int y = height / 2 + (scope_radius - 200.f) * sin(float(i) / float(checkPoints) * 2.f * 3.14f);
      float val = debayer(x, y);
      if (val < threshold_value) { countBlack++; }
      if (countBlack > tooBlack) {
        return true;
      }
    }
    return false;
  }

  double Image::check_blur() {
    if (!in_memory()) {
      throw std::invalid_argument("Image not in memory during blur check");
    }

    cv::Mat temp = cv::Mat(Size(width, height), CV_8UC1, raw_buffer, Mat::AUTO_STEP);
    int steps = 4;
    int radius = 2190;
    double tempVariance = 0;
    for (int i = 0; i < steps; i++) {
      int xloc = width / 2 + (radius - 640) * cos(float(i) / float(steps) * 2.f * 3.14f);
      int yloc = height / 2 + (radius - 640) * sin(float(i) / float(steps) * 2.f * 3.14f);
      xloc += xloc % 2;
      yloc += yloc % 2;
      cv::Rect rectROI(xloc - 64, yloc - 64, 128, 128);
      Mat ROI = temp(rectROI).clone();
      cvtColor(ROI, ROI, COLOR_BayerBG2GRAY);
      /*Mat laplacian;
      cv::Laplacian(ROI, laplacian, CV_64F);
      cv::Scalar mean, stddev;
      cv::meanStdDev(laplacian, mean, stddev); */
      cv::Mat grad_x, grad_y;
      Sobel(ROI, grad_x, CV_64F, 1, 0, 3);
      Sobel(ROI, grad_y, CV_64F, 0, 1, 3);
      cv::Mat grad_magnitude;
      magnitude(grad_x, grad_y, grad_magnitude);
      cv::Scalar mean, stddev;
      cv::meanStdDev(grad_magnitude, mean, stddev);
      double tempVariance = std::pow(stddev[0], 2);

      if (blurVariance < tempVariance) {
        blurVariance = tempVariance;
      }
    }

    return blurVariance;
  }

  void Image::build_whitebalance_Mat(cv::Mat flat_field) {
    Mat image_Mat = cv::Mat(height, width, CV_8U, get_Raw(), Mat::AUTO_STEP);
    Mat gry;
    Mat sbt;
    cvtColor(image_Mat, gry, COLOR_BayerBG2GRAY);
    cvtColor(image_Mat, image_Mat, COLOR_BayerBG2BGR);
    divide(image_Mat, flat_field, image_Mat, 1, CV_8U);
    auto adj = Mat3f(height, width, Vec3f(0.92, 1.08, 0.92));
    Mat locations = gry > 240;
    image_Mat.copyTo(sbt, locations);
    image_Mat -= sbt;
    cv::multiply(sbt, adj, sbt, 1, CV_8U);
    image_Mat += sbt;
    readyImage = image_Mat;
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
  Mat ROI;
  if(!reg_image.empty()){
    if(reg_image.cols < 64 || reg_image.rows < 64){
      reg_image.copyTo(ROI);
    }else{
      Size ROI_size = Size(64,64);
      cv::Rect ROIrect (reg_image.cols/2 - 32, reg_image.rows/2 - 32, 64, 64);
      ROI = reg_image(ROIrect);
    }
  }else{
    buffer_mutex.lock();
    Size image_size = Size(width,height);
    cv::Mat temp = cv::Mat(image_size, CV_8UC1, raw_buffer, Mat::AUTO_STEP);
    unsigned int center_x = width/2;
    center_x += center_x%2; //force it to be even
    unsigned int center_y = height/2;
    center_y += center_y%2; //force it to be even
    cv::Rect ROIrect (center_x - 32, center_y - 32, 64, 64);
    cv::Mat ROI = temp(ROIrect).clone();
    cvtColor(ROI,ROI,COLOR_BayerBG2GRAY);
    buffer_mutex.unlock();
  }
  */
/*
    if (is_mostly_black()) {
      label = _UNDEREXP;
      return;
    }*/
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

    if (label == _2X || label == _4X){
      return blurVariance > 500.0;
    }

    return true;
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

//    reg_scale_initial = _reg_scale;
//    reg_crop_initial = _reg_crop;

    Size image_size = Size(width, height);

    //this should be redone, this extra copy is unnecessary and is a significant inefficiency
    cv::Mat temp = cv::Mat(image_size, CV_8UC1, raw_buffer, Mat::AUTO_STEP);
    temp.copyTo(reg_image);
    buffer_mutex.unlock();

    if (release) { free_memory_RAW(); }

    if (convert) {
      cvtColor(reg_image, reg_image, COLOR_BayerBG2GRAY);
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

