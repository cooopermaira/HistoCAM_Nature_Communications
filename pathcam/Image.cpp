#include "pathCam.h"

using namespace cv;

namespace pathCam{

Image::Image(MemoryPool *mempool): width(6464), height(4852), label(_NOLABEL), mempool(mempool), raw_buffer(0), reference_count(0), image_file(Poco::Path()){};

Image::~Image(){
  free_memory_RAW(true);
}

bool Image::is_mostly_black(double threshold_value){
  if(!reg_image.empty()){
    Mat thresholded;
    threshold(reg_image, thresholded, threshold_value, 255, THRESH_BINARY);
    unsigned int black_pixels = countNonZero(thresholded);
    unsigned int total_pixels = (unsigned int)reg_image.total();
    
    return black_pixels > (total_pixels/2);
  }
  return false;
}

void Image::create_reg_image(double _reg_scale, double _reg_crop, bool convert, int interpolation, bool real){
  buffer_mutex.lock();
  if(raw_buffer == 0 || !reg_image.empty()){ buffer_mutex.unlock(); return; }
  
  reg_scale = _reg_scale;
  reg_crop = _reg_crop;
  
  Size image_size = Size(width,height);

  cv::Mat temp = cv::Mat(image_size, CV_8UC1, raw_buffer, Mat::AUTO_STEP);
  temp.copyTo(reg_image);
 
  if(convert){
    cvtColor(reg_image,reg_image,COLOR_BayerBG2GRAY);
  }
  if(real){
    reg_image.convertTo(reg_image, CV_32FC1);
  }
  if(reg_scale != 1.0){
    image_size = Size(image_size.width*reg_scale,image_size.height*reg_scale);
    cv::resize(reg_image, reg_image,image_size);
  }
  if(reg_crop != 1.0){
    Size old_image_size = image_size;
    image_size = Size(image_size.width*reg_crop,image_size.height*reg_crop);
    cv::Rect myROI((old_image_size.width/2) - (image_size.width/2),
                   (old_image_size.height/2) - image_size.height/2,
                   image_size.width, image_size.height);
    reg_image = reg_image(myROI);
  }
  
  buffer_mutex.unlock();

};

}

