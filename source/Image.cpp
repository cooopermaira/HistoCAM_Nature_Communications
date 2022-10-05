#include "common.h"

using namespace cv;

namespace pathCam{

Image::Image(char *raw_buffer): width(6464), height(4852), raw_buffer(raw_buffer), filename(""){ };

Image::~Image(){
  free_memory_RAW();
}

void Image::allocate_memory_RAW(){
  if(raw_buffer ==0){
    raw_buffer = new char[31363328];
  }
}

void Image::free_memory_RAW(){
  delete[] raw_buffer;
  raw_buffer = 0;
}



void Image::create_reg_image(float _reg_scale, bool convert){
  reg_scale = _reg_scale;
  
  if(raw_buffer == 0){ return; }
  reg_image.release();
  reg_image = cv::Mat(Size(6464,4852), CV_8UC1, raw_buffer, Mat::AUTO_STEP);
  if(convert){ cvtColor(reg_image,reg_image,COLOR_BayerBG2GRAY); }
  cv::resize(reg_image, reg_image, Size(6464/reg_scale,4852/reg_scale));
  
};

}

