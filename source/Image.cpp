#include "common.h"

using namespace cv;

namespace pathCam{

Image::Image(char *raw_buffer): width(6464), height(4852), raw_buffer(raw_buffer), filename(""){ };

Image::~Image(){
  free_memory_RAW();
}

void Image::create_reg_image(double _reg_scale, double _reg_crop, bool convert, int interpolation){
  reg_scale = _reg_scale;
  reg_crop = _reg_crop;
  
  Size image_size = Size(6464,4852);

  if(raw_buffer == 0){ return; }
  reg_image.release();
  //this will change raw_buffer, not a copy!
  reg_image = cv::Mat(image_size, CV_8UC1, raw_buffer, Mat::AUTO_STEP);
  if(convert){
    cvtColor(reg_image,reg_image,COLOR_BayerBG2GRAY);
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

};

}

