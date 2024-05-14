#include "pathCam.h"

using namespace cv;

namespace pathCam{

Image::Image(MemoryPool *mempool): width(6464), height(4852), label(_NOLABEL), mempool(mempool), raw_buffer(0), reference_count(0), image_file(Poco::Path()){};

Image::~Image(){
  free_memory_RAW(true);
}

bool Image::is_mostly_black(Mat ROI){
  unsigned int threshold_value = 25;
  Mat thresholded;
  threshold(ROI, thresholded, threshold_value, 255, THRESH_BINARY);
  unsigned int total_pixels = (unsigned int)ROI.total();
  unsigned int black_pixels = total_pixels-countNonZero(thresholded);
  return black_pixels > (total_pixels/2);
  
}

bool Image::is_mostly_white(Mat ROI){
  unsigned int threshold_value = 225;
  Mat thresholded;
  threshold(ROI, thresholded, threshold_value, 255, THRESH_BINARY);
  unsigned int total_pixels = (unsigned int)ROI.total();
  unsigned int white_pixels = countNonZero(thresholded);
  return white_pixels > (total_pixels/2);
}

float Image::debayer(int x, int y){
  //Assuming RGGB
  float red = (uint8_t)raw_buffer[y*width + x];
  float green = (uint8_t)raw_buffer[(y+1)*width + x];
  green += (uint8_t)raw_buffer[(y-1)*width + x];
  green += (uint8_t)raw_buffer[(y)*width + x+1];
  green += (uint8_t)raw_buffer[(y)*width + x-1];
  green /= 4.0;
  float blue = (uint8_t)raw_buffer[(y+1)*width + x+1];
  blue += (uint8_t)raw_buffer[(y+1)*width + x-1];
  blue += (uint8_t)raw_buffer[(y-1)*width + x-1];
  blue += (uint8_t)raw_buffer[(y-1)*width + x+1];
  blue /= 4.0;
  return 0.30*red + 0.59*green + 0.11*blue;
}

bool Image::is_2x(){
  buffer_mutex.lock();
  float center = 0.f;
  int steps = 20;
  int radius_of_test = 1000; //pixels
  for (int i = 0;i < steps; i++) {
    int j = 2*i*radius_of_test/(steps - 1) - radius_of_test;
    center += debayer(width/2 + j, height/2 + j);
    center += debayer(width/2 + j, height/2 - j);
  }
  center/=(2*steps);
  float center_bottom = debayer(width/2, height-3);
  buffer_mutex.unlock();
  if(image_file.toString()=="frame-10282021160818-570.Raw") {
    int k = 0;
  }
  return (center - center_bottom) >= 180;
  
}


void Image::find_label(){
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
  
  
  if(is_mostly_black(ROI)){ label = _UNDEREXP; return; }
  if(is_2x()){
    label = _2X;
    /*
    unsigned int count_non_black = 0;
    for(unsigned int i=0; i < width; i++){
      if(raw_buffer[int(height/2)*width + i] > 50){
        count_non_black++;
      }
    }
    
    if(count_non_black < 4800){
      label = _LENS_CHANGE;
    }
    */
    return;
  }
  
  
  label = _UNKNOWN;
  return;
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
    //Mat mask = cv::Mat::zeros(cv::Size(image_size), CV_64F);
    //circle(mask, cv::Point(3232, 2426), int(2190), (255, 255, 255), -1);
    //bitwise_and(reg_image, reg_image, mask = mask);
    //Mat outframe;
    //reg_image.copyTo(outframe, mask);
  }
  
  buffer_mutex.unlock();

};

}

