#ifndef IMAGE
#define IMAGE

#include "common.h"

namespace pathCam{
class Image{
public:
  
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  
  
  Image(char *raw_buffer=0);
  ~Image();
  
  void set_disk_file(std::string _filename){
    filename = _filename;
  }
  
  bool load_raw_from_disk(){
    if(filename != ""){
      std::ifstream stream;
      stream.open(filename, std::ios::binary);
      if(raw_buffer == 0){ allocate_memory_RAW(); }
      stream.read(raw_buffer,31363328);
      stream.close();
    }
  }
  
  void create_reg_image(double reg_scale, double reg_crop, bool convert=true, int interpolation=cv::INTER_LINEAR);
  
  //I'm not sure if this makes a copy
  inline cv::Mat get_reg_image(){ return reg_image; }
  
  inline bool in_memory(){
    return (raw_buffer != 0);
  }
  
  
private:
  
  std::string filename;
  
  inline void allocate_memory_RAW(){
    if(raw_buffer ==0){
      raw_buffer = new char[31363328];
    }
  }

  inline void free_memory_RAW(){
    delete[] raw_buffer;
    raw_buffer = 0;
  }
  
  unsigned int width, height;
  char *raw_buffer;
  
  cv::Mat reg_image;
  double reg_scale;
  double reg_crop;
  
  
  
};
}


#endif // IMAGE
