#ifndef IMAGE
#define IMAGE

#include "common.h"


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
  
  void create_reg_image(float reg_scale);
  
  //I'm not sure if this makes a copy
  inline cv::Mat get_reg_image(){ return reg_image; }
  
  
  
private:
  
  std::string filename;
  
  void allocate_memory_RAW();
  void free_memory_RAW();

  
  unsigned int width, height;
  char *raw_buffer;
  
  cv::Mat reg_image;
  float reg_scale;


  
};



#endif // IMAGE
