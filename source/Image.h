#ifndef IMAGE
#define IMAGE

#include "pathCam.h"

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
  
  void load_raw_from_disk(){
    if(filename != ""){
      std::ifstream stream;
      stream.open(filename, std::ios::binary);
      if(raw_buffer == 0){ allocate_memory_RAW(); }
      stream.read(raw_buffer,width*height);
      stream.close();
    }
  }
  
  inline char * get_Raw(){ return raw_buffer;}
  
  inline std::string get_File(){ return filename;}
  
  void create_reg_image(double reg_scale, double reg_crop, bool convert=true, int interpolation=cv::INTER_LINEAR, bool real=false);
  
  //I'm not sure if this makes a copy
  inline cv::Mat get_reg_image(){ return reg_image; }
  
  inline bool in_memory(){ return (raw_buffer != 0); }
  
  inline double get_reg_scale(){ return reg_scale; }
  
  
private:
  
  std::string filename;
  
  inline void allocate_memory_RAW(){
    if(raw_buffer ==0){
      raw_buffer = new char[width*height];
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
