#ifndef IMAGE
#define IMAGE

#include "pathCam.h"

namespace pathCam{

using Poco::MemoryPool;

class Image{
public:
  unsigned int width, height;

  
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  
  
  Image(MemoryPool *mempool=0);
  ~Image();
  
  void set_memory_pool(MemoryPool *mempool_in){
    mempool = mempool_in;
  }
  
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
  
  inline void free_memory_RAW(){
    if(raw_buffer != 0){
      if(mempool){
        mempool->release(raw_buffer);
      }else{
        delete[] raw_buffer;
      }
    }
    raw_buffer = 0;
  }
  
  
private:
  
  std::string filename;
  MemoryPool *mempool;
  
  inline void allocate_memory_RAW(){
    if(raw_buffer ==0){
      if(mempool){raw_buffer = reinterpret_cast<char*>(mempool->get());
      }else{ raw_buffer = new char[width*height]; }
    }
  }
  
  char *raw_buffer;
  
  cv::Mat reg_image;
  double reg_scale;
  double reg_crop;
  
  
  
};
}


#endif // IMAGE
