//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

#include "Poco/Thread.h"
#include "Poco/Runnable.h"
#include <iostream>

using namespace pathCam;


class HelloRunnable: public Poco::Runnable{
private:
  char * ground_buffer;
  cv::Mat ground_reg;
  Image * image;

public:
  
  HelloRunnable(Image * image, char * ground_buffer, cv::Mat ground_reg): image(image), ground_buffer(ground_buffer), ground_reg(ground_reg){};

  virtual void run(){
    int id = Poco::Thread::current()->id();
    srand(Poco::Thread::current()->id());
    Poco::Thread::sleep(int(float(rand())/RAND_MAX *1000));
    image->load_raw_from_disk();
    image->create_reg_image(1.0, 1.0);
    assert(image->get_Raw() != 0);
    assert(memcmp(ground_buffer, image->get_Raw(), image->width*image->height) == 0);
    cv::Mat diff = ground_reg != image->get_reg_image();
    // Equal if no elements disagree
    assert(cv::countNonZero(diff) == 0);
    Poco::Thread::sleep(int(float(rand())/RAND_MAX *1000));
    image->free_memory_RAW();
  }
    
};


int main(int argc, char** argv){
  
  std::string test_file = "../../resources/temp-07282022114119-1244.Raw";
  
  Image *ground = new Image();
  ground->set_disk_file(test_file);
  ground->load_raw_from_disk();
  ground->create_reg_image(1.0, 1.0);
  
  char *raw_buffer_copy = new char[ground->width*ground->height];
  ground->get_Raw();
  memcpy(raw_buffer_copy, ground->get_Raw(), ground->width*ground->height);
  cv::Mat reg_copy = ground->get_reg_image().clone();
  
  ground->load_raw_from_disk();

  ground->load_raw_from_disk();

  ground->free_memory_RAW();
  assert(ground->get_Raw() != 0);
  
  ground->free_memory_RAW();
  assert(ground->get_Raw() != 0);

  ground->free_memory_RAW();
  assert(ground->get_Raw() == 0);
  
  delete ground;
  
  
  {
    Image * test_image = new Image();
    test_image->set_disk_file(test_file);
    
    unsigned int num_threads = 20;
    
    std::vector < HelloRunnable > runnable(num_threads, HelloRunnable(test_image,raw_buffer_copy,reg_copy));
    std::vector < Poco::Thread > thread(num_threads);
    
    for(unsigned int i=0; i < num_threads; i++){
      Poco::Thread::sleep( int((float)rand()/RAND_MAX * 100));
      thread[i].start(runnable[i]);
    }
    
    for(unsigned int i=0; i < num_threads; i++){
      thread[i].join();
    }
    
    assert(test_image->get_Raw() == 0);
    
    delete test_image;
  }
  
  {
    Poco::MemoryPool * mempool =  new MemoryPool(6464*4852);
    Image * test_image = new Image(mempool);
    test_image->set_disk_file(test_file);
    
    unsigned int num_threads = 20;
    
    std::vector < HelloRunnable > runnable(num_threads, HelloRunnable(test_image,raw_buffer_copy,reg_copy));
    std::vector < Poco::Thread > thread(num_threads);
    
    for(unsigned int i=0; i < num_threads; i++){
      Poco::Thread::sleep( int((float)rand()/RAND_MAX * 100));
      thread[i].start(runnable[i]);
    }
    
    for(unsigned int i=0; i < num_threads; i++){
      thread[i].join();
    }
    
    assert(test_image->get_Raw() == 0);
    
    delete test_image;
  }

  
  
  
  return 0;
}
