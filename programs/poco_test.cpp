//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

#include <iostream>

using namespace pathCam;


typedef struct cache_element{
  std::string name;
} cache_element;



int main(int argc, char** argv){
  
  
  Poco::DateTime time = Poco::DateTime();
  
  std::stringstream ss1;
  ss1 << time.year() << time.month();
  ss1 << time.day() << time.hour();
  ss1 << time.minute() << time.millisecond();
  
  std::string captureSetName = ss1.str();
  
  
  std::queue < cache_element > cache;
  Poco::Path root_path = Poco::Path("D:/pcamTest");

  
  time = Poco::DateTime();
  
  std::stringstream ss;
  ss << time.year() << time.month();
  ss << time.day() << time.hour();
  ss << time.minute() << time.millisecond();
  cache_element image_in_cache;
  image_in_cache.name = ss.str();
  

  cache.push(image_in_cache);
  
  cache_element front = cache.front();
  std::string name = front.name + ".raw";
  
  Poco::Path image_path = root_path;
  image_path.append(Poco::Path(captureSetName));
  image_path.append(Poco::Path(name));

  std::cout << image_path.toString(Poco::Path::PATH_WINDOWS) << "\n";
  

//  std::vector < Poco::Runnable * > jobs;
//
//  for (unsigned int i=0; i < 20; i++){
//    jobs.push_back(new HelloRunnable(i));
//  }
//
//  ThreadQueue q(2,5);
//  q.run_jobs(jobs);
  
  
  
  
  return 0;
}
