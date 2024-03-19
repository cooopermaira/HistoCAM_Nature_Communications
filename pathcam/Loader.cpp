//
//  Loader.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam{


Loader::Loader(StreamCam *parent, JobQueue *queue): parent(parent), queue(queue), successful(false){};


void Loader::run(){
    int j = 0;
  while(!parent->disk_empty || !parent->buffer.empty()){
    if(parent->buffer.empty()){
      Poco::Thread::sleep(100);
    }else{
      pathCam::Image *image = new pathCam::Image;
      
      parent->buffer_mutex->lock();
      image->set_disk_file(parent->disk_image.front());
      parent->disk_image.pop();
      
      image->copy_in(parent->buffer.front());
      delete [] parent->buffer.front();
      parent->buffer.pop();
      parent->buffer_mutex->unlock();
      j++;
      if(!image->in_memory()){
        successful = false;
        image->label = Image::_BAD_FILE;
        return;
      }
      
      image->find_label();

      if(image->is_good()){
        image->create_reg_image(parent->scale_factor,parent->crop_factor,parent->debayer,parent->interpolation, parent->real);
        image->free_memory_RAW();

        pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(parent->feature_type, parent->use_FREAK);
        
        switch(parent->feature_type){
          case _SIFT:
            detector->set_SIFT_params(parent->SIFT_params);
            break;
          case _SURF:
            detector->set_SURF_params(parent->SURF_params);
            break;
          case _AKAZE:
            detector->set_AKAZE_params(parent->AKAZE_params);
            break;
          case _BRISK:
            detector->set_BRISK_params(parent->BRISK_params);
            break;
          case _ORB:
            detector->set_ORB_params(parent->ORB_params);
            break;
        }
        
        detector->detect_and_compute(image);
        
        if(image->keypoints.size() < 200){
          detector->set_ORB_params();
          detector->detect_and_compute(image);
        }
        
        delete detector;
        
        if(image->keypoints.size() < 200){
          successful = false;
          image->label = Image::_LOWFEAT;
          return;
        }
        
        image_index = parent->add_image(image);

        //since we dont know the number of frames ahead of time, these 3 vectors have to be dynamically resized. To avoid doing this constantly, it's done 100 slots at a time. This is probably not thread safe and likely needs a mutex since the resize might cause a reallocation. If it doesn't cause a reallocation, it's fine since its only adding onto the end of the vector.
        if(image_index % 100 == 0){
          parent->matchM.resize(image_index + 100);
          parent->reg_results.resize(image_index + 100, RegInfo());
          parent->visited.resize(image_index + 100, false);
        }
        
        MatchRunnable *matchjob = new MatchRunnable(parent,image_index);
        queue->add_runnable(matchjob);
        
      }
      else {
          image->free_memory_RAW();
      }
    }//end if
    
  }//end while
  successful = true;
  parent->jobs_queued = true; //termination condition for other processes
  std::cout << j << std::endl;
  std::cout << "jobs queued ";
}//end run


SpinLoader::SpinLoader(StreamCam* parent, JobQueue* queue) : parent(parent), queue(queue), successful(false) {};


void SpinLoader::run() {

    while (parent->microscope_input || !parent->spin_image_buffer.empty()) {
        if (parent->spin_image_buffer.empty()) {
            Poco::Thread::sleep(100);
        }
        else {
            pathCam::Image* image = parent->get_Q_front_Spin();

            if (!image->in_memory()) {
                successful = false;
                image->label = Image::_BAD_FILE;
                std::cout << "Image failed to load" << std::endl;
                return;
            }

            image->find_label();

            if (image->is_good()) {
                image->create_reg_image(parent->scale_factor, parent->crop_factor, parent->debayer, parent->interpolation, parent->real);
                image->free_memory_RAW();

                pathCam::FeatureDetector* detector = new pathCam::FeatureDetector(parent->feature_type, parent->use_FREAK);

                switch (parent->feature_type) {
                case _SIFT:
                    detector->set_SIFT_params(parent->SIFT_params);
                    break;
                case _SURF:
                    detector->set_SURF_params(parent->SURF_params);
                    break;
                case _AKAZE:
                    detector->set_AKAZE_params(parent->AKAZE_params);
                    break;
                case _BRISK:
                    detector->set_BRISK_params(parent->BRISK_params);
                    break;
                case _ORB:
                    detector->set_ORB_params(parent->ORB_params);
                    break;
                }

                detector->detect_and_compute(image);

                if (image->keypoints.size() < 200) {
                    detector->set_ORB_params();
                    detector->detect_and_compute(image);
                }

                delete detector;

                if (image->keypoints.size() < 200) {
                    successful = false;
                    image->label = Image::_LOWFEAT;
                    return;
                }

                image_index = parent->add_image(image);

                //since we dont know the number of frames ahead of time, these 3 vectors have to be dynamically resized. To avoid doing this constantly, it's done 100 slots at a time. This is probably not thread safe and likely needs a mutex since the resize might cause a reallocation. If it doesn't cause a reallocation, it's fine since its only adding onto the end of the vector.
                if (image_index % 100 == 0) {
                    parent->matchM.resize(image_index + 100);
                    parent->reg_results.resize(image_index + 100, RegInfo());
                    parent->visited.resize(image_index + 100, false);
                }

                MatchRunnable* matchjob = new MatchRunnable(parent, image_index);
                queue->add_runnable(matchjob);

            }
            else {
                image->free_memory_RAW();
            }
        }//end if

    }//end while
    successful = true;
    parent->jobs_queued = true; //termination condition for other processes
    std::cout << "jobs queued ";
}//end run


}//end namespace
