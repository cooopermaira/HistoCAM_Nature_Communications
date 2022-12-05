//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

int main( int argc, char* argv[] )
{
  
  //Should probably add fancier command line parsing
  if(argc < 3){
    std::cout << "Missing input. Use:\n";
    std::cout << "process_list_truth <path to text file input> <path to text file output\n";
  }
  std::string file = argv[1];
  std::string outfile_name = argv[2];
  
  double crop_factor = 1.0;
  double scale_factor = 1.0;
  bool debayer = true;
  bool real = false;

  int interpolation = cv::INTER_CUBIC;
  
  int features = pathCam::_ORB;

  bool use_FREAK = false;
      
  pathCam::FeatureDetector::ORBParameters ORB_params = pathCam::FeatureDetector::ORBParameters(500, 1.0, 1, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

  cv::DescriptorMatcher::MatcherType matcher_type = cv::DescriptorMatcher::BRUTEFORCE_HAMMING;

  int estimator_type = cv::RANSAC;
  
  
  pathCam::ImageList *pathCam_session =  new pathCam::ImageList();
  
  pathCam_session->loadFileList(file);
  
  std::ofstream outfile;
  outfile.open(outfile_name);
  

  
  for(unsigned int i=0; i < pathCam_session->images.size()-1; i++){
    
    pathCam::Image * image_1 = pathCam_session->images[i];
    pathCam::Image * image_2 = pathCam_session->images[i+1];
    
    pathCam::MotionEstimator *mot = new pathCam::MotionEstimator();
    pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(features, use_FREAK);
    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(matcher_type);


    image_1->load_raw_from_disk();
    image_2->load_raw_from_disk();
    

    if(!image_1->in_memory() || !image_2->in_memory()){
      std::cout << "Issue loading image.\n";
      continue;
    }
    
    image_1->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);
    image_2->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);

    detector->set_ORB_params(ORB_params);
    
    detector->detect_and_compute(image_1);
    detector->detect_and_compute(image_2);
    
    outfile << image_1->get_File() << "\t" << image_2->get_File() << "\t";
    outfile << image_1->keypoints.size() << "\t" <<  image_2->keypoints.size() << "\t";

    
    if(image_1->keypoints.size() < 4 || image_2->keypoints.size() < 4){
      outfile << "failed. Not enough keypoints\n";
      continue;
    }

    pathCam::Match *m = new pathCam::Match(image_1,image_2);
    matcher->match(m);

    
    int result = mot->findHomography(m, estimator_type);
    
    if(result == 1){
      outfile << mot->t_x << "\t" << mot->t_y << "\n";
    }
    if(result == -1){
      outfile << "failed. Not enough matches\n";
    }
    if(result == -2){
        outfile << "failed. Translation not found.\n";
    }
   
    
    image_1->free_memory_RAW();
    delete image_1;
    image_2->free_memory_RAW();
    delete m;
    delete mot;
    delete detector;
    delete matcher;

  }
 
  outfile.close();
  
  delete pathCam_session;
  return 0;
  
}
