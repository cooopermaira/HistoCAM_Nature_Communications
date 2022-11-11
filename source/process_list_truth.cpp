//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "common.h"

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
  
  int features = pathCam::_SIFT;

  bool use_FREAK = false;
      
  pathCam::FeatureDetector::SIFTParameters SIFT_params;
  
  cv::DescriptorMatcher::MatcherType matcher_type = cv::DescriptorMatcher::BRUTEFORCE;
  
  int estimator_type = cv::RANSAC;
  
  
  pathCam::ImageList *pathCam_session =  new pathCam::ImageList();
  
  pathCam_session->loadFileList(file);
  
  std::ofstream outfile;
  outfile.open(outfile_name);
  
  for(unsigned int i=0; i < pathCam_session->images.size()-1; i++){
    
    pathCam::Image * image_1 = pathCam_session->images[i];
    pathCam::Image * image_2 = pathCam_session->images[i+1];

    image_1->load_raw_from_disk();
    image_2->load_raw_from_disk();

    if(!image_1->in_memory() || !image_2->in_memory()){
      std::cout << "Issue loading image.\n";
      continue;
    }
    
    image_1->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);
    image_2->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);

    pathCam::MotionEstimator *mot = new pathCam::MotionEstimator();
    pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(features, use_FREAK);

    detector->set_SIFT_params(SIFT_params);
    
    detector->detect_and_compute(image_1);
    detector->detect_and_compute(image_2);

    pathCam::Match *m = new pathCam::Match(image_1,image_2);
    pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(matcher_type);
    matcher->match(m);

    mot->findHomography(m, estimator_type);
   
    outfile << image_1->get_File() << "\t" << image_2->get_File() << "\t";
    outfile << mot->t_x << "\t" << mot->t_y << "\n";
  }
 
  outfile.close();
  
  delete pathCam_session;
  return 0;
  
}
