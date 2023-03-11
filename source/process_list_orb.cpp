//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"


class Vec2{
public:
  double x, y;
  Vec2(double x, double y): x(x), y(y){};
};

class Bbox{
public:
  double min_x, min_y, max_x, max_y;
  Bbox(double min_x=std::numeric_limits<double>::infinity(),
       double min_y=std::numeric_limits<double>::infinity(),
       double max_x=-std::numeric_limits<double>::infinity(),
       double max_y=-std::numeric_limits<double>::infinity()):
  min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y) {};
};

class RegInfo{
public:
  bool successful;
  Bbox bbox;
  Vec2 vec;
  RegInfo(bool successful=false, Vec2 vec=Vec2(0.0, 0.0), Bbox bbox = Bbox()):
  successful(successful), vec(vec), bbox(bbox) {};
};

int main( int argc, char* argv[] )
{
  
  //Should probably add fancier command line parsing
  if(argc < 3){
    std::cout << "Missing input. Use:\n";
    std::cout << "process_list_truth <path to text file input> <path to text file output> OPTIONAL[<path to output image>]\n";
  }

  std::string file = argv[1];
  std::string outfile_name = argv[2];
  
  std::string outimage_name = "";
  
  if(argc == 4){
    outimage_name = argv[3];
    std::cout << "Composited image will be written to " << outimage_name << "\n";
  }
  
  double crop_factor = 0.5;
  double scale_factor = 0.25;
  bool debayer = true;
  bool real = false;
  
  int interpolation = cv::INTER_CUBIC;
  
  int features = pathCam::_ORB;
  
  bool use_FREAK = false;
  
  std::vector < RegInfo > reg_results;
  
  pathCam::FeatureDetector::ORBParameters ORB_params = pathCam::FeatureDetector::ORBParameters(500, 1.0, 1, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
  
  cv::DescriptorMatcher::MatcherType matcher_type = cv::DescriptorMatcher::BRUTEFORCE_HAMMING;
  
  int estimator_type = cv::RANSAC;
  
  
  pathCam::ImageList *pathCam_session =  new pathCam::ImageList();
  
  pathCam_session->loadFileList(file);
  
  std::ofstream outfile;
  outfile.open(outfile_name);
  
  pathCam::MotionEstimator *mot = new pathCam::MotionEstimator();
  pathCam::FeatureDetector *detector = new pathCam::FeatureDetector(features, use_FREAK);
  pathCam::DescriptorMatcher *matcher = new pathCam::DescriptorMatcher(matcher_type);
  
  auto reg_begin = std::chrono::high_resolution_clock::now();

  std::cout << "Performing Registration:\n";
  
  reg_results.resize(pathCam_session->images.size());
  unsigned int last_index = 0;
  
  for(unsigned int i=0; i < pathCam_session->images.size()-1; i++){
    
    if(i==0){
        Bbox box = Bbox(0, 0, pathCam_session->images[0]->width, pathCam_session->images[0]->height);
        reg_results[0] = RegInfo(true, Vec2(0, 0), box);
    }
      
    pathCam::Image * last_registered = pathCam_session->images[last_index];
    pathCam::Image * next_image = pathCam_session->images[i+1];

    last_registered->load_raw_from_disk();
    next_image->load_raw_from_disk();
    
    if(!last_registered->in_memory() || !next_image->in_memory()){
      std::cout << "Issue loading image.\n";
      continue;
    }
    
    outfile << last_registered->get_File() << "\t";
    outfile << next_image->get_File() << "\t";
    
    auto begin = std::chrono::high_resolution_clock::now();
    
    last_registered->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);
    next_image->create_reg_image(scale_factor,crop_factor,debayer,interpolation, real);
    
    detector->set_ORB_params(ORB_params);
    
    detector->detect_and_compute(last_registered);
    detector->detect_and_compute(next_image);
    
    if(last_registered->keypoints.size() < 200 || next_image->keypoints.size() < 200){
      std::cout << "Too little features detected.  Going back to defaults\n";
      delete detector;
      detector = new pathCam::FeatureDetector(features, use_FREAK);
      detector->detect_and_compute(last_registered);
      detector->detect_and_compute(next_image);
    }
    
    if(last_registered->keypoints.size() < 100 || next_image->keypoints.size() < 100){
      outfile << "failed. Not enough keypoints\n";
      reg_results[i+1] = RegInfo(false, reg_results[i].vec);
      continue;
    }
    
    pathCam::Match *m = new pathCam::Match(last_registered,next_image);
    matcher->match(m);
    
    int result = mot->findHomography(m, estimator_type);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    outfile << last_registered->keypoints.size() << "\t";
    outfile << next_image->keypoints.size() << "\t";
    
    
    if(result == 1){
      outfile << m->t_x << "\t" << m->t_y << "\t";
      outfile << elapsed.count() * 1e-9 << "\n";
      double t_x = reg_results[last_index].vec.x-m->t_x;
      double t_y = reg_results[last_index].vec.y-m->t_y;
      Bbox box = Bbox(t_x, t_y, next_image->width+t_x, next_image->height+t_y);
      reg_results[i+1] = RegInfo(true, Vec2(t_x, t_y), box);
      last_index = i+1;
    }
    if(result == -1){
      outfile << "failed. Not enough matches\n";
      reg_results[i+1] = RegInfo(false, reg_results[i].vec);
    }
    if(result == -2){
      outfile << "failed. Translation not found.\n";
      reg_results[i+1] = RegInfo(false, reg_results[i].vec);
    }
    
    //Need to only unload if not using again, but doing this to make sure
    //initial program has no memory leaks
    
    last_registered->free_memory_RAW();
    next_image->free_memory_RAW();
      
    delete m;
  }
    
  std::cout << "Done Registering Images\n";
    
  auto reg_end = std::chrono::high_resolution_clock::now();
  auto reg_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(reg_end - reg_begin);
  std::cout << reg_elapsed.count() * 1e-9 << " seconds including I/O\n";

  
  if(outimage_name == ""){
    delete pathCam_session;
    return 0;
  }
  
  auto comp_begin = std::chrono::high_resolution_clock::now();

  std::cout << "Compositing Images:\n";
  
  Bbox combined_box = Bbox();
  
  for(unsigned int i=0; i < reg_results.size(); i++){
    if(reg_results[i].successful){
      if(reg_results[i].bbox.min_x <  combined_box.min_x){
        combined_box.min_x = reg_results[i].bbox.min_x;
      }
      if(reg_results[i].bbox.min_y <  combined_box.min_y){
        combined_box.min_y = reg_results[i].bbox.min_y;
      }
      if(reg_results[i].bbox.max_x >  combined_box.max_x){
        combined_box.max_x = reg_results[i].bbox.max_x;
      }
      if(reg_results[i].bbox.max_y >  combined_box.max_y){
        combined_box.max_y = reg_results[i].bbox.max_y;
      }
    }
  }
  
  
//  std::cout << "combined bbox: ";
//  std::cout << combined_box.min_x << "\t" << combined_box.min_y << "\t";
//  std::cout << combined_box.max_x << "\t" << combined_box.max_y << "\n";
  
  
  if(combined_box.min_x < 0.0){
    for(unsigned int i=0; i < reg_results.size(); i++){
      if(reg_results[i].successful){
        reg_results[i].bbox.min_x -= combined_box.min_x;
        reg_results[i].bbox.max_x -= combined_box.max_x;
      }
    }
    combined_box.max_x -= combined_box.min_x;
    combined_box.min_x -= combined_box.min_x;
  }
  
  if(combined_box.min_y < 0.0){
    for(unsigned int i=0; i < reg_results.size(); i++){
      if(reg_results[i].successful){
        reg_results[i].bbox.min_y -= combined_box.min_y;
        reg_results[i].bbox.max_y -= combined_box.max_y;
      }
    }
    combined_box.max_y -= combined_box.min_y;
    combined_box.min_y -= combined_box.min_y;
  }
  
//  std::cout << "combined bbox: ";
//  std::cout << combined_box.min_x << "\t" << combined_box.min_y << "\t";
//  std::cout << combined_box.max_x << "\t" << combined_box.max_y << "\n";
//  
//  std::cout << "width height: ";
//  std::cout << "[ " << combined_box.max_x-combined_box.min_x << ", ";
//  std::cout << combined_box.max_y-combined_box.min_y << "]\n";
  
  
  Mat3b combined(combined_box.max_y-combined_box.min_y,combined_box.max_x-combined_box.min_x, Vec3b(0,0,0));
  std::cout << combined.size()  << "\n";
  
  for(unsigned int i=0; i < pathCam_session->images.size(); i++){
    if(reg_results[i].successful){
      pathCam::Image * temp = pathCam_session->images[i];
      temp->load_raw_from_disk();
      
      Mat image_Mat = cv::Mat(Size(temp->width,temp->height), CV_8UC1, temp->get_Raw(), Mat::AUTO_STEP);
      cvtColor(image_Mat,image_Mat,COLOR_BayerBG2BGR);
      
      image_Mat.copyTo(combined(Rect(reg_results[i].bbox.min_x, reg_results[i].bbox.min_y,                                                     image_Mat.cols, image_Mat.rows)));
      
      temp->free_memory_RAW();
    }
  }
  
  imwrite(outimage_name, combined);
  std::cout << "Done Compositing Images\n";
    
  auto comp_end = std::chrono::high_resolution_clock::now();
  auto comp_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - comp_begin);
  std::cout << comp_elapsed.count() * 1e-9 << " seconds including I/O\n";

  auto total_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - reg_begin);
  std::cout << total_elapsed.count() * 1e-9 << " seconds total time including I/O\n";

    
  delete pathCam_session;
  return 0;
  
}
