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
    std::cout << "debayer <path to input image> <path to output image>\n";
  }
  
  std::string file = argv[1];
  std::string outfile_name = argv[2];
  
  pathCam::Image * image = new pathCam::Image();
  
  image->set_disk_file(file);
  image->load_raw_from_disk();
  
  
  if(!image->in_memory() ){
    std::cout << "Issue loading image.\n";
    return -1;
  }
  
  image->create_reg_image(1.0,1.0,true,cv::INTER_CUBIC, false);
  
  imwrite(outfile_name, image->get_reg_image());
  
  delete image;
  return 0;
  
}
