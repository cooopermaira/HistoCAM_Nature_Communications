//
//  ImageList.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

namespace pathCam{

  bool ImageList::loadFileList(std::string filelist, unsigned int image_width, unsigned int image_height){
    
    std::ifstream infile(filelist.c_str());
    
    if(!infile.good()){
      std::cout << "Unable to open file\n";
      return false;
    }
    
    std::string imageFile;
    
    while (infile >>imageFile){
      if(imageFile.size() == 0){ continue;}
      pathCam::Image *image = new pathCam::Image(image_width, image_height);
      image->set_disk_file(imageFile);
      images.push_back(image);
    }
    
    if(images.size() == 0){
      std::cout << "No images loaded.\n";
      return false;
    }
    
    std::cout << images.size() << " images loaded.\n";
    infile.close();
    
    return true;
  }

}
