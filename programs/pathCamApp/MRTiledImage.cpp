//
//  MRTiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/11/24.
//

#include "JuceHeader.h"

std::vector < TileQuery > MRTiledImage::getTiles(fRectangle bounds){
  
  //box = worldToLevel(box, 1);
  
  
  return level[1]->getTiles(bounds);
}


void MRTiledImage::build(cv::Mat image_in){
  //Determine the number of levels
  unsigned int height = image_in.rows;
  unsigned int width = image_in.cols;
  
  unsigned int num_levels = 0;
  
  while(image_in.cols > tile_size && image_in.rows > tile_size){
    std::shared_ptr< TiledImage > current = std::make_shared< TiledImage >(tile_size);
    current->insertMat(image_in, fRectangle(0,0, image_in.cols, image_in.rows));
    level.push_back(current);
    
    cv::resize(image_in, image_in, cv::Size(image_in.cols/2, image_in.rows/2));
    num_levels += 1;
  }
  
  std::cout << "Image has " << num_levels << " levels.";
  
}
