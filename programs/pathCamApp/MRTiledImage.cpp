//
//  MRTiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/11/24.
//

#include "JuceHeader.h"

std::vector < TileQuery > MRTiledImage::getTiles(fRectangle view, juce::Rectangle <int> screen){
  
  if(level.size() == 0){ return std::vector<TileQuery>(); }
  
  float scale = max(view.getHorizontalRange().getLength()/float(screen.getHorizontalRange().getLength()),
                    view.getVerticalRange().getLength()/float(screen.getVerticalRange().getLength()));  
  scale = log2(scale);
  unsigned int i_scale =  (unsigned int)(scale+0.5);
  i_scale = min((unsigned int)(level.size()-1), i_scale);
  
  return level[i_scale]->getTiles(view);
}

/*
void MRTiledImage::build(cv::Mat &image_in){
  //Determine the number of levels
  unsigned int height = image_in.rows;
  unsigned int width = image_in.cols;
  
  bounds = fRectangle(0,0,width,height);
  
  std::cout << (unsigned int)(max(log2(width),log2(height)) - log2(tile_size) + 2) << "\n";

  unsigned int num_levels = 1;
  std::shared_ptr< TiledImage > current = std::make_shared< TiledImage >(this,tile_size, tile_size,0);
  current->insertMat(image_in, fRectangle(0,0, width, height));
  level.push_back(current);
  
  while(image_in.cols > tile_size || image_in.rows > tile_size){
    std::shared_ptr< TiledImage > current = std::make_shared< TiledImage >(tile_size, tile_size*pow(2,num_levels));
    cv::resize(image_in, image_in, cv::Size(image_in.cols/2, image_in.rows/2));
    current->insertMat(image_in, fRectangle(0,0, width, height));
    level.push_back(current);
    num_levels += 1;
  }
  
  std::cout << "Image has " << num_levels << " levels.";
  
}
*/

void MRTiledImage::insertMat(cv::Mat &image_in, fRectangle box){
    
  bounds = bounds.getUnion(box);

  //This assumes that the # of levels won't change after adding a new image,
  //which isn't going to be necessarily true.
  for(unsigned int i=0; i < level.size(); i++){
    level[i]->insertMat(image_in, box);
    cv::resize(image_in, image_in, cv::Size(image_in.cols/2, image_in.rows/2));
  }
  
}
