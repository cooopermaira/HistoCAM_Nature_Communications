//
//  MRTiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/11/24.
//

#include "pathCam.h"

std::vector < TileQuery > MRTiledImage::getTiles(cv::Rect_<float> view, cv::Rect_<int> screen, bool pullFromBase){
  if (pullFromBase) {
    return level[0]->getTiles(view);
  }

  if(level.size() == 0){ return std::vector<TileQuery>(); }
  float scale = max(view.width/float(screen.width),
                    view.height/float(screen.height));

  scale = log2(scale);
  unsigned int i_scale =  (unsigned int)(scale+0.5);
  i_scale = min((unsigned int)(level.size()-1), i_scale);
  return level[i_scale]->getTiles(view);
}

MRTiledImage::MRTiledImage(pathCam::StreamCam *parent, unsigned int _tile_size) : parent(parent),tile_size(_tile_size){
  if(tile_size == 0){
    tile_size = parent->tileSize;
  }
}

int MRTiledImage::get_class_for_tile(std::tuple<int,int,unsigned> _tile) {
  if (parent) {
    if (parent->tileCoordToClass.find(_tile) != parent->tileCoordToClass.end()) {
      return parent->tileCoordToClass[_tile];
    }
  }
  return -1;
}


void MRTiledImage::insertTilesAtBase(cv::Mat image_in, cv::Mat mask, cv::Rect_<float> box,
                                    std::vector<Point2i> retileIndices) {
  level[0]->insertTilesAtBase(image_in,mask,box,retileIndices);
}

void MRTiledImage::insertMat(cv::Mat &image_in, cv::Rect_<float> box){
    
  bounds = bounds | box;

  //This assumes that the # of levels won't change after adding a new image,
  //which isn't going to be necessarily true.
  for(unsigned int i=0; i < level.size(); i++){
    level[i]->insertMat(image_in, box);
    cv::resize(image_in, image_in, cv::Size(image_in.cols/2, image_in.rows/2));
  }
  
}

void MRTiledImageSet::update_bounds() {
  auto minX = bounds.x;
  auto minY = bounds.y;
  auto maxX = minX+bounds.width;
  auto maxY = minY+bounds.height;
  for (const auto & image : images){
    auto imageMinX = image->bounds.x + image->offset.x * image->scale;
    auto imageMinY = image->bounds.y + image->offset.y * image->scale;
    minX = fmin(minX,imageMinX);
    minY = fmin(minY,imageMinY);

    auto imageMaxX = imageMinX + image->scale * image->bounds.width;
    auto imageMaxY = imageMinY + image->scale * image->bounds.height;
    maxX = max(double(maxX),imageMaxX);
    maxY = max(double(maxY),imageMaxY);
  }
  bounds.x = minX;
  bounds.y = minY;
  bounds.width = maxX - minX;
  bounds.height = maxY - minY;
}
