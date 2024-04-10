//
//  TiledImage.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/6/24.
//

#pragma once

#include "JuceHeader.h"


namespace pCApp{
  typedef juce::Rectangle<int> Rectangle;
  typedef juce::Point<int> Point;
};

template <typename T>
class Dense2DArray {
public:
  Dense2DArray(int64_t minX=-4000, int64_t maxX=4000, int64_t minY=-4000, int64_t maxY=4000)
  : minX(minX), minY(minY), width(maxX - minX + 1), height(maxY - minY + 1) {
    data.resize(width * height, NULL);
  }
  
  ~Dense2DArray(){
    for(auto i=0; i < data.size(); i++){
      delete data[i];
    }
  }
  
  T& operator()(int64_t x, int64_t y) {
    return data[getIndex(x, y)];
  }
  
  const T& operator()(int64_t x, int64_t y) const {
    return data[getIndex(x, y)];
  }
  
private:
  std::vector<T> data;
  int64_t minX, minY, width, height;
  
  inline int64_t getIndex(int64_t x, int64_t y) const {
    assert(x >= minX && x < minX + width);
    assert(y >= minY && y < minY + height);
    return (x - minX) + (y - minY) * width;
  }
};

struct TileQueryElem{
public:
  juce::Image *image;
  int i, j;
  
  TileQueryElem(juce::Image *image, int i, int j): image(image), i(i), j(j) {};
};

class TiledImage{
private:
  pCApp::Rectangle bounds;
  
  unsigned int tile_size;
  Dense2DArray < juce::Image * > tiles;
  
public:
  TiledImage(unsigned int tile_size=512): tile_size(tile_size){};
  ~TiledImage(){ };
  
  unsigned int getTileSize(){ return tile_size; }
  
  void insertMat(cv::Mat image_in, pCApp::Rectangle i_bounds);
  
  inline juce::Image * getTile(int i, int j){ return tiles(i,j); }
  
  std::vector < TileQueryElem > getTiles(pCApp::Rectangle box);
  
private:
  void matToImage(const cv::Mat& mat, juce::Image *image, pCApp::Point offset,
                  pCApp::Rectangle image_box, pCApp::Rectangle tile_box);
  
  inline pCApp::Point getIJ(pCApp::Point p){
    pCApp::Point ij = pCApp::Point(p.getX()/(int)tile_size, p.getY()/(int)tile_size);
    if(p.getX() < 0){ ij.x--;}
    if(p.getY() < 0){ ij.y--;}
    return ij;
  }
  
  bool tileToDisk(const juce::Image *image, const juce::String& filePath) {
      juce::File file(filePath);
      auto outputStream = file.createOutputStream();
      if (!outputStream) return false;

      juce::PNGImageFormat pngFormat;
      bool success = pngFormat.writeImageToStream(*image, *outputStream);
      return success;
  }
  
  
};

class MRTiledImage{
public:
  MRTiledImage(){};
  ~MRTiledImage(){ level.clear(); };
  
  std::vector < TiledImage > level;
  
  void setLevel(unsigned int l, TiledImage &image){
    level[l] = image;
  }
  
};
