//
//  TiledImage.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/6/24.
//

#pragma once

#include "JuceHeader.h"

typedef juce::Rectangle< float > fRectangle;
typedef juce::Point< int > iPoint;
typedef juce::Point< float > fPoint;


template <typename T>
class Dense2DArray {
public:
  Dense2DArray(int minX=-4000, int maxX=4000, int minY=-4000, int maxY=4000)
  : minX(minX), minY(minY), width(maxX - minX + 1), height(maxY - minY + 1) {
    data.resize(width * height, NULL);
  }
  
  ~Dense2DArray(){
    for(auto i=0; i < data.size(); i++){
      delete data[i];
    }
  }
  
  T& operator()(int x, int y) {
    return data[getIndex(x, y)];
  }
  
  const T& operator()(int x, int y) const {
    return data[getIndex(x, y)];
  }
  
private:
  std::vector<T> data;
  int minX, minY, width, height;
  
  inline int64_t getIndex(int x, int y) const {
    assert(x >= minX && x < minX + width);
    assert(y >= minY && y < minY + height);
    return (x - minX) + (y - minY) * width;
  }
};

struct TileQuery{
public:
  juce::Image *image;
  int i, j;
  fRectangle bounds;
  TileQuery(juce::Image *image, int i, int j, fRectangle bounds): 
  image(image), i(i), j(j), bounds(bounds){};
};

class TiledImage{
private:
  fRectangle bounds;
  
  unsigned int tile_size;
  unsigned int logic_size;

  Dense2DArray < juce::Image * > tiles;
  
public:
  TiledImage(unsigned int tile_size=512, unsigned int logic_size=512):
  tile_size(tile_size), logic_size(logic_size){};
  ~TiledImage(){ };
  
  unsigned int getTileSize(){ return tile_size; }
  
  void insertMat(cv::Mat image_in, fRectangle i_bounds);
  
  inline juce::Image * getTile(int i, int j){ return tiles(i,j); }
  
  std::vector < TileQuery > getTiles(fRectangle box);
  
private:
  void matToImage(const cv::Mat& mat, juce::Image *image, fPoint offset,
                  fRectangle image_box, fRectangle tile_box);
  
  inline iPoint getIJ(fPoint p){
    iPoint ij = iPoint(p.getX()/(int)logic_size, p.getY()/(int)logic_size);
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

