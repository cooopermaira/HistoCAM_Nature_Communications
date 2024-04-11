//
//  TiledImage.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/6/24.
//

#pragma once

#include "JuceHeader.h"


namespace pCApp{
  typedef int cType;
  typedef juce::Rectangle<cType> Rectangle;
  typedef juce::Point<cType> Point;
};

template <typename T>
class Dense2DArray {
public:
  Dense2DArray(pCApp::cType minX=-4000, pCApp::cType maxX=4000, pCApp::cType minY=-4000, pCApp::cType maxY=4000)
  : minX(minX), minY(minY), width(maxX - minX + 1), height(maxY - minY + 1) {
    data.resize(width * height, NULL);
  }
  
  ~Dense2DArray(){
    for(auto i=0; i < data.size(); i++){
      delete data[i];
    }
  }
  
  T& operator()(pCApp::cType x, pCApp::cType y) {
    return data[getIndex(x, y)];
  }
  
  const T& operator()(pCApp::cType x, pCApp::cType y) const {
    return data[getIndex(x, y)];
  }
  
private:
  std::vector<T> data;
  pCApp::cType minX, minY, width, height;
  
  inline int64_t getIndex(pCApp::cType x, pCApp::cType y) const {
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
  
  inline juce::Image * getTile(pCApp::cType i, pCApp::cType j){ return tiles(i,j); }
  
  std::vector < TileQueryElem > getTiles(pCApp::Rectangle box);
  
private:
  void matToImage(const cv::Mat& mat, juce::Image *image, pCApp::Point offset,
                  pCApp::Rectangle image_box, pCApp::Rectangle tile_box);
  
  inline pCApp::Point getIJ(pCApp::Point p){
    pCApp::Point ij = pCApp::Point(p.getX()/(pCApp::cType)tile_size, p.getY()/(pCApp::cType)tile_size);
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

