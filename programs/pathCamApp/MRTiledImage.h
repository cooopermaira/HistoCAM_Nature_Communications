//
//  MRTiledImage.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/11/24.
//

#ifndef MRTiledImage_h
#define MRTiledImage_h

#include "JuceHeader.h"

class MRTiledImage{
public:
  pCApp::Rectangle bounds;
  unsigned int tile_size;
 
  
  MRTiledImage(unsigned int tile_size=512):tile_size(tile_size){};
  ~MRTiledImage(){ level.clear(); };
  
  std::vector < std::shared_ptr< TiledImage > > level;
  
  void build(cv::Mat image_in);
  
  std::vector < TileQueryElem > getTiles(pCApp::Rectangle box);
  
  
private:
  
  inline pCApp::Rectangle worldToLevel(pCApp::Rectangle r, unsigned int level){
    return r/(2*(int)level);
  }
  
  inline pCApp::Rectangle levelToWorld(pCApp::Rectangle r, unsigned int level){
    return r*(2*(int)level);
  }
  
  inline pCApp::Point worldToLevel(pCApp::Point p, unsigned int level){
    return p/(2*(int)level);
  }
  
  inline pCApp::Point levelToWorld(pCApp::Point p, unsigned int level){
    return p*(2*(int)level);
  }
  
};



class MRTiledImageSet{
public:
  MRTiledImageSet(){};

  void add(MRTiledImage &image, double scale, pCApp::Point offset){
    images.push_back(image);
    scales.push_back(scale);
    offsets.push_back(offset);
  }

private:
  std::vector < MRTiledImage > images;
  std::vector < double > scales;
  std::vector < pCApp::Point > offsets;
};


#endif /* MRTiledImage_h */
