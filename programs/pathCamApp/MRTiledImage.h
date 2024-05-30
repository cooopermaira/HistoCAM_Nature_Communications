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
  
  friend class LoadingThread;
  
public:
  fRectangle bounds;
  unsigned int tile_size;
 
  
  MRTiledImage(unsigned int tile_size=512):tile_size(tile_size){};
  ~MRTiledImage(){ level.clear(); };
    
  void insertMat(cv::Mat &image_in, fRectangle box);
  
  void build(cv::Mat &image_in);
  
  std::vector < TileQuery > getTiles(fRectangle bounds, juce::Rectangle < int > screen);


    std::vector < std::shared_ptr< TiledImage > > level;
private:

    inline fRectangle worldToLevel(fRectangle r, unsigned int level){
    return r/(2*level);
  }
  
  inline fRectangle levelToWorld(fRectangle r, unsigned int level){
    return r*(2*level);
  }
  
  inline fPoint worldToLevel(fPoint p, unsigned int level){
    return p/(2*level);
  }
  
  inline fPoint levelToWorld(fPoint p, unsigned int level){
    return p*(2*level);
  }
  
};



class MRTiledImageSet{
public:
  MRTiledImageSet(){};

  void add(MRTiledImage &image, double scale, fPoint offset){
    images.push_back(image);
    scales.push_back(scale);
    offsets.push_back(offset);
  }

private:
  std::vector < MRTiledImage > images;
  std::vector < double > scales;
  std::vector < fPoint > offsets;
};


#endif /* MRTiledImage_h */
