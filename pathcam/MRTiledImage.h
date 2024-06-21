//
//  MRTiledImage.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/11/24.
//

#ifndef MRTiledImage_h
#define MRTiledImage_h

#include "pathCam.h"

class MRTiledImage{
  
  friend class LoadingThread;
  
public:
  cv::Rect_<float> bounds;
  unsigned int tile_size;
 
  
  MRTiledImage(unsigned int tile_size=512):tile_size(tile_size){};
  ~MRTiledImage(){ level.clear(); };
    
  void insertMat(cv::Mat &image_in, cv::Rect_<float> box);
  
  void build(cv::Mat &image_in);
  
  std::vector < TileQuery > getTiles(cv::Rect_<float> bounds, cv::Rect_<int> screen);

  std::vector < std::shared_ptr< TiledImage > > level;

private:

  inline cv::Rect_<float> worldToLevel(cv::Rect_<float> r, unsigned int level){
    float denom = (2.0f*level);
    return cv::Rect_<float>(r.x/denom, r.y/denom, r.width/denom, r.height/denom);
  }
  
  inline cv::Rect_<float> levelToWorld(cv::Rect_<float> r, unsigned int level){
    float denom = (2.0f*level);
    return cv::Rect_<float>(r.x*denom, r.y*denom, r.width*denom, r.height*denom);
  }
  
  inline Point2f worldToLevel(Point2f p, unsigned int level){
    return p/(2.0f*level);
  }
  
  inline Point2f levelToWorld(Point2f p, unsigned int level){
    return p*(2.0f*level);
  }
  
};



class MRTiledImageSet{
public:
  MRTiledImageSet(){};

  void add(MRTiledImage &image, double scale, Point2f offset){
    images.push_back(image);
    scales.push_back(scale);
    offsets.push_back(offset);
  }

private:
  std::vector < MRTiledImage > images;
  std::vector < double > scales;
  std::vector < Point2f > offsets;
};


#endif /* MRTiledImage_h */
