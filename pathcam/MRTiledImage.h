//
//  MRTiledImage.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/11/24.
//

#ifndef MRTiledImage_h
#define MRTiledImage_h

#include "pathCam.h"
#include "TiledImage.h"

class MRTiledImage{
  
  friend class LoadingThread;
  
public:
  cv::Rect_<float> bounds;
  unsigned int tile_size;
  double scale;
  Point2f offset;
  Poco::Event scaleSet;
 
  
  MRTiledImage(unsigned int tile_size=512):tile_size(tile_size), scaleSet(false){};
  ~MRTiledImage(){ level.clear(); };
    
  void insertMat(cv::Mat &image_in, cv::Rect_<float> box);

  void insertTilesAtBase(cv::Mat image_in, cv::Mat mask, cv::Rect_<float> box, std::vector<Point2i> retileIndices);
  
  void build(cv::Mat &image_in);

  void set_scale(double _scale){scale = _scale;}

  void set_offset(Point2f _offset){offset = _offset;}
  
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
  
  friend class ImageViewComponent;
  friend class CaptureComponent;
  
public:
  cv::Rect_<float> bounds;

  MRTiledImageSet(){};

  void add(std::shared_ptr<MRTiledImage> image){
    images.push_back(image);
  }
  
  bool empty(){ return images.empty(); }
  
  void update_bounds();

private:
  std::vector < std::shared_ptr < MRTiledImage> > images;
};


#endif /* MRTiledImage_h */
