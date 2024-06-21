//
//  Composite.h
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//

#ifndef Composite_h
#define Composite_h

#include <stdio.h>
#include "pathCam.h"
//#include "TiledImage.h"

namespace pathCam {
template <typename T>
struct PointCompare{
  inline bool operator() (const T& p1, const T& p2) {
    if (p1.x == p2.x){
      return p1.y < p2.y;
    }
    return p1.x < p2.x;
  }
};
template <typename T>
struct PointEquality{
  inline bool operator() (const T& p1, const T& p2) {
    return p1.x == p2.x && p1.y == p2.y;
  }
};


class Composite {
  friend class CompositeManager;
  
protected:
  StreamCam *parent;
  Mat local_quality_score, composite_z_buffer, flat_field;
  Mat3f flat_field_composite;
  Mat4b composite;
  Vec2 root_offset, max_offset;
  Bbox subdiv_Bbox;
  cv::Rect_<float> tiledImageBounds;
  
  
public:
  Composite(StreamCam *parent);
  
  void add_images(std::vector<RegInfo> new_info);
  
  void update_Bbox(std::vector<RegInfo> new_info);
  
  void update(std::vector<RegInfo> new_info);
  
  Mat get_composite();
  
  Mat score_image_2X(int, int, int);
  
};


class CompositeVoronoi : public Composite {
private:
  Mat3b threeChannelPreallocated;
  Mat4b fourChannelPreallocated;
  cv::Subdiv2D subdiv;
  std::vector<std::pair<std::string, bool>> memberImages;
  std::vector<Point2i> imageBoundsAsPolygon;
  std::vector<Mat> channels;
  Mat circleMask;
  cv::Size image_size;
  
  long segment_yval_at_point(float xloc, cv::Point2f p1, cv::Point2f p2);
  
  void calculate_effected_tiles(std::vector<Point2i> maskAsPolygon, std::vector<Point2i>& result, Vec2 absCoord);
  static void remove_duplicates_without_sort( std::vector<Point2i>& vec);
  void reset_image_as_polygon();
  void debug_write_contribution_on_grid(std::string name, Vec2 absCoord, Mat &img, Mat &mask);
public:
  CompositeVoronoi(StreamCam *parent, cv::Size image_size);
  
  void update(std::vector<RegInfo> new_info);
  
  void add_images(std::vector<RegInfo> new_info);
  
  void expand_subdiv(std::vector<RegInfo> new_info);
  
};
}

#endif /* Composite_h */
