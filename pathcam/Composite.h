//
//  Composite.h
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//

#ifndef Composite_h
#define Composite_h

#include <stdio.h>
#include <random>
#include "pathCam.h"
//#include "TiledImage.h"

namespace pathCam {


  template<typename T>
  struct PointCompare {
    inline bool operator()(const T &p1, const T &p2) {
      if (p1.x == p2.x) {
        return p1.y < p2.y;
      }
      return p1.x < p2.x;
    }
  };

  template<typename T>
  struct PointEquality {
    inline bool operator()(const T &p1, const T &p2) {
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
    Rect_<float> tiledImageBounds;
    Poco::FastMutex *update_mutex;


  public:

    std::shared_ptr< MRTiledImage >  imagePyramid;

    Composite(StreamCam *parent);

    void add_images(std::vector<RegInfo> new_info);

    void update_Bbox(std::vector<RegInfo> new_info);

    void update(std::vector<RegInfo> new_info);

    Mat get_composite();

    Mat score_image_2X(int, int, int);
    

  };


  class CompositeVoronoi : public Composite {
    friend class ImageToTileCopyRunnable;
    friend class CompositeManager;
  private:
    Mat circleMask;
    Mat polyMaskOutput;
    Mat freshMask;
    Mat3b threeChannelPreallocated;
    Mat4b fourChannelPreallocated;
    Subdiv2D subdiv;
    cv::Size image_size;
    std::vector<Mat> channels;
    std::vector<Point2i> imageBoundsAsPolygon;
    std::atomic<unsigned int> jobCount = 0;
    Bbox subdiv_Bbox;
    Poco::Event wakeEvent;


    long segment_yval_at_point(float xloc, cv::Point2f p1, cv::Point2f p2);

    void add_images_with_composite(std::vector<RegInfo> new_info);

    void add_images_no_composite(std::vector<RegInfo> new_info);

    void add_images_multithread(std::vector<RegInfo> new_info);

    void calculate_effected_tiles(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result, Vec2 absCoord);

    static void remove_duplicates_without_sort(std::vector<Point2i> &vec);

    void reset_image_as_polygon();

    void save_pyramid_as_image();

    void debug_write_contribution_on_grid(std::string name, Vec2 absCoord, Mat &img, Mat &mask);

    int add_point_to_delaunay_triangulation(cv::Point2f _point, pathCam::Image *_image,
                                            std::vector<Point2i> &_face);

    void expand_subdiv(std::vector<RegInfo> new_info);

    void self_reset();
    
    double coopers_conjugate_gradient(cv::Mat A, cv::Mat b, cv::Mat x, int steps, double epsilon);


  public:

    CompositeVoronoi(StreamCam *parent, cv::Size image_size, unsigned int componentIndex);

    unsigned int componentIndex;
    std::atomic<unsigned int> matchableCount = 0;
    std::vector<std::pair<long, long>> matchedEdges;
    std::vector<std::pair<Image*, bool>> memberImages;
    std::map<int, unsigned long> delaunayMembers;

    void update(std::vector<RegInfo> new_info);

    void update_Bbox_no_composite(std::vector<RegInfo> new_info);

    void perform_global_alignment(unsigned int flag, double closenessFactor);

  protected:
    std::priority_queue<unsigned int> freeMasks;
    std::vector<Mat> masks;
    std::vector<Mat> threeChanPreals;
    std::vector<Mat> fourChanPreals;
    void notify_job_complete();
  };
}

#endif /* Composite_h */
