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
#include <numeric>
#include "pathCam.h"


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
    friend class RebuildRunnable;

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

    void add_images(std::vector<RegInfo*> new_info);

    void update_Bbox(std::vector<RegInfo*> new_info);

    void update(std::vector<RegInfo*> new_info);

    Mat get_composite();

    Mat score_image_2X(int, int, int);
    

  };


  class CompositeVoronoi : public Composite {
    friend class ImageToTileCopyRunnable;
    friend class CompositeManager;
    friend class RebuildRunnable;
  private:
    unsigned long lastAcceptedImageIndex;
    unsigned int minPixelDistanceBetweenFrames;
    Vec2 lastAcceptedImageAbC;
    Vec2 lastImageAbC;
    Mat circleMask;
    Mat rectMask;
    Mat polyMaskOutput;
    Mat freshMask;
    Mat ff;
    Mat convertHolding;
    Mat threeChannelPreallocated;
    Mat fourChannelPreallocated;
#ifdef HAVE_OPENCV_CUDAARITHM
    cuda::GpuMat diffGPU;
    cuda::GpuMat xp1;
    cuda::GpuMat xp2;
    cuda::GpuMat binaryCompare;

    long long bigx = 0,bigy = 0;
    cuda::GpuMat meshGridX;
    cuda::GpuMat meshGridY;
    cuda::GpuMat rectMaskGPU;
    cuda::GpuMat circleMaskGPU;
    cuda::GpuMat polyMaskGPU;
    std::vector<cuda::GpuMat> channelsGPU;
    cuda::GpuMat ffGPU;
    cuda::GpuMat convertHoldingGPU;
    cuda::GpuMat threeChannelPrealGPU;
    cuda::GpuMat fourChannelPrealGPU;
    cuda::GpuMat gry;
    cuda::GpuMat gry2;

    std::vector<RegInfo*> delaunayRegInfos;
    std::vector<Image*> delaunayImages;
#endif

    Subdiv2D subdiv;
    cv::Size image_size;
    std::vector<Mat> channels;
    std::vector<Point2i> imageBoundsAsPolygon;
    std::vector<double> blurVals;
    int blurValIndex = 0;
    std::atomic<unsigned int> jobCount = 0;
    Bbox subdiv_Bbox;
    Poco::Event wakeEvent;
    RegInfo* storedNewInfo;

    std::vector<Point2i> push_for_inferencing(std::vector<Point2i> &_tiles);

    void check_set_render_info();

    long segment_yval_at_point(float xloc, cv::Point2f p1, cv::Point2f p2);

    void add_images_with_composite(std::vector<RegInfo*> new_info);

    void add_images_no_composite(std::vector<RegInfo *> new_info, bool _force_add = false);

    static void clean_face(std::vector<Point2i> &_face);

    static void ensure_clockwise(std::vector<Point2i> &_face);

#ifdef HAVE_OPENCV_CUDAARITHM
    void make_meshgrid();

    void GPU_add_images_no_composite(std::vector<RegInfo *> new_info, bool _force_add = false);

    std::vector<std::pair<Image*,Image*>> calculate_new_overlaps();

    void move_sift_to_cpu(Image*);

    SiftData GPU_extract_SIFT(cuda::GpuMat &_img);

    void coopers_GPU_vectorized_convex_mask_maker(std::vector<Point2i>& _face);
#endif

    void rebuild_DT_elementwise(std::vector<RegInfo *> new_info, bool forceAdd, bool shuffle);

    void create_and_submit_rebuild_jobs();

    void calculate_effected_tiles_round(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result, Vec2 absCoord);

    void calculate_effected_tiles_count_nonzero(Mat polyMaskOutput,std::vector<Point2i> &result, Vec2 absCoord);

    void calculate_effected_tiles(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result, Vec2 absCoord, std::vector<Point2i> *additionalResult = {});

    static void remove_duplicates_without_sort(std::vector<Point2i> &vec);

    void reset_image_as_polygon();

    void debug_draw_voronoi_face(cv::Mat img, std::vector<Point2i> maskAsPolygon, int line_thickness);

    void debug_write_contribution_on_grid(std::string name, Vec2 absCoord, Mat &img, Mat &mask);

    void expand_subdiv(std::vector<RegInfo*> new_info);

    void self_reset();

    void exclude_for_blur();

    int add_point_to_delaunay_triangulation(cv::Point2f _point, pathCam::Image *_image,
                                            std::vector<Point2i> &_face, bool _forceAdd);

    int add_point_to_delaunay_triangulation_with_adjustment(cv::Point2f _point, pathCam::Image *_image,
                                            std::vector<Point2i> &_face, bool _forceAdd);
    
    void coopers_conjugate_gradient(cv::Mat A, cv::Mat b, cv::Mat x, int steps, double epsilon, bool shouldCleanData, std::map<long, long>& systemIndexToFrameIndex, double epsilonClean = 0, cv::Mat bOther = cv::Mat());

    void coopers_conjugate_gradient2(cv::Mat A, cv::Mat b, cv::Mat x, int steps, double epsilon,
                                     bool shouldCleanData, std::map<long, long> &systemIndexToFrameIndex,
                                     double epsilonClean, cv::Mat bOther, int flag);


    void clean_data(cv::Mat A, cv::Mat b, cv::Mat bOther, cv::Mat x, std::map<long, long> &systemIndexToFrameIndex);

  public:

    enum{SIFT_GPU = 0,ORB_CPU};
    CompositeVoronoi(StreamCam *parent, cv::Size image_size, unsigned int componentIndex);

    unsigned int componentMagLabel;
    unsigned int componentIndex;
    long firstImageIdx = -1;
    bool needsAlignment = false;
    std::atomic<unsigned int> matchableCount = 0;
    std::vector<std::pair<long, long>> matchedEdges;
    std::vector<std::pair<Image*, bool>> memberImages;
    std::map<int, unsigned long> delaunayMembers;
    std::vector<Point2i> queuedTiles;
    int inferenceCount = 0;
    int minTilex = 1000;
    int maxTilex = 0;
    int minTiley = 1000;
    int maxTiley = 0;


    void deduce_label();

    void get_flatfield();

    void store_new_info(RegInfo* _new_info);

    void update_from_stored_info();

    void update(std::vector<RegInfo*> _new_info, bool _force_add = false);

    void update_Bbox_no_composite(std::vector<RegInfo*> new_info);

    void perform_global_alignment(unsigned int flag, double closenessFactor);

    void perform_bundle_adjustment(int _featureTypeAndLocation);

    void build_system_from_DT(std::map<long, long> &systemIndexToFrameIndex,
                              std::map<long, long> &frameIndexToSystemIndex, cv::Mat &A, cv::Mat &bx,
                              cv::Mat &by, cv::Mat &x, cv::Mat &y);

    std::map<std::string,int> tileToSumNonZero;
    bool rebuildTile(Point2i tile,int sum);
    void notify_job_complete();
    void save_pyramid_as_image(std::string _fileName = "", bool _withGrid = false ,bool _withGridAndIndexes = false, bool _withEffectedTiles = true, bool _outline = false,std::vector<Point2i> effectedTiles = {});
    void debug_draw_voronoi(Mat &img, Subdiv2D &subdiv,bool _drawPathInsteadOfFaces = false,bool _drawIntersect = false, Point2i _intrCenter = Point2i(0,0));

  protected:
    //std::vector<std::pair<int,int>> falselyClaimedTiles;
    std::vector<Point_<int>> falselyClaimedTiles;
    std::priority_queue<unsigned int> freeMasks;
    std::vector<Mat> masks;
    int removeCount = 0;
  };
}

#endif /* Composite_h */
