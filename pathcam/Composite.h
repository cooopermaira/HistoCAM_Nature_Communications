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
#include "MRTiledImage.h"
#include "AccessSAM.h"
#include "StreamCam.h"


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

  public:
    StreamCam *parent;
    Mat local_quality_score, composite_z_buffer, flat_field;
    Mat3f flat_field_composite;
    Mat4b composite;
    Point2i root_offset, max_offset;
    Rect_<float> tiledImageBounds;
    Poco::FastMutex *update_mutex;

    std::shared_ptr<MRTiledImage> imagePyramid;

    int minTilex = 1000;
    int maxTilex = 0;
    int minTiley = 1000;
    int maxTiley = 0;
    int componentIndex = 0;
    int componentMagLabel;
    bool needsAlignment = false;

#ifdef HAVE_OPENCV_CUDAARITHM
    cuda::GpuMat diffGPU;
    cuda::GpuMat xp1;
    cuda::GpuMat xp2;
    cuda::GpuMat binaryCompare;

    long long bigx = 0, bigy = 0;
    cuda::GpuMat meshGridX;
    cuda::GpuMat meshGridY;
    cuda::GpuMat rectMaskGPU;
    cuda::GpuMat circleMaskGPU;
    cuda::GpuMat circleMaskGPU255;
    cuda::GpuMat polyMaskGPU;
    std::vector<cuda::GpuMat> channelsGPU;
    cuda::GpuMat ffGPU;
    cuda::GpuMat convertHoldingGPU;
    cuda::GpuMat threeChannelPrealGPU;
    cuda::GpuMat fourChannelPrealGPU;
    cuda::GpuMat gry;
    cuda::GpuMat gry2;


#endif
    Mat circleMask;

    Mat rectMask;
    Mat threeChannelPreallocated;
    Mat fourChannelPreallocated;

    std::vector<RegInfo *> contributingRegInfos;
    std::set<Image *> contributingImages;
    std::map<int, unsigned long> delaunayMembers;
    std::queue<RegInfo*> staging;


    SiftData GPU_extract_SIFT(cuda::GpuMat &_img, int _numPts);

    bool prepare_4CPA(Image *img, std::vector<Point2i> &affectedTiles);
    bool prepare_4CPA(Image*img, Rect roi = Rect());

    Size imageSize;

    std::vector<float> candidateScaleRatios;

    Composite(StreamCam *parent, Size image_size, int _componentIndex);

    virtual void align_and_rebuild(){};

    void calculate_effected_tiles_round(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result,
                                        Point2f absCoord);

    void stage(RegInfo* _ri){staging.push(_ri);}

    void establish_scale_at_root(Image *_rootImg);

    void set_scale(double _scale);

    void set_offset(const Point2f &_offset) const;

    void deduce_label();

    void set_candidate_scale_ratios();

    void get_flatfield();

    void add_images(std::vector<RegInfo *> new_info);

    void update_Bbox(std::vector<RegInfo *> new_info);

    void update_Bbox_no_composite(std::vector<RegInfo *> new_info);

    virtual void update();

    Mat get_composite();

    Mat score_image_2X(int, int, int);

    void save_pyramid_as_image(std::string _fileName = "", bool _withGrid = false, bool _withGridAndIndexes = false,
                           bool _withEffectedTiles = true, bool _outline = false,
                           std::vector<Point2i> effectedTiles = {});
  };


  class CompositeVoronoi : public Composite {
    friend class ImageToTileCopyRunnable;
    friend class CompositeManager;
    friend class RebuildRunnable;

  public:
    unsigned long lastAcceptedImageIndex;
    unsigned int minPixelDistanceBetweenFrames;
    Vec2 lastAcceptedImageAbC;
    Vec2 lastImageAbC;
    Mat polyMaskOutput;
    Mat freshMask;
    Mat ff;
    Mat convertHolding;



    Subdiv2D subdiv;
    cv::Size image_size;
    std::vector<Mat> channels;
    std::vector<Point2i> imageBoundsAsPolygon;
    std::vector<double> blurVals;
    int blurValIndex = 0;
    std::atomic<unsigned int> jobCount = 0;
    Bbox subdiv_Bbox;
    Poco::Event wakeEvent;
    RegInfo *storedNewInfo;

    std::vector<Point2i> push_for_inferencing(std::vector<Point2i> &_tiles);

    void check_set_render_info();

    void add_images_with_composite(std::vector<RegInfo *> new_info);

    void add_images_no_composite(std::vector<RegInfo *> new_info, bool _force_add = false);

    static void clean_face(std::vector<Point2i> &_face);

    static void ensure_clockwise(std::vector<Point2i> &_face);

#ifdef HAVE_OPENCV_CUDAARITHM
    static std::pair<Point2d, double> phase_correlate_GPU(const cuda::GpuMat &A32F, const cuda::GpuMat &B32F,
                                                          cudaStream_t stream = nullptr);

    static ScaleResult estimate_cale_discrete_GPU(
      const cuda::GpuMat &dLow_bgr,
      const cuda::GpuMat &dHigh_bgr,
      const std::vector<double> &scales,
      cudaStream_t stream);




    static cuda::GpuMat preprocess_GPU(const cuda::GpuMat& bgr_or_gray, cudaStream_t stream);



    static void establish_scale_between_two_centered_Images(Image* img1, Image* img2, double &scale, Point2f &offset);

    void make_meshgrid();

    void GPU_add_images_no_composite(std::vector<RegInfo *> _newInfo, bool _force_add = false);

    std::vector<std::pair<Image *, Image *> > calculate_new_overlaps();



    void rebuild();

    void rebuild_and_initialize_SAM();

    void coopers_GPU_vectorized_convex_mask_maker(std::vector<Point2i> &_face);

    void ff_correct_and_brighten();

    void populate_SAM_tile(SAMTile *_samTile);
#endif
    int pixels_overlapping_between(Image *_img, Rect _rect);

    void rebuild_DT_elementwise(std::vector<RegInfo *> new_info, bool forceAdd, bool shuffle);

    void create_and_submit_rebuild_jobs();

    void calculate_effected_tiles(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result, Point2f absCoord,
                                  std::vector<Point2i> *additionalResult = {});

    static void remove_duplicates_without_sort(std::vector<Point2i> &vec);

    void reset_image_as_polygon();

    void debug_draw_voronoi_face(cv::Mat img, std::vector<Point2i> maskAsPolygon, int line_thickness);

    void debug_write_contribution_on_grid(std::string name, Vec2 absCoord, Mat &img, Mat &mask);

    void expand_subdiv(std::vector<RegInfo *> new_info);

    void self_reset();

    void exclude_for_blur();

    int add_point_to_delaunay_triangulation(cv::Point2f _point, pathCam::Image *_image,
                                            std::vector<Point2i> &_face, bool _forceAdd, bool _drawMask = true);

    int add_point_to_delaunay_triangulation_with_adjustment(cv::Point2f _point, pathCam::Image *_image,
                                                            std::vector<Point2i> &_face, bool _forceAdd);

    void coopers_conjugate_gradient(cv::Mat A, cv::Mat b, cv::Mat x, int steps, double epsilon, bool shouldCleanData,
                                    std::map<long, long> &systemIndexToFrameIndex, double epsilonClean = 0,
                                    cv::Mat bOther = cv::Mat());

    void coopers_conjugate_gradient2(cv::Mat A, cv::Mat b, cv::Mat x, int steps, double epsilon,
                                     bool shouldCleanData, std::map<long, long> &systemIndexToFrameIndex,
                                     double epsilonClean, cv::Mat bOther, int flag);


    void clean_data(cv::Mat A, cv::Mat b, cv::Mat bOther, cv::Mat x, std::map<long, long> &systemIndexToFrameIndex);

  public:
    enum { SIFT_GPU = 0, ORB_CPU };

    CompositeVoronoi(StreamCam *parent, cv::Size image_size, unsigned int componentIndex);

    std::vector<float> candidateScaleRatios;
    long firstImageIdx = -1;
    std::atomic<unsigned int> matchableCount = 0;
    std::vector<std::pair<long, long> > matchedEdges;
    std::vector<std::pair<Image *, bool> > memberImages;

    std::vector<Point2i> queuedTiles;
    int inferenceCount = 0;


    //void set_scale(double _scale);

    // void set_offset(const Point2f &_offset) const;

//    void deduce_label();

    //void set_candidate_scale_ratios();

    //void get_flatfield();

    void store_new_info(RegInfo *_new_info);

    void update_from_stored_info();

    void update(std::vector<RegInfo *> _new_info, bool _force_add = false);

    void perform_global_alignment(unsigned int flag, double closenessFactor);

    void build_system_from_DT(std::map<long, long> &systemIndexToFrameIndex,
                              std::map<long, long> &frameIndexToSystemIndex, cv::Mat &A, cv::Mat &bx,
                              cv::Mat &by, cv::Mat &x, cv::Mat &y);

    std::map<std::string, int> tileToSumNonZero;

    bool rebuildTile(Point2i tile, int sum);

    void notify_job_complete();



    void debug_draw_voronoi(Mat &img, Subdiv2D &subdiv, bool _drawPathInsteadOfFaces = false,
                            bool _drawIntersect = false, Point2i _intrCenter = Point2i(0, 0));

  protected:
    //std::vector<std::pair<int,int>> falselyClaimedTiles;
    std::vector<Point_<int> > falselyClaimedTiles;
    std::priority_queue<unsigned int> freeMasks;
    std::vector<Mat> masks;
    int removeCount = 0;
    double tileupwardsTime = 0;
  };
}

#endif /* Composite_h */
