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
  class FeatureTrackGenerator;
  class BundleAdjustmentIntegrator;
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
    virtual ~Composite();

    StreamCam *parent;
    Image *root;
    Mat local_quality_score, composite_z_buffer, flat_field;
    Mat3f flat_field_composite;
    Mat4b composite;
    Point2i root_offset, max_offset;
    Rect_<float> tiledImageBounds;
    Poco::FastMutex update_mutex;

    std::shared_ptr<Composite> joinedTo;
    std::unordered_set<int> relatedComponents;

    std::atomic<bool> alignmentHasBegun = false;
    std::atomic<bool> suspended = false;
    std::atomic<int> outstandingCMS_jobs = 0;
    std::vector<std::shared_ptr<Composite>> absorbedComponents;

    bool flatfieldKnown = false;
    bool xcMatchInitiated = false;
    bool xcMatchShouldContinue = true;
    std::vector<Image *> landmarkFrames;
    Image *xcRegLandmark = nullptr;
    Point2f xcPwDist;
    std::vector<std::tuple<Image*,Image*,std::vector<KeyPoint>,std::vector<KeyPoint>>> extraMatches;

    std::shared_ptr<MRTiledImage> imagePyramid;

    int minTilex = 1000;
    int maxTilex = 0;
    int minTiley = 1000;
    int maxTiley = 0;
    int componentIndex = 0;
    int componentMagLabel = -1;
    bool needsAlignment = false;
    int memberCount = 0;
    long maxIndex = -1;

    std::unordered_map<unsigned int, int> observedLabels;


#ifdef HAVE_OPENCV_CUDAARITHM
    cuda::GpuMat diffGPU;
    cuda::GpuMat xp1;
    cuda::GpuMat xp2;
    cuda::GpuMat binaryCompare;

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
    cuda::GpuMat cvtBuffer;


#endif
    Mat circleMask;

    char* threeChnBuf;
    char* fourChnBuf;
    char* rectMaskBuf;

    Mat rectMask;
    Mat threeChannelPreallocated;
    Mat fourChannelPreallocated;
    Mat ff;
    Mat convertHolding;
    std::vector<Mat> channels;


    std::vector<RegInfo *> contributingRegInfos;
    std::set<Image *> contributingImages;
    std::map<int, long> delaunayMembers;
    std::queue<RegInfo *> staging;

    std::vector<Image*> memberFrames;
    std::unordered_set<Image *> contributingFrames, newContributingFrames;
    int frameCount = 0;

    std::atomic<bool> xcInProgress = false;
    inline static std::mutex EstRoot_mutex;


    FeatureTrackGenerator* ftg;
    BundleAdjustmentIntegrator* bai;

    // SiftData GPU_extract_SIFT(cuda::GpuMat &_img, int _numPts);

    bool prepare_4CPA(Image *img, const std::vector<Point2i> &affectedTiles, bool forceFullImage = false);

    bool prepare_4CPA_cpu(Image *img, const std::vector<Point2i> &affectedTiles, bool forceFullImage = false);

    bool prepare_4CPA(Image *img, Rect roi_ = Rect());

    bool prepare_4CPA_cpu(Image *img,Rect roi_ = Rect());

    Size imageSize;

    std::vector<float> candidateScaleRatios;

    Composite(StreamCam *parent, Size image_size, int _componentIndex);

    void launch_component_match_search(Image* img_, bool alertDoubleLoad_ = false);

    virtual int get_exit_rep_count(){ return 0;}

    virtual void align_and_rebuild() {};

    virtual std::unordered_set<Image *> find_contributing_images() const {
      return contributingFrames;
    };

    std::vector<std::pair<Image *, Image *> > calculate_member_overlaps(std::vector<Image *> images) const;


    void sort_overlaps_by_likelihood(std::vector<std::pair<pathCam::Image *, cv::Rect> > &_overlaps, const float &_targetScale);

    virtual void add_landmark_frame(Image* img){};

    void calculate_effected_tiles_round(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result,
                                        Point2f absCoord);

    void suspend() {
      suspended = true;
      imagePyramid->suspended = true;
      imagePyramid->level.clear();
    }

    void stage(RegInfo *_ri) { staging.push(_ri); }

    bool establish_scale_between_pairs(Image *_rootImg, Image *_target, bool _fullImageFtExtract);

    void establish_scale_at_root(Image *_rootImg);

    void establish_scale_at_root_cpu(Image *_rootImg);

    void suspend_and_join(RegInfo *ri_, Image *matchedTo_, Point2i relCoords_);

    // static void sift_to_cvMatch(const SiftData &siftData, Image *image1, Image *image2, int inlierCount,
    //                             const std::vector<uint8_t> &inlierMask, std::vector<
    //                               KeyPoint> &keypoints1, std::vector<KeyPoint> &keypoints2);

    void set_scale(float _scale, bool _ffCorrectExistingTiles = false);

    void ff_correct_existing_tiles();

    double get_scale() const { return imagePyramid->scale; }

    void set_offset(const Point2f &_offset) const;

    void correct_offset() const;

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


}

#endif /* Composite_h */
