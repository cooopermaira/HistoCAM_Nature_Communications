//
// Created by cooper maira on 7/25/25.
//

#ifndef ACCESSSAM_H
#define ACCESSSAM_H

#include "pathCam.h"
//#include "speedSam.h"

namespace pathCam {
  struct Point2iComparator {
    bool operator()(const cv::Point2i &a, const cv::Point2i &b) const {
      return std::tie(a.x, a.y) < std::tie(b.x, b.y);
    }
  };


  class AccessSAM;
  class CompositeVoronoi;

  class SAMTile {
  public:
    int ID;
    int priority = 10;
    Point2i location;
    unsigned int size;
    unsigned int componentIndex;
    AccessSAM *as;
    std::vector<std::pair<SAMTile *, std::vector<Point2i> > > neighbors;
    std::vector<std::pair<Point2i, Point2i> > componentTiles;
    cuda::GpuMat noncontiguousWrapper;
    Mat ncwStoreLocal;

    bool embeddingComplete = false;
    void *rawBuffer, *embed_data_d_, *feats_1_data_d_, *feats_0_data_d_;
    void *clicksGPU, *clickLabelsGPU, *inputMask, *hasMaskInputGPU, *outputMask, *confidence;
    bool hasMaskInput = false;
    std::vector<Point3f> clicksVec;

    cuda::GpuMat inputMaskMat;
    std::map<int, cuda::GpuMat> segmentations;

    cudaEvent_t embeddingCompleteCudaEvent;

    SAMTile(int ID, Point2i _location, AccessSAM *_as, unsigned _componentIndex, unsigned _size = 1024);

    ~SAMTile() {
    };

    void set_component_tile(Point2i _tileID, Point2i _subLocation, cuda::GpuMat &_tileMat);

    void get_tile_data(CompositeVoronoi *_comp, unsigned int _interval);

    void make_raw_buffer(void *_buffer);

    void increase_embed_priority();

    void run_segmentation(int _segmentationID);

    Mat debug_draw_tile_with_clicks_and_mask(const cv::Mat &bgraImage, const cv::Mat &binaryMask,
                                              const cv::Scalar &shadeColor, float alpha);
  };


  inline bool tile_compare(const SAMTile *a, const SAMTile *b) {
    return a->priority < b->priority;
  }

  //class SpeedSam;
  class AccessSAM {
  public:
    bool initialized = false;

    StreamCam *parent;
    std::vector<SAMTile *> tiles;
    SpeedSam *speedSam;

    std::queue<SAMTile *> segmentProcessQ;

    std::map<Point2i, std::vector<std::pair<int, Mat> >, Point2iComparator> segmentationMasks;

    AccessSAM(StreamCam *_parent): parent(_parent) {
    };

    ~AccessSAM() {
    };

    //loads SAM encoder and decoder as speedSam object. Can take .engine or .onnx
    void load_model();

    //creates SAM tile objects and links them to their neighbors. This function does not manipulate image data
    void initialize();

    //transforms click data into raw buffers (floats)
    static void get_clicks_embedding(std::vector<Point3f> &_clicks, void *&_clicksGPU, void *&_clickLabelsGPU);

    //calculates the tile ID from the point location and component index
    int get_tile_id(Point2i _tileIndexPoint, unsigned int _componentIndex) const;

    //fills SAM tiles with image data and creates SAM embedding for each tile. Priority of each tile can be adjusted on the fly
    void embed_SAM_tiles();

    void push_mask_for_display(Point2i _tile, unsigned int _componentIndex, const Mat &_mask, int _segID);

    void create_segmentation(std::vector<Point3f> &_clicks, int _segID);

    std::vector<int> get_tiles_covering_point(const Point2f &_p, int _stride = 768);

    static std::map<int, std::vector<Point3f>> choose_clicks_for_each_tile(const std::map<int, std::vector<std::pair<Point3f,bool>>>& clicksByTile,
                                                                           int cap = 10);

    void process_segmentation_Q(int _segID);

    static int floorDiv(int a, int b) {
      int q = a / b;
      int r = a % b;
      return (r && ((r > 0) != (b > 0))) ? (q - 1) : q;
    }
  };
}

#endif //ACCESSSAM_H
