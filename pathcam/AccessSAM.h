//
// Created by cooper maira on 7/25/25.
//

#ifndef ACCESSSAM_H
#define ACCESSSAM_H

#include "pathCam.h"
//#include "speedSam.h"

namespace pathCam {
  struct Point2iComparator {
    bool operator()(const cv::Point2i& a, const cv::Point2i& b) const {
      return std::tie(a.x, a.y) < std::tie(b.x, b.y);
    }
  };


  class AccessSAM;

  class SAMTile {
  public:
    int ID;
    int priority = 10;
    Point2i location;
    unsigned int size;
    AccessSAM* as;
    std::vector<std::pair<SAMTile*,std::vector<Point2i> > > neighbors;
    std::vector<std::pair<Point2i,Point2i>> componentTiles;
    cuda::GpuMat noncontiguousWrapper;

    void *rawBuffer,*embed_data_d_,*feats_1_data_d_,*feats_0_data_d_;
    void *clicksGPU, *clickLabelsGPU,*inputMask,*hasMaskInputGPU,*outputMask,*confidence;
    bool hasMaskInput = false;
    std::vector<Point3f> clicksVec;

    cuda::GpuMat inputMaskMat;
    std::map<int,cuda::GpuMat> segmentations;

    SAMTile(int ID, Point2i _location, AccessSAM* _as, unsigned int _size = 1024);
    ~SAMTile() {};

    void set_component_tile(Point2i _tileID, Point2i _subLocation, cuda::GpuMat &_tileMat);

    void get_tile_data(CompositeVoronoi* _comp,unsigned int _interval);

    void make_raw_buffer(void *_buffer);

    void on_click();

    void run_segmentation(int _segmentationID);
  };


  inline bool tile_compare(const SAMTile* a, const SAMTile* b) {
    return a->priority < b->priority;
  }
  //class SpeedSam;
  class AccessSAM {
    public:
    StreamCam* parent;
    std::vector<SAMTile*> tiles;
    char* batchImageEmbedBuffer;
    SpeedSam* speedSam;

    std::queue<SAMTile*> segmentProcessQ;

    std::map<Point2i,std::vector<std::pair<int,Mat>>,Point2iComparator> segmentationMasks;

    AccessSAM(StreamCam* _parent):parent(_parent){};
    ~AccessSAM(){};

    void load_model();
    void initialize();
    void embed_SAM_tiles();
    static void get_clicks_embedding(std::vector<Point3f> &_clicks, void *&_clicksGPU, void *&_clickLabelsGPU);
    int get_tile_id(Point2i _location, unsigned int _componentIndex) const;




  };
}

#endif //ACCESSSAM_H
