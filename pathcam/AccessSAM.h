//
// Created by cooper maira on 7/25/25.
//

#ifndef ACCESSSAM_H
#define ACCESSSAM_H

#include "pathCam.h"
//#include "speedSam.h"

namespace pathCam {

  class SAMTile {
  public:
    int ID;
    int priority = 10;
    Point2i location;
    unsigned int size;
    std::vector<std::pair<SAMTile*,std::vector<Point2i> > > neighbors;
    std::vector<std::pair<Point2i,Point2i>> componentTiles;
    cuda::GpuMat noncontiguousWrapper;
    void *rawBuffer,*embed_data_d_,*feats_1_data_d_,*feats_0_data_d_;

    SAMTile(int ID,Point2i _location,unsigned int _size = 1024);
    ~SAMTile() {};

    void set_component_tile(Point2i _tileID, Point2i _subLocation, cuda::GpuMat &_tileMat);

    void get_tile_data(CompositeVoronoi* _comp,unsigned int _interval);

    void make_raw_buffer(void *_buffer);

    void on_click();
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

    AccessSAM(StreamCam* _parent):parent(_parent){};
    ~AccessSAM(){};

    void load_model();
    void initialize();
    void embed_SAM_tiles();
    int get_tile_id(Point2i _location, unsigned int _componentIndex) const;




  };
}

#endif //ACCESSSAM_H
