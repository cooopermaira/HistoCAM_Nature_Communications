//
// Created by cooper maira on 7/25/25.
//

#ifndef ACCESSSAM_H
#define ACCESSSAM_H

#include "pathCam.h"

namespace pathCam {

  class SAMTile {
  public:
    int ID;
    Point2i location;
    unsigned int size;
    std::vector<std::pair<SAMTile*,std::vector<Point2i> > > neighbors;
    std::vector<std::pair<Point2i,Point2i>> componentTiles;
    cuda::GpuMat noncontiguousWrapper;
    char* rawBuffer;

    SAMTile(int ID,Point2i _location,unsigned int _size = 1024);
    ~SAMTile(){};

    void set_component_tile(Point2i _tileID, Point2i _subLocation, cuda::GpuMat &_tileMat);

    void make_raw_buffer(char *_buffer);

  };


  class AccessSAM {
    public:
    StreamCam* parent;
    std::vector<SAMTile*> tiles;
    char* batchImageEmbedBuffer;

    Ort::Session* session;

    AccessSAM(StreamCam* _parent):parent(_parent){};
    ~AccessSAM(){};

    void load_model();
    void initialize();
    int get_tile_id(Point2i _location, unsigned int _componentIndex) const;




  };
}

#endif //ACCESSSAM_H
