//
// Created by cooper maira on 7/25/25.
//

#ifndef ACCESSSAM_H
#define ACCESSSAM_H

#include "pathCam.h"

namespace pathCam {

  class SAMTile {
    int ID;
    Point2i location;
    int size;
    std::vector<std::pair<SAMTile*,std::vector<Point2i> > > neighbors;
    std::vector<std::pair<Point2i,Point2i>> componentTiles;
    cuda::GpuMat noncontiguousWrapper;
    char* rawBuffer;

    SAMTile(int _ID,Point2i _location,int _size = 1024);
    ~SAMTile();

    void set_component_tile(Point2i _tileID, Point2i _subLocation, cuda::GpuMat &_tileMat);

    void make_raw_buffer();

  };


  class AccessSAM {
    public:
    StreamCam* parent;

    AccessSAM(StreamCam* _parent);
    ~AccessSAM();

    void initialize();


  };
}

#endif //ACCESSSAM_H
