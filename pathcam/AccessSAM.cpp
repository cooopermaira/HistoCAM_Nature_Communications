//
// Created by cooper maira on 7/25/25.
//

#include "AccessSAM.h"

namespace pathCam {
  SAMTile::SAMTile(int _ID, Point2i _location, int _size) : ID(_ID), location(_location), size(_size) {
    noncontiguousWrapper = cuda::GpuMat(size, size,CV_8UC4, Scalar(0, 0, 0, 0));
  }

  void SAMTile::set_component_tile(Point2i _tileID, Point2i _subLocation, cuda::GpuMat &_tileMat) {
    componentTiles.emplace_back(_subLocation, _tileID);
    Rect ROI(_subLocation.x * _tileMat.cols,_subLocation.y * _tileMat.rows, _tileMat.cols, _tileMat.rows);
    _tileMat.copyTo(noncontiguousWrapper(ROI));
  }
}
