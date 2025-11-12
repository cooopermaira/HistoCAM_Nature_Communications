//
// Created by cooper maira on 11/11/25.
//

#include "pathCam.h"

namespace pathCam {

    MetricComposite::MetricComposite(StreamCam *parent, Size image_size, int _componentIndex) : Composite(parent, image_size, _componentIndex){
    }

    void MetricComposite::update() {
        if (!staging.empty()) {
            auto ri = staging.front();
            staging.pop();
            update_Bbox_no_composite({ri});

            //process tiles that new frame can increase status of
        }

    }

    std::vector<std::pair<Point2i,int>> MetricComposite::calculate_effected_tiles_with_status(Point2f _AbC) {
        std::vector<std::pair<Point2i,int>> results;

        if (componentMagLabel == Image::_2X) {
            Point2i center(imageSize.width / 2 + _AbC.x, imageSize.height / 2 + _AbC.y);
            int distance = parent->scope_radius + parent->tileSize;
            int distSq = pow(distance,2);

            Point2f ulP(center.x - distance,center.y - distance);
            Point2f lrP(center.x + distance,center.y + distance);

            auto ulTileInd = imagePyramid->level[0]->getIJ(ulP);
            auto lrTileInd = imagePyramid->level[0]->getIJ(lrP);

            int inlierCorners;
            for (int x = ulTileInd.x; x <= lrTileInd.x ; ++x) {
                for (int y = ulTileInd.y; y <= lrTileInd.y; ++y) {
                    inlierCorners = 0;
                    for (int i = 0; i < 4; ++i) {
                        int locX = parent->tileSize * (x + i%2);
                        int locy = parent->tileSize * (y + i/2);
                        if (pow(locX - center.x,2) +
                            pow(locy - center.y, 2) < distSq) {
                            ++inlierCorners;
                        }
                    }
                    if (inlierCorners == 4) {
                        results.emplace_back(Point2i(x,y),TileObj::singleFrameCoverage);
                    }else if (inlierCorners > 0) {
                        results.emplace_back(Point2i(x,y),TileObj::partialCoverage);
                    }
                }
            }
        }else {
            auto lrP = _AbC + Point2f(float(imageSize.width),float(imageSize.height));

            auto ulTileInd = imagePyramid->level[0]->getIJ(_AbC);
            auto lrTileInd = imagePyramid->level[0]->getIJ(lrP);

            for (int x = ulTileInd.x; x <= lrTileInd.x; ++x) {
                for (int y = ulTileInd.y; y <= lrTileInd.y; ++y) {

                    if (parent->tileSize * x >= _AbC.x && parent->tileSize * x + parent->tileSize <= _AbC.x + imageSize.width &&
                        parent->tileSize * y >= _AbC.y && parent->tileSize * y + parent->tileSize <= _AbC.y + imageSize.height) {
                        results.emplace_back(Point2i(x,y),TileObj::singleFrameCoverage);
                    }else {
                        results.emplace_back(Point2i(x,y),TileObj::partialCoverage);
                    }
                }
            }
        }
        return results;
    }
}