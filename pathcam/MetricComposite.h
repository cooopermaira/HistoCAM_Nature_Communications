//
// Created by cooper maira on 11/11/25.
//

#ifndef PATHCAM_METRICCOMPOSITE_H
#define PATHCAM_METRICCOMPOSITE_H

#include "Composite.h"
#include "pathCam.h"

namespace pathCam {
    class MetricComposite : public Composite {
    public:
        MetricComposite(StreamCam *parent, Size image_size, int componentIndex);

        void update() override;

        std::vector<std::pair<Point2i,int>> calculate_affected_tiles_with_status(Point2f AbC);

        void process_tiles(Image *img, std::vector<Point2i> &tiles);

        std::vector<std::pair<Image*,std::vector<Point2i>>> waitingFrames;

        int frameDelay;
        int positionForNextWaitngFrame = 0;

        int debugFrameCount = 0;
        int debugTileCount1 = 0,debugTileCount2 = 0;

        std::shared_ptr<TiledImage> compositeImage;




    };





}
#endif //PATHCAM_METRICCOMPOSITE_H