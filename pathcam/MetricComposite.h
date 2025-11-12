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

        std::vector<std::pair<Point2i,int>> calculate_effected_tiles_with_status(Point2f _AbC);

        //std::queue<std::pair<Image*,std::vector




    };





}
#endif //PATHCAM_METRICCOMPOSITE_H