//
// Created by cooper maira on 11/11/25.
//

#ifndef PATHCAM_METRICCOMPOSITE_H
#define PATHCAM_METRICCOMPOSITE_H

#include "Composite.h"
#include "pathCam.h"

namespace pathCam {
    class FeatureTrackGenerator;
    class MetricComposite : public Composite {
    public:
        MetricComposite(StreamCam *parent, Size image_size, int componentIndex);

        void update() override;

        std::vector<std::pair<Point2i,int>> calculate_affected_tiles_with_status(Point2f AbC) const;

        int get_sqrd_center_distance_tile_to_img(Point2i _imgAbC, Point2i _tileCoord);

        void process_tiles(Image *img, std::vector<Point2i> &tiles);

        void align_and_rebuild() override;

        bool image_improves_tile(std::shared_ptr<TileObj> _to, Image* _img);

        std::unordered_set<Image *> find_contributing_images() const;

        std::vector<std::pair<Image *, Image *>> calculate_member_overlaps(std::vector<Image *> images = {});

        std::vector<std::pair<Image*,std::vector<Point2i>>> waitingFrames;

        FeatureTrackGenerator* ftg;

        int frameDelay;
        int positionForNextWaitngFrame = 0;
        int debugFrameCount = 0;
        int debugTileCount1 = 0,debugTileCount2 = 0;
        long fhTime = 0;

        std::atomic<int> outstandingCMS_jobs = 0;

        bool xcMatchInitiated = false;

        std::shared_ptr<TiledImage> compositeImage;

        Image* mostRecentFrame = nullptr;

        inline static std::mutex EstRoot_mutex;

    };





}
#endif //PATHCAM_METRICCOMPOSITE_H