//
// Created by cooper maira on 11/11/25.
//

#ifndef PATHCAM_METRICCOMPOSITE_H
#define PATHCAM_METRICCOMPOSITE_H

#include "Composite.h"
#include "pathCam.h"

namespace pathCam {
    class FeatureTrackGenerator;
    class BundleAdjustmentIntegrator;
    class MetricComposite : public Composite {
    public:
        MetricComposite(StreamCam *parent, Size image_size, int componentIndex);
        ~MetricComposite();

        void update() override;

        std::vector<std::pair<Point2i,int>> calculate_affected_tiles_with_status(Point2f AbC) const;

        int get_sqrd_center_distance_tile_to_img(Point2i _imgAbC, Point2i _tileCoord) const;

        void process_tiles(Image *img, std::vector<Point2i> &tiles, bool alertDoubleLoad = true, bool forceFullImage = false);

        void align_and_rebuild() override;

        void rebuild(const std::vector<Image *> &members);

        bool image_improves_tile(const std::shared_ptr<TileObj>& _to, const Image* _img) const;

        void search_and_absorb_other_components();

        size_t consolidate_tile_ownership();

        [[nodiscard]] std::unordered_set<Image *> find_contributing_images() const;

        std::vector<std::pair<Image *, Image *>> calculate_member_overlaps(std::vector<Image *> images = {});

        std::vector<std::pair<Image*,std::vector<Point2i>>> waitingFrames;
        std::vector<Image*> memberFrames;

        FeatureTrackGenerator* ftg;
        BundleAdjustmentIntegrator* bai;

        SiftData compSiftData;
        std::vector<std::tuple<Image*,Image*,std::vector<KeyPoint>,std::vector<KeyPoint>>> extraMatches;

        int frameDelay;
        int positionForNextWaitngFrame = 0;
        int debugFrameCount = 0;
        int debugTileCount1 = 0,debugTileCount2 = 0;
        long fhTime = 0;

        bool successfullyAligned = false;
        // bool componentSiftDataInit = false;

        std::atomic<int> outstandingCMS_jobs = 0;
        std::atomic<bool> xcInProgress = false;
        std::atomic<bool> alignmentHasBegun = false;

        std::shared_ptr<TiledImage> baseImage;

        inline static std::mutex EstRoot_mutex;

        Poco::FastMutex cvtMutex;

        char* threeChnBuf;
        char* fourChnBuf;
        char* rectMaskBuf;

        ImageGraph *ig;
    };



}
#endif //PATHCAM_METRICCOMPOSITE_H