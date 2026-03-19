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
        ~MetricComposite() override;

        void update() override;

        std::vector<std::pair<Point2i,int>> calculate_affected_tiles_with_status(Point2f AbC) const;

        int get_sqrd_center_distance_tile_to_img(Point2i _imgAbC, Point2i _tileCoord) const;

        void process_tiles(Image *img, std::vector<Point2i> &tiles, bool alertDoubleLoad = true, bool forceFullImage = false);

        void align_and_rebuild() override;

        void rebuild(const std::vector<Image *> &members);

        bool image_improves_tile(const std::shared_ptr<TileObj>& _to, const Image* _img) const;

        size_t consolidate_tile_ownership();

        void add_landmark_frame(Image *img) override;

        int get_exit_rep_count() override {return frameDelay;}

        [[nodiscard]] std::unordered_set<Image *> find_contributing_images() const override;

        std::vector<std::pair<Image*,std::vector<Point2i>>> waitingFrames;

        std::unordered_set<Image *> reduce_members_through_competition(std::unordered_set<Image *> _members) const;


        SiftData compSiftData;


        inline static int frameDelay = 10;
        int positionForNextWaitngFrame = 0;
        int debugFrameCount = 0;
        int debugTileCount1 = 0,debugTileCount2 = 0;
        long fhTime = 0;

        bool successfullyAligned = false;
        // bool componentSiftDataInit = false;

        std::shared_ptr<TiledImage> baseImage;

    };



}
#endif //PATHCAM_METRICCOMPOSITE_H