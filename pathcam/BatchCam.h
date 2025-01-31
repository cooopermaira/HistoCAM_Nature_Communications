//
//  BatchCam.h
//  pathCamLib
//
//  Created by Brian on 3/15/23.
//

#ifndef BatchCam_hp
#define BatchCam_hp

#include "pathCam.h"

using Poco::MemoryPool;
using Poco::Path;
using Poco::Logger;

namespace pathCam {
    class BatchCam {
        friend class PairRegRunnable;

    public:
        std::vector<MemoryPool *> mempool;
        Poco::Logger *logger;
        Poco::Logger::Ptr results_logger;
        unsigned int threads;


        Poco::Path input_images;
        Poco::Path out_image;

        Poco::Path flat_field_file_2x;
        Poco::Path flat_field_file_4x;
        Poco::Path flat_field_file_10x;
        Poco::Path flat_field_file_20x;

        Poco::Path tile_encoder_path;
        Poco::Path slide_encoder_path;
        Poco::Path python_venv_path;
        Poco::Path classifier_path;


        //Registration Params
        double crop_factor;
        double scale_factor;
        bool debayer;
        bool real;

        bool inferencing;
        bool aggregating;
        bool pyVenv;
        bool classifying;
        bool classifyingComplete = false;

        int interpolation;
        int feature_type;
        unsigned int image_width;
        unsigned int image_height;
        unsigned int scope_radius;
        bool use_FREAK;
        cv::DescriptorMatcher::MatcherType matcher_type;
        int estimator_type;

        pathCam::FeatureDetector::SIFTParameters SIFT_params;
        pathCam::FeatureDetector::SURFParameters SURF_params;
        pathCam::FeatureDetector::AKAZEParameters AKAZE_params;
        pathCam::FeatureDetector::BRISKParameters BRISK_params;
        pathCam::FeatureDetector::ORBParameters ORB_params;


        std::vector<Image *> images;
        std::vector<RegInfo *> reg_results;
        std::vector<Bbox> box;

        MatchMatrix matchM;
        OverlapMatrix overlapM;
        Bbox combined_box;

    public:
        BatchCam(Poco::Util::LayeredConfiguration::Ptr config);

        ~BatchCam() {
            for (unsigned int i = 0; i < threads; i++) {
                delete mempool[i];
            }
            mempool.clear();

            for (unsigned int i = 0; i < images.size(); i++) {
                delete images[i];
            }
            images.clear();
        }

        virtual bool run();

        float get_scope_radius() { return scope_radius; }

    protected:
        bool parseConfig(Poco::Util::LayeredConfiguration::Ptr pConf);

        bool loadFileList();

        virtual bool resolve_bboxes();

        void find_overlaps();

        bool compositing();

        Mat dome_score_image(int, int, int);

        Mat donut_score_image(int, int, int);
    };
}

#endif /* BatchCam_hp */
