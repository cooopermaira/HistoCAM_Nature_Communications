//
// Created by cooper maira on 6/4/25.
//

#include "pathCam.h"
#include <unordered_map>

namespace pathCam {
  int FeatureTrackGenerator::getOrCreateFeatureIndex(const ImageFeaturePair &_pair) {
    auto it = feature_to_index.find(_pair);
    if (it != feature_to_index.end()) {
      return it->second;
    }

    // Create new index
    int new_index = index_to_feature.size();
    feature_to_index[_pair] = new_index;
    index_to_feature.push_back(_pair);

    // Expand Union-Find if necessary
    if (!uf_ptr) {
      uf_ptr = std::make_unique<UnionFind>(1);
    } else if (new_index >= uf_ptr->size()) {
      // Expand Union-Find structure
      uf_ptr->expand(new_index + 1);
    }

    return new_index;
  }




  void FeatureTrackGenerator::process_match(long _srcImgIdx, long _dstImgIdx, const DMatch &_match) {
    ImageFeaturePair feat1{(long) _srcImgIdx, _match.queryIdx};
    ImageFeaturePair feat2{(long) _dstImgIdx, _match.trainIdx};

    // Get or create indices for both features
    int idx1 = getOrCreateFeatureIndex(feat1);
    int idx2 = getOrCreateFeatureIndex(feat2);

    // Unite them in the Union-Find structure
    uf_ptr->unite(idx1, idx2);
  }


  // Generate tracks from current state
  std::vector<FeatureTrack> FeatureTrackGenerator::generateCurrentTracks(const std::vector<Image *> &images) {
    if (!uf_ptr || index_to_feature.empty()) {
      return {};
    }

    auto ans = createTracksFromConnections(images, *uf_ptr);
    return ans;
  }

  std::vector<FeatureTrack> FeatureTrackGenerator::createTracksFromConnections(
    const std::vector<Image *> &images,
    const UnionFind &uf) {
    // Group features by their root in Union-Find
    std::unordered_map<int, std::vector<int> > root_to_features;

    for (int i = 0; i < index_to_feature.size(); i++) {
      int root = uf.find(i);
      root_to_features[root].push_back(i);
    }

    // Create image lookup for coordinates
    std::unordered_map<long, const Image *> image_lookup;
    for (auto img: images) {
      image_lookup[img->index] = img;
    }

    // Create tracks
    std::vector<FeatureTrack> tracks;
    int track_id = 0;

    for (const auto &[root, feature_indices]: root_to_features) {
      // Skip single-image tracks (not useful for bundle adjustment)
      if (feature_indices.size() < 2) continue;

      FeatureTrack track(track_id++);

      // Check for multiple observations in same image (should not happen with good matching)
      std::set<long> images_in_track;
      bool valid_track = true;

      for (int global_idx: feature_indices) {
        const auto &pair = index_to_feature[global_idx];

        if (images_in_track.count(pair.image_id)) {
          // Multiple features from same image in track - invalid
          valid_track = false;
          break;
        }
        images_in_track.insert(pair.image_id);

        // Get feature coordinates
        auto img_it = image_lookup.find(pair.image_id);
        bool cond1 = img_it != image_lookup.end();
        if (cond1 && pair.feature_id < img_it->second->keypointsImageSpace.size()) {
          const auto &pos = img_it->second->keypointsImageSpace[pair.feature_id];


          auto fo = FeatureObservation(int(pair.image_id), pair.feature_id,
            pos.pt.x, pos.pt.y, img_it->second);
          track.addObservation(fo);
          track.world_x += pos.pt.x + img_it->second->regInfo->absoluteCoords.x;
          track.world_y += pos.pt.y + img_it->second->regInfo->absoluteCoords.y;
        }else {
          int k = 0;
        }
      }

      if (valid_track && track.observations.size() >= 2) {
        //average out world location;
        track.world_x /= track.observations.size();
        track.world_y /= track.observations.size();

        tracks.push_back(track);
      }
    }

    return tracks;
  }


  void BundleAdjustmentIntegrator::run_bundle_adjustment(const std::vector<FeatureTrack> &_tracks,
                                                         const std::vector<Image *> &_images) {
    optimizer->clear();
    //camera poses represent absolute coordinates of images
    int stayFixedCount = 0;
    for (const auto &img: _images) {
      cuba::CameraParams camParams;

      //10000 is for numerical stability.
      camParams.fx = 10000;
      camParams.fy = 10000;

      //this is essentially "where the camera sits relative to the image it took" ie the middle (generally)
      //but for simplicity we say the camera was at the image origin (upper left corner)
      camParams.cx = 0; //parent->image_width / 2;
      camParams.cy = 0; //parent->image_height / 2;

      //bf only used for stereo photos - not relevant. this is actually set in the constructor as well.
      camParams.bf = 0;

      //images have no rotation
      auto AbC = img->regInfo->absoluteCoords;

      auto camRotation = Eigen::Quaterniond::Identity();
      cuba::Array<double, 3> translation(-AbC.x,-AbC.y,10000);// * scale);

      bool fixed = img->regInfo->rootOfRoot || img->regInfo->stayFixedDuringBundleAdjustment;
      if (fixed){++stayFixedCount;}

      auto poseVertex = new cuba::PoseVertex(img->index, camRotation, translation, camParams, fixed);

      //add it to the optimizer
      optimizer->addPoseVertex(poseVertex);

      //keep possession of it
      poseVertices[img->index] = std::move(poseVertex);
    }
    assert(stayFixedCount == 1);


    //landmark vertexes are feature points placed in world/composite pixel coordinates
    for (const auto &track: _tracks) {
      cuba::Array<double, 3> featurePositionInComposite = {track.world_x, track.world_y, 0};

      auto landmarkVertex = new cuba::LandmarkVertex(track.track_id, featurePositionInComposite, false);

      optimizer->addLandmarkVertex(landmarkVertex);

      landmarkVertices[track.track_id] = landmarkVertex;

      //add edges (observations) for this track
      for (const auto &obs: track.observations) {
        //get the pose (image) associated with the obervation
        auto poseVertex = optimizer->poseVertex(obs.image_id);

        cuba::Array<double, 2> landmarkPositionInFrame = {obs.x, obs.y};

        auto edge = new cuba::MonoEdge(landmarkPositionInFrame, 1.0, poseVertex, landmarkVertex);

        optimizer->addMonocularEdge(edge);

        monoEdges.push_back(edge);
      }
    }
    constexpr auto robustKernelType = cuba::RobustKernelType::HUBER;
    const double deltaMono = sqrt(5.9);

    optimizer->setRobustKernels(robustKernelType, deltaMono, cuba::EdgeType::MONOCULAR);

    optimizer->initialize();
    optimizer->setPoseUpdateAllowance(false, true);

    optimizer->optimize(100);
  }
}
