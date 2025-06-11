//
// Created by cooper maira on 6/4/25.
//

#include "pathCam.h"

namespace pathCam {
  int FeatureTrackGenerator::get_or_create_feature_index(const ImageFeaturePair &_pair) {
    auto it = featureToIndex.find(_pair);
    if (it != featureToIndex.end()) {
      return it->second;
    }

    // Create new index
    int new_index = indexToFeature.size();
    featureToIndex[_pair] = new_index;
    indexToFeature.push_back(_pair);

    // Expand Union-Find if necessary
    if (!uf_ptr) {
      uf_ptr = std::make_unique<UnionFind>(1);
    } else if (new_index >= uf_ptr->size()) {
      // Expand Union-Find structure
      uf_ptr->expand(new_index + 1);
    }

    return new_index;
  }


  // Generate tracks from current state
  std::vector<FeatureTrack> FeatureTrackGenerator::generate_current_tracks(const std::vector<Image *> &_images) {
    if (!uf_ptr || indexToFeature.empty()) {
      return {};
    }

    return createTracksFromConnections(_images, *uf_ptr);
  }

  void FeatureTrackGenerator::process_match(unsigned long _srcImgIdx, unsigned long _dstImgIdx, const DMatch &_match) {
    ImageFeaturePair feat1{_srcImgIdx, _match.queryIdx};
    ImageFeaturePair feat2{_dstImgIdx, _match.trainIdx};

    auto it1 = featureToIndex.find(feat1);
    auto it2 = featureToIndex.find(feat2);

    // Both features should exist since we added all images upfront
    if (it1 != featureToIndex.end() && it2 != featureToIndex.end()) {
      uf_ptr->unite(it1->second, it2->second);
    }else {
      int k = 0;
    }
  }


  void FeatureTrackGenerator::build_connection_graph(const std::vector<pMatch> &all_matches,
                                                   UnionFind &uf) {
    for (const auto &match_info: all_matches) {
      for (const auto &match: match_info.matches) {
        ImageFeaturePair feat1{match_info.src_img_idx, match.queryIdx};
        ImageFeaturePair feat2{match_info.dst_img_idx, match.trainIdx};

        auto it1 = featureToIndex.find(feat1);
        auto it2 = featureToIndex.find(feat2);

        if (it1 != featureToIndex.end() && it2 != featureToIndex.end()) {
          uf.unite(it1->second, it2->second);
        }
      }
    }
  }

  std::vector<FeatureTrack> FeatureTrackGenerator::generate_tracks(const std::vector<Image *> &images,
                                                                  const std::vector<pMatch> &all_matches) {
    // Step 1: Create global indexing for all features
    create_global_feature_index(images);

    // Step 2: Build connection graph using Union-Find
    UnionFind uf(indexToFeature.size());
    build_connection_graph(all_matches, uf);

    // Step 3: Group connected features into tracks
    return createTracksFromConnections(images, uf);
  }

  void FeatureTrackGenerator::create_global_feature_index(const std::vector<Image *> &images) {
    int global_index = 0;

    for (const auto &img: images) {
      for (int feat_id = 0; feat_id < img->siftData.numPts; feat_id++) {
        ImageFeaturePair pair{img->index, feat_id};
        featureToIndex[pair] = global_index;
        indexToFeature.push_back(pair);
        global_index++;
      }
    }
  }

  void FeatureTrackGenerator::addImageFeatures(const Image *_img) {
    if (processedImages.count(_img->index)) {
      return; // Already processed this image
    }

    // Add all features from this image
    for (int feat_id = 0; feat_id < _img->siftData.numPts; feat_id++) {
      ImageFeaturePair pair{_img->index, feat_id};
      int global_idx = indexToFeature.size();

      featureToIndex[pair] = global_idx;
      indexToFeature.push_back(pair);
    }

    // Expand Union-Find if needed
    if (!uf_ptr) {
      uf_ptr = std::make_unique<UnionFind>(indexToFeature.size());
    } else {
      uf_ptr->expand(indexToFeature.size());
    }

    processedImages.insert(_img->index);
  }


  void FeatureTrackGenerator::add_images(const std::vector<Image *> &_images) {
    for (const auto &img: _images) {
      addImageFeatures(img);
    }
  }


  std::vector<FeatureTrack> FeatureTrackGenerator::createTracksFromConnections(
    const std::vector<Image *> &images,
    const UnionFind &uf) {
    // Group features by their root in Union-Find
    std::unordered_map<int, std::vector<int> > root_to_features;

    for (int i = 0; i < indexToFeature.size(); i++) {
      int root = uf.find(i);
      root_to_features[root].push_back(i);
    }

    // Create image lookup for coordinates
    std::unordered_map<unsigned long, const Image *> image_lookup;
    for (auto img: images) {
      image_lookup[img->index] = img;
    }

    // Create tracks
    std::vector<FeatureTrack> tracks;
    int track_id = 0;

    for (const auto &[root, feature_indices]: root_to_features) {
      // Skip single-image tracks (not useful for bundle adjustment)
      if (feature_indices.size() < 3) continue;

      FeatureTrack track(track_id++);

      // Check for multiple observations in same image (should not happen with good matching)
      std::set<int> images_in_track;
      bool valid_track = true;

      for (int global_idx: feature_indices) {
        const auto &pair = indexToFeature[global_idx];

        if (images_in_track.count(pair.image_id)) {
          // Multiple features from same image in track - invalid
          valid_track = false;
          break;
        }
        images_in_track.insert(pair.image_id);

        // Get feature coordinates
        unordered_map<unsigned long, const Image *>::iterator img_it = image_lookup.find(pair.image_id);
        if (img_it != image_lookup.end() && pair.feature_id < img_it->second->siftData.numPts) {
          const auto &pos = img_it->second->siftData.h_data[pair.feature_id];
          track.addObservation(FeatureObservation(pair.image_id, pair.feature_id, pos.xpos, pos.ypos));
          track.world_x += pos.xpos + img_it->second->absoluteCoords.x;
          track.world_y += pos.ypos + img_it->second->absoluteCoords.y;
        }
      }

      if (valid_track && track.observations.size() >= 3) {
        //average out world location;
        track.world_x /= track.observations.size();
        track.world_y /= track.observations.size();

        tracks.push_back(track);
      }
    }

    return tracks;
  }


  void BundleAdjustmentIntegrator::setupBundleAdjustment(const std::vector<FeatureTrack> &_tracks,
                                                         const std::vector<Image *> &_images) {
    //camera poses represent absolute coordinates of images
    for (const auto &img: _images) {
      cuba::CameraParams camParams;

      //10000 is for numerical stability.
      camParams.fx = 10000;
      camParams.fy = 10000;

      //this is essentially "where the camera sits relevant to the image it took" ie the middle
      camParams.cx = 0; //parent->image_width / 2;
      camParams.cy = 0; //parent->image_height / 2;

      //bf only used for stereo photos - not relevant. this is actually set in the constructor as well.
      camParams.bf = 0;

      //images have no rotation
      auto camRotation = Eigen::Quaterniond::Identity();

      //translation should be our current absolute coordinates -> essentially a first guess
      cuba::Array<double, 3> translation = {(img->absoluteCoords.x), (img->absoluteCoords.y), 10000};

      //only fix the root image of the first component, everything else is based on that
      bool fixed = img->regInfo->root;

      auto poseVertex = new cuba::PoseVertex(img->index, camRotation, translation, camParams, fixed);

      //add it to the optimizer
      optimizer->addPoseVertex(poseVertex);

      //keep possession of it
      poseVertices[img->index] = std::move(poseVertex);
    }

    double maxval = 0;
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
    const cuba::RobustKernelType robustKernelType = cuba::RobustKernelType::HUBER;
    const double deltaMono = sqrt(5.991);

    optimizer->setRobustKernels(robustKernelType, deltaMono, cuba::EdgeType::MONOCULAR);

    optimizer->initialize();
    optimizer->optimize(100);
  }
}
