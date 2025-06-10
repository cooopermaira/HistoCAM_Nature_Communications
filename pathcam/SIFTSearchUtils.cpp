//
// Created by cooper maira on 6/4/25.
//

#include "pathCam.h"

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


  // Generate tracks from current state
  std::vector<FeatureTrack> FeatureTrackGenerator::generateCurrentTracks(const std::vector<Image*>& images) {
    if (!uf_ptr || index_to_feature.empty()) {
      return {};
    }

    return createTracksFromConnections(images, *uf_ptr);
  }

  void FeatureTrackGenerator::process_match(unsigned long _srcImgIdx, unsigned long _dstImgIdx, const DMatch &_match) {
    ImageFeaturePair feat1{_srcImgIdx, _match.queryIdx};
    ImageFeaturePair feat2{_dstImgIdx, _match.trainIdx};

    // Get or create indices for both features
    int idx1 = getOrCreateFeatureIndex(feat1);
    int idx2 = getOrCreateFeatureIndex(feat2);

    // Unite them in the Union-Find structure
    uf_ptr->unite(idx1, idx2);
  }



  void FeatureTrackGenerator::buildConnectionGraph(const std::vector<pMatch> &all_matches,
                                                   UnionFind &uf) {
    for (const auto &match_info: all_matches) {

      for (const auto &match: match_info.matches) {

        ImageFeaturePair feat1{match_info.src_img_idx, match.queryIdx};
        ImageFeaturePair feat2{match_info.dst_img_idx, match.trainIdx};

        auto it1 = feature_to_index.find(feat1);
        auto it2 = feature_to_index.find(feat2);

        if (it1 != feature_to_index.end() && it2 != feature_to_index.end()) {
          uf.unite(it1->second, it2->second);
        }
      }
    }
  }

  std::vector<FeatureTrack> FeatureTrackGenerator::generateTracks(const std::vector<Image *> &images,
                                                                  const std::vector<pMatch> &all_matches) {
    // Step 1: Create global indexing for all features
    createGlobalFeatureIndex(images);

    // Step 2: Build connection graph using Union-Find
    UnionFind uf(index_to_feature.size());
    buildConnectionGraph(all_matches, uf);

    // Step 3: Group connected features into tracks
    return createTracksFromConnections(images, uf);
  }

  void FeatureTrackGenerator::createGlobalFeatureIndex(const std::vector<Image *> &images) {
    int global_index = 0;

    for (const auto &img: images) {
      for (int feat_id = 0; feat_id < img->siftData.numPts; feat_id++) {
        ImageFeaturePair pair{img->index, feat_id};
        feature_to_index[pair] = global_index;
        index_to_feature.push_back(pair);
        global_index++;
      }
    }
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
        const auto &pair = index_to_feature[global_idx];

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


  void BundleAdjustmentIntegrator::setupBundleAdjustment(const std::vector<FeatureTrack> &_tracks, const std::vector<Image *> &_images) {

    //camera poses represent absolute coordinates of images
    for (const auto &img : _images) {
      cuba::CameraParams camParams;

      //10000 is for numerical stability.
      camParams.fx = 10000;
      camParams.fy = 10000;

      //this is essentially "where the camera sits relevant to the image it took" ie the middle
      camParams.cx = 0;//parent->image_width / 2;
      camParams.cy = 0;//parent->image_height / 2;

      //bf only used for stereo photos - not relevant. this is actually set in the constructor as well.
      camParams.bf = 0;

      //images have no rotation
      auto camRotation = Eigen::Quaterniond::Identity();

      //translation should be our current absolute coordinates -> essentially a first guess
      cuba::Array<double,3> translation = {(img->absoluteCoords.x), (img->absoluteCoords.y), 10000};

      //only fix the root image of the first component, everything else is based on that
      bool fixed = img->regInfo->root;

      auto poseVertex = new cuba::PoseVertex(img->index,camRotation, translation, camParams,fixed);

      //add it to the optimizer
      optimizer->addPoseVertex(poseVertex);

      //keep possession of it
      poseVertices[img->index] = std::move(poseVertex);
    }

    double maxval = 0;
    //landmark vertexes are feature points placed in world/composite pixel coordinates
    for (const auto &track : _tracks) {
      cuba::Array<double, 3> featurePositionInComposite = {track.world_x,track.world_y,0};

      auto landmarkVertex = new cuba::LandmarkVertex(track.track_id,featurePositionInComposite,false);

      optimizer->addLandmarkVertex(landmarkVertex);

      landmarkVertices[track.track_id] = landmarkVertex;

      //add edges (observations) for this track
      for (const auto &obs : track.observations) {
        //get the pose (image) associated with the obervation
        auto poseVertex = optimizer->poseVertex(obs.image_id);

        cuba::Array<double,2> landmarkPositionInFrame = {obs.x, obs.y};

        auto edge = new cuba::MonoEdge(landmarkPositionInFrame,1.0,poseVertex,landmarkVertex);

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
