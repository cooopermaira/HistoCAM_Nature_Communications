//
// Created by cooper maira on 6/4/25.
//

#include "pathCam.h"

namespace pathCam {

  void FeatureTrackGenerator::initializeTrackWorldPosition(FeatureTrack &track) {
    // Simple initialization: average of first two observations
    // For your planar case, you might want to use more sophisticated initialization
    if (track.observations.size() >= 2) {
      double avg_x = 0, avg_y = 0;
      for (const auto &obs: track.observations) {
        avg_x += obs.x;
        avg_y += obs.y;
      }
      track.world_x = avg_x / track.observations.size();
      track.world_y = avg_y / track.observations.size();
      track.world_z = 0.0; // Planar assumption
    }
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
      if (feature_indices.size() < 2) continue;

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
          auto coords = img_it->second->absoluteCoords;
          track.addObservation(FeatureObservation(pair.image_id, pair.feature_id, pos.xpos + coords.x, pos.ypos + coords.y));
        }
      }

      if (valid_track && track.observations.size() >= 2) {
        // Initialize world coordinates (you might want to triangulate here)
        initializeTrackWorldPosition(track);
        tracks.push_back(track);
      }
    }

    return tracks;
  }
}
