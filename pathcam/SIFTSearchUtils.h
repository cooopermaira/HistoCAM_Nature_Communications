//
// Created by cooper maira on 6/4/25.
//

#ifndef SIFTSEARCHUTILS_H
#define SIFTSEARCHUTILS_H

#include "pathCam.h"

namespace pathCam {
  struct FeatureObservation {
    int image_id;
    int feature_id; // feature index within that image
    double x, y; // pixel coordinates

    FeatureObservation(int img_id, int feat_id, double x_coord, double y_coord)
      : image_id(img_id), feature_id(feat_id), x(x_coord), y(y_coord) {
    }
  };

  // Represents a track - a single 3D landmark observed across multiple images
  struct FeatureTrack {
    int track_id;
    std::vector<FeatureObservation> observations;
    double world_x, world_y, world_z; // 3D position (z will be 0 for your planar case)

    FeatureTrack(int id) : track_id(id), world_x(0), world_y(0), world_z(0) {
    }

    void addObservation(const FeatureObservation &obs) {
      observations.push_back(obs);
    }

    // Check if this track contains an observation from a specific image
    bool hasImageObservation(int image_id) const {
      for (const auto &obs: observations) {
        if (obs.image_id == image_id) return true;
      }
      return false;
    }

    // Get observation for specific image (returns nullptr if not found)
    const FeatureObservation *getObservation(int image_id) const {
      for (const auto &obs: observations) {
        if (obs.image_id == image_id) return &obs;
      }
      return nullptr;
    }
  };


  class UnionFind {
  private:
    mutable std::vector<int> parent;
    mutable std::vector<int> rank;

  public:
    UnionFind(int n) : parent(n), rank(n, 0) {
      for (int i = 0; i < n; i++) {
        parent[i] = i;
      }
    }

    int find(int x) const {
      if (parent[x] != x) {
        parent[x] = find(parent[x]); // path compression
      }
      return parent[x];
    }

    void unite(int x, int y) {
      int px = find(x);
      int py = find(y);

      if (px == py) return;

      if (rank[px] < rank[py]) {
        parent[px] = py;
      } else if (rank[px] > rank[py]) {
        parent[py] = px;
      } else {
        parent[py] = px;
        rank[px]++;
      }
    }

    bool connected(int x, int y) {
      return find(x) == find(y);
    }
  };

  struct pMatch {
    unsigned long src_img_idx;
    unsigned long dst_img_idx;
    std::vector<DMatch> matches;
  };


  class FeatureTrackGenerator {
  private:
    struct ImageFeaturePair {
      unsigned long image_id;
      int feature_id;

      bool operator<(const ImageFeaturePair &other) const {
        if (image_id != other.image_id) return image_id < other.image_id;
        return feature_id < other.feature_id;
      }

      bool operator==(const ImageFeaturePair &other) const {
        return image_id == other.image_id && feature_id == other.feature_id;
      }
    };

    struct PairHash {
      size_t operator()(const ImageFeaturePair &p) const {
        return std::hash<unsigned long>()(p.image_id) ^ (std::hash<int>()(p.feature_id) << 1);
      }
    };

    // Map from (image_id, feature_id) to unique global index
    std::unordered_map<ImageFeaturePair, int, PairHash> feature_to_index;
    std::vector<ImageFeaturePair> index_to_feature;

    // Your SIFT data structure - adapt as needed
    struct ImageData {
      int index;
      // Add your SiftData structure here
      std::vector<cv::Point2f> feature_positions; // x,y coordinates of features
    };

  public:
    std::vector<FeatureTrack> generateTracks(const std::vector<Image *> &images,
                                             const std::vector<pMatch> &all_matches);

  private:
    void createGlobalFeatureIndex(const std::vector<Image *> &images);

    void buildConnectionGraph(const std::vector<pMatch> &all_matches, UnionFind &uf);

    std::vector<FeatureTrack> createTracksFromConnections(
      const std::vector<Image *> &images,
      const UnionFind &uf);

    void initializeTrackWorldPosition(FeatureTrack &track);
  };

class BundleAdjustmentIntegrator {
public:
  void setupBundleAdjustment(
      const std::vector<FeatureTrack>& tracks,
      const std::vector<Image*>& images,
      cuba::CudaBundleAdjustment::Ptr optimizer) {

    // Camera parameters for different magnifications
    std::map<int, cuba::CameraParams> magnification_cameras = {
        {2,  {6000 * 2,  6000 * 2,  3232, 2426, 0}}, // 2x
        {4,  {6000 * 4,  6000 * 4,  3232, 2426, 0}}, // 4x
        {10, {6000 * 10, 6000 * 10, 3232, 2426, 0}}, // 10x
        {20, {6000 * 20, 6000 * 20, 3232, 2426, 0}}, // 20x
        {40, {6000 * 40, 6000 * 40, 3232, 2426, 0}}  // 40x
    };

    // Add pose vertices (one per image)
    for (const auto& img : images) {
      // You'll need to determine magnification for each image
      int magnification = getMagnificationForImage(img.index); // Implement this
      auto camera = magnification_cameras[magnification];

      // Initial pose (identity rotation, zero translation)
      Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
      cuba::Array<double, 3> t = {0.0, 0.0, 0.0};

      auto pv = cuba::PoseVertex()
      auto pose_vertex = obj.create<cuba::PoseVertex>(img->index, q, t, camera, false);
      optimizer->addPoseVertex(pose_vertex);
    }

    // Add landmark vertices (one per track)
    for (const auto& track : tracks) {
      cuba::Array<double, 3> world_pos = {track.world_x, track.world_y, track.world_z};
      auto landmark_vertex = obj.create<cuba::LandmarkVertex>(track.track_id, world_pos, false);
      optimizer->addLandmarkVertex(landmark_vertex);
    }

    // Add edges (observations)
    for (const auto& track : tracks) {
      auto landmark_vertex = optimizer->landmarkVertex(track.track_id);

      for (const auto& obs : track.observations) {
        auto pose_vertex = optimizer->poseVertex(obs.image_id);

        cuba::Array<double, 2> measurement = {obs.x, obs.y};
        double information = 1.0; // You might want to adjust this based on feature quality

        auto edge = obj.create<cuba::MonoEdge>(measurement, information, pose_vertex, landmark_vertex);
        optimizer->addMonocularEdge(edge);
      }
    }
  }

private:
  int getMagnificationForImage(int image_id) {
    // Implement based on your image naming or metadata
    // This is just a placeholder
    return 10; // Default to 10x
  }
};

}

#endif //SIFTSEARCHUTILS_H
