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
    const Image *imgRef;

    FeatureObservation(int img_id, int feat_id, double x_coord, double y_coord, const Image *_imgRef)
      : image_id(img_id), feature_id(feat_id), x(x_coord), y(y_coord), imgRef(_imgRef) {
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

    void expand(int new_size) {
      if (new_size <= parent.size()) return;

      int old_size = parent.size();
      parent.resize(new_size);
      rank.resize(new_size, 0);

      // Initialize new elements
      for (int i = old_size; i < new_size; i++) {
        parent[i] = i;
      }
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

    size_t size() const { return parent.size(); }
  };

  struct pMatch {
    unsigned long src_img_idx;
    unsigned long dst_img_idx;
    std::vector<DMatch> matches;
  };


  class FeatureTrackGenerator {
  private:
    struct ImageFeaturePair {
      long image_id;
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
    std::unique_ptr<UnionFind> uf_ptr;

  public:
    std::vector<FeatureTrack> generateTracks(const std::vector<Image *> &images,
                                             const std::vector<pMatch> &all_matches);

    void reset() {
      feature_to_index.clear();
      index_to_feature.clear();
      uf_ptr.reset();
    }


    void process_match(long _srcImgIdx, long _dstImgIdx, const DMatch &_match);

    void store_match(std::shared_ptr<Match> _match) { storedMatches.insert(_match); }

    std::vector<FeatureTrack> generateCurrentTracks(const std::vector<Image *> &images);

    Poco::FastMutex accessMutex;

    std::unordered_set<std::shared_ptr<Match> > storedMatches;

    std::unordered_map<int,std::unordered_set<std::shared_ptr<Match>>> interComponentMatches;

  private:
    int getOrCreateFeatureIndex(const ImageFeaturePair &_pair);

    void createGlobalFeatureIndex(const std::vector<Image *> &images);

    void buildConnectionGraph(const std::vector<pMatch> &all_matches, UnionFind &uf);

    std::vector<FeatureTrack> createTracksFromConnections(const std::vector<Image *> &images, const UnionFind &uf);
  };

  class BundleAdjustmentIntegrator {
  public:
    BundleAdjustmentIntegrator() {
      // optimizer = cuba::CudaBundleAdjustment::create();
    };

    // cuba::CudaBundleAdjustment::Ptr optimizer;


    void run_bundle_adjustment(const std::vector<FeatureTrack> &_tracks, const std::vector<Image *> &_images);

    static void run_coopers_planar_bundle_adjustment(const std::vector<FeatureTrack> &_tracks,
                                                     const std::vector<Image *> &_images);

    static std::pair<int, int> run_coopers_planar_ba_edge_list(
      const std::vector<FeatureTrack> &tracks,
      std::vector<Image *> &images,
      int maxIters = 2000,
      double tolRel = 1e-8);


    // Store vertex pointers to maintain ownership
    // std::unordered_map<int, cuba::PoseVertex *> poseVertices;
    // std::unordered_map<int, cuba::LandmarkVertex *> landmarkVertices;
    // std::vector<cuba::MonoEdge *> monoEdges;
    // std::vector<cuba::StereoEdge *> stereoEdges;
  };
}

#endif //SIFTSEARCHUTILS_H
