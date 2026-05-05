//
// Created by cooper maira on 6/4/25.
//

#ifndef SIFTSEARCHUTILS_H
#define SIFTSEARCHUTILS_H

#include "pathCam.h"

namespace pathCam {
  struct MatchPtrHash {
    size_t operator()(const std::shared_ptr<Match> &m) const {
      auto a = m->image_1;
      auto b = m->image_2;

      // order-independent: sort pointers
      if (a > b) std::swap(a, b);

      size_t h1 = std::hash<Image *>{}(a);
      size_t h2 = std::hash<Image *>{}(b);

      // combine hashes
      return h1 ^ (h2 << 1);
    }
  };

  struct MatchPtrEqual {
    bool operator()(const std::shared_ptr<Match> &m1,
                    const std::shared_ptr<Match> &m2) const {
      auto a1 = m1->image_1;
      auto b1 = m1->image_2;

      auto a2 = m2->image_1;
      auto b2 = m2->image_2;

      return (a1 == a2 && b1 == b2) ||
             (a1 == b2 && b1 == a2);
    }
  };

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

  struct SolverState {
    std::vector<float> img_r_x, img_r_y;
    std::vector<float> img_p_x, img_p_y;
    std::vector<float> img_Ap_x, img_Ap_y;
    std::vector<float> feat_r_x, feat_r_y;
    std::vector<float> feat_p_x, feat_p_y;
    std::vector<float> feat_Ap_x, feat_Ap_y;

    std::vector<float> img_z_x, img_z_y;
    std::vector<float> feat_z_x, feat_z_y;

    std::vector<float> img_inv_diag;
    std::vector<float> feat_inv_diag;
  };

  struct BAImage {
    float x = 0;
    float y = 0;
    bool fixed = false;

    RegInfo *regInfo = nullptr;
    int systemIdx = -1;
    int touchIdx = -1;
  };

  struct BAFeature {
    float x = 0;
    float y = 0;
    bool active = true;
    int lastIteration = 0;

    int systemIdx = -1;
    BAFeature *parent = this;

    std::unordered_map<Image*,int> imageFeatures;

    BAFeature *find() {
      if (parent != this)
        parent = parent->find();
      return parent;
    }

    static BAFeature* unite(BAFeature* a, BAFeature* b) {
      return a->find();
    }
  };

  struct Observation {
    BAImage *image;
    BAFeature *feature;

    int image_idx;
    int feature_idx;

    float obs_x, obs_y;
    float weight = 1.0;
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

    int find(int x) const;

    void expand(int new_size);

    void unite(int x, int y);

    bool connected(int x, int y) const {
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
    std::vector<std::unordered_map<long, int> > component_features;
    std::vector<std::vector<std::shared_ptr<Match> > > adjacency;
    std::unordered_map<long, Image *> imageRefs;
    std::vector<ImageFeaturePair> index_to_feature;
    std::unique_ptr<UnionFind> uf_ptr;

    std::vector<BAFeature *> baFeatures;
    std::vector<BAImage *> baImages;
    SolverState solverState;

    int invalidCount = 0;

  public:
    std::vector<FeatureTrack> generateTracks(const std::vector<Image *> &images,
                                             const std::vector<pMatch> &all_matches);

    void reset() {
      feature_to_index.clear();
      index_to_feature.clear();
      uf_ptr.reset();
    }


    void process_match(const std::shared_ptr<Match> &match_);

    void process_match2(const std::shared_ptr<Match> &match_);

    void add_image(Image *img);

    void store_match(std::shared_ptr<Match> _match) { storedMatches.insert(_match); }

    void launch_inprocess_sparse_CG_iterator(const std::vector<Observation *> &observations, int maxIters = 20, float tol = 1e-4f);

    std::vector<FeatureTrack> generateCurrentTracks(const std::vector<Image *> &images);

    Poco::FastMutex accessMutex;

    // std::unordered_set<std::shared_ptr<Match>, MatchPtrHash, MatchPtrEqual> storedMatches;
    std::unordered_set<std::shared_ptr<Match> > storedMatches;
    std::unordered_map<int, std::unordered_set<std::shared_ptr<Match> > > interComponentMatches;

  private:
    int get_or_create_feature_index(const ImageFeaturePair &_pair);

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
