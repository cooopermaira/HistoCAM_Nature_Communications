//
// Created by cooper maira on 6/4/25.
//

#ifndef SIFTSEARCHUTILS_H
#define SIFTSEARCHUTILS_H

#include "pathCam.h"

namespace pathCam {
  inline uint32_t make_stable_id(const long imageIdx, const int featureIdx){
    return (static_cast<uint32_t>(imageIdx) << 10) | static_cast<uint32_t>(featureIdx);
  }
  inline uint32_t image_idx(uint32_t stableID){
    return stableID >> 10;
  }
  inline uint32_t feature_idx(uint32_t stableID){
    return stableID & ((1 << 10) - 1);
  }
  inline uint64_t make_edge_ID(uint32_t a, uint32_t b) {
    if (a > b) {
      std::swap(a,b);
    }
    return ((uint64_t)a << 32) | b;
  }

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
    double world_x, world_y, world_z; // 3D position

    FeatureTrack(int id) : track_id(id), world_x(0), world_y(0), world_z(0) {
    }

    void addObservation(const FeatureObservation &obs) {
      observations.push_back(obs);
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
    bool live = false;
    int lastIteration = 0;

    int systemIdx = -1;
    BAFeature *parent = this;

    std::unordered_map<Image *, int> imageFeatures;

    Point2i cellIdx{0, 0};
    int indexInCell = -1;

    std::vector<uint32_t> stableID;

    BAFeature *find() {
      if (parent != this)
        parent = parent->find();
      return parent;
    }



    static BAFeature *unite(BAFeature *a, BAFeature *b) {
      return a->find();
    }
  };

  struct Observation {
    BAImage *image;
    BAFeature *feature;

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

  struct CellCoord {
    int x, y;

    bool operator==(const CellCoord &o) const {
      return x == o.x && y == o.y;
    }
  };

  struct CellHash {
    size_t operator()(const CellCoord &c) const {
      return std::hash<long long>()(((long long) c.x << 32) ^ (long long) c.y);
    }
  };

  class FeatureGrid {
  public:
    mutable Poco::RWLock rwLock;

    explicit FeatureGrid(float cellSize)
      : cellSize(cellSize), invCellSize(1.0f / cellSize) {
    }

    // ---- insert ----
    void insert(BAFeature *f) {
      auto [cx, cy] = getCell(f->x, f->y);
      auto &cell = grid[{cx, cy}];

      f->cellIdx.x = cx;
      f->cellIdx.y = cy;
      f->indexInCell = (int) cell.size();

      cell.push_back(f);
    }

    // ---- remove ----
    void remove(BAFeature *f) {
      if (f == nullptr) return;
      if (f->indexInCell < 0) return; // already removed / not in grid

      auto it = grid.find({f->cellIdx.x, f->cellIdx.y});
      if (it == grid.end()) {
        f->indexInCell = -1;
        return;
      }

      auto &vec = it->second;
      int idx = f->indexInCell;

      if (idx < 0 || idx >= (int) vec.size() || vec[idx] != f) {
        // stale/corrupt bookkeeping; do slow recovery
        auto vit = std::find(vec.begin(), vec.end(), f);
        if (vit == vec.end()) {
          f->indexInCell = -1;
          return;
        }
        idx = (int) std::distance(vec.begin(), vit);
      }

      BAFeature *last = vec.back();

      vec[idx] = last;
      last->indexInCell = idx;

      vec.pop_back();

      f->indexInCell = -1;
      f->cellIdx.x = 0;
      f->cellIdx.y = 0;

      if (vec.empty()) {
        grid.erase(it);
      }
    }

    // ---- update position (after BA) ----
    void update(BAFeature *f) {
      auto [newX, newY] = getCell(f->x, f->y);

      if (newX == f->cellIdx.x && newY == f->cellIdx.y)
        return;

      remove(f);

      auto &newCell = grid[{newX, newY}];
      f->cellIdx.x = newX;
      f->cellIdx.y = newY;
      f->indexInCell = (int) newCell.size();

      newCell.push_back(f);
    }

    // ---- range query (EXACT) ----
    template<typename Callback>
    void query(float minX, float minY, float maxX, float maxY, Callback &&cb) const {
      Poco::RWLock::ScopedReadLock lock(rwLock); //for .remove()

      int cx0 = (int) std::floor(minX * invCellSize);
      int cy0 = (int) std::floor(minY * invCellSize);
      int cx1 = (int) std::floor(maxX * invCellSize);
      int cy1 = (int) std::floor(maxY * invCellSize);

      for (int cy = cy0; cy <= cy1; ++cy) {
        for (int cx = cx0; cx <= cx1; ++cx) {
          auto it = grid.find({cx, cy});
          if (it == grid.end()) continue;

          for (BAFeature *f: it->second) {
            // exact filter
            if (!f->active) continue;

            // if (f->x >= minX && f->x <= maxX &&
            //     f->y >= minY && f->y <= maxY) {
            cb(f);
            // }
          }
        }
      }
    }

  private:
    float cellSize;
    float invCellSize;

    std::unordered_map<CellCoord, std::vector<BAFeature *>, CellHash> grid;

    inline std::pair<int, int> getCell(float x, float y) const {
      return {
        (int) std::floor(x * invCellSize),
        (int) std::floor(y * invCellSize)
      };
    }
  };

  class FeatureTrackGenerator {
  public:

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

    Poco::RWLock rwLock;

    // Map from (image_id, feature_id) to unique global index
    std::unordered_map<ImageFeaturePair, int, PairHash> feature_to_index;
    std::vector<std::unordered_map<long, int> > component_features;
    std::vector<std::vector<std::shared_ptr<Match> > > adjacency;
    std::vector<ImageFeaturePair> index_to_feature;
    std::unique_ptr<UnionFind> uf_ptr;

    std::unordered_set<std::shared_ptr<Match>> matches;

    FeatureGrid featureGrid{256};

    std::vector<BAFeature *> baFeatures;
    std::vector<BAImage *> baImages;
    SolverState solverState;

    std::unordered_map<uint64_t,uint16_t> coVisEdgeSupport;

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

    void update_coVis_support(Image *img);

    int process_match_queue();

    void add_image(Image *img);

    std::pair<Point2i, bool> estimate_image_coords_from_feature_tracks(Image *img);

    void store_match(std::shared_ptr<Match> _match) { storedMatches.insert(_match); }

    void queue_match(std::shared_ptr<Match> _match) { queuedMatches.push(_match); }

    void launch_inprocess_sparse_CG_iterator(const std::vector<Observation *> &observations, int maxIters = 30,
                                             float tol = 1e-4f);

    std::vector<FeatureTrack> generateCurrentTracks(const std::vector<Image *> &images);

    Poco::FastMutex accessMutex;

    // std::unordered_set<std::shared_ptr<Match>, MatchPtrHash, MatchPtrEqual> storedMatches;
    std::queue<std::shared_ptr<Match>> queuedMatches;
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
