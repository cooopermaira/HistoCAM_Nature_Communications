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

  bool coopers_conjugate_gradient(const Mat &A, Mat &x, const Mat &b, int maxIters = 1000, double tol = 1e-8) {
    Mat ATranspose = A.t();
    Mat ATA = ATranspose * A;
    Mat ATb = ATranspose * b;
    Mat r = ATb - (ATA * x);
    Mat p = r.clone();
    double n0 = cv::norm(A * x - b);

    double n1;
    Mat ATAp;

    int iters = 0;
    for (int i = 0; i < maxIters; ++i) {
      ++iters;

      double rdr = r.dot(r);
      ATAp = ATA * p;
      double stepSize = rdr / p.dot(ATAp);

      x += stepSize * p;
      n1 = cv::norm(A * x - b);

      if (abs(n1 - n0) < tol) {
        break;
      }
      n0 = n1;

      r -= stepSize * ATAp;
      double adjustment = r.dot(r) / rdr;
      p = r + adjustment * p;
    }
    std::cout<<"conjugate gradient ran "<<iters<<" iterations"<<std::endl;
    return iters < maxIters;
  }

  void BundleAdjustmentIntegrator::run_coopers_planar_bundle_adjustment(const std::vector<FeatureTrack> &_tracks, const std::vector<Image *> &_images) {

    int totalConstraints = 0;
    for (auto &t:_tracks) {
      totalConstraints += t.observations.size();
    }

    Mat A(totalConstraints,_images.size() - 1 + _tracks.size(),CV_64FC1,Scalar(0));
    Mat xx(_images.size() - 1 + _tracks.size(),1,CV_64FC1,Scalar(0));
    Mat xy(_images.size() - 1 + _tracks.size(),1,CV_64FC1,Scalar(0));
    Mat bx(totalConstraints,1,CV_64FC1,Scalar(0));
    Mat by(totalConstraints,1,CV_64FC1,Scalar(0));

    std::map<long,int> imageToSystem;
    std::map<int,Image*> systemToImage;

    {
      int rootCount = 0;
      int i = 0;
      for (auto & img : _images) {
        if (img->regInfo->root) {
          ++rootCount;
          continue;
        }
        imageToSystem[img->index] = i;
        systemToImage[i] = img;

        xx.at<double>(i,0) = img->regInfo->absoluteCoords.x;
        xy.at<double>(i,0) = img->regInfo->absoluteCoords.y;

        ++i;
      }
      assert(rootCount == 1);
    }

    int rowConstraintIdx = 0;
    for (int i = 0; i < _tracks.size(); ++i){
      int landmarkCol = i + _images.size() - 1;

      xx.at<double>(landmarkCol) = _tracks[i].world_x;
      xy.at<double>(landmarkCol) = _tracks[i].world_y;

      for (auto &obs : _tracks[i].observations) {
        A.at<double>(rowConstraintIdx,landmarkCol) = 1;
        if (!obs.imgRef->regInfo->root) {
          A.at<double>(rowConstraintIdx,imageToSystem[obs.image_id]) = -1;
        }
        bx.at<double>(rowConstraintIdx) = obs.x;
        by.at<double>(rowConstraintIdx) = obs.y;
        ++rowConstraintIdx;
      }
    }

    coopers_conjugate_gradient(A,xx,bx);
    coopers_conjugate_gradient(A,xy,by);

    for (auto img : _images) {
      if (img->regInfo->root) {continue;}
      img->regInfo->absoluteCoords.x = xx.at<double>(imageToSystem[img->index]);
      img->regInfo->absoluteCoords.y = xy.at<double>(imageToSystem[img->index]);
    }

  }

// CPU edge-list conjugate gradient solver for planar (no-rotation) "bundle adjustment"
// Model per observation k (camera i observes landmark j):
//   (L_j - C_i) = u_ij
// Solve least squares:  min_x ||A x - b||^2  where x = [C (Nc), L (Nl)]  (scalar per axis)
// We solve x and y separately with identical structure.
//
// Key points:
// - No OpenCV, no dense matrices
// - No explicit A, no At, no AtA
// - CG is run on normal equations implicitly using edge list
// - Work per CG iteration is O(M) where M=#observations
// - Gauge: root camera is removed from variables (no entry) => treated as fixed at 0
//
// Typical sizes: Nc~299, Nl~20k, M~100k+
// This should be orders of magnitude faster than dense cv::Mat.

#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <cassert>
#include <algorithm>

// ------------------------- Edge list -------------------------

struct EdgeObs {
  int cam;       // 0..Nc-1 (for non-root cameras only)
  int lm;        // 0..Nl-1
  double u;      // observed coordinate (x or y) in camera coords
  bool hasCam;   // false means observation came from root camera (cam term omitted)
};

// ------------------------- Small vector ops -------------------------

static inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
  assert(a.size() == b.size());
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

static inline double norm2(const std::vector<double>& a) {
  return dot(a, a);
}

static inline void axpy(std::vector<double>& y, double alpha, const std::vector<double>& x) {
  assert(y.size() == x.size());
  for (size_t i = 0; i < y.size(); ++i) y[i] += alpha * x[i];
}

static inline void xpay(std::vector<double>& y, const std::vector<double>& x, double beta) {
  // y = x + beta*y
  assert(y.size() == x.size());
  for (size_t i = 0; i < y.size(); ++i) y[i] = x[i] + beta * y[i];
}

// ------------------------- Core operator: AtA * p -------------------------
//
// For each observation (i,j):
//   row residual on vector v is: q = (v_L[j] - (hasCam ? v_C[i] : 0))
// Then (AtA v) accumulates:
//   out_L[j] += q
//   if hasCam: out_C[i] += -q
//
static inline void apply_AtA(
    const std::vector<EdgeObs>& edges,
    int numCams, int numLms,
    const std::vector<double>& v,   // size = numCams + numLms
    std::vector<double>& out)       // size = numCams + numLms
{
  const int N = numCams + numLms;
  assert((int)v.size() == N);
  out.assign(N, 0.0);

  // index helpers: cams [0..numCams-1], lms [numCams..numCams+numLms-1]
  for (const auto& e : edges) {
    const int camIdx = e.cam;
    const int lmIdx  = numCams + e.lm;

    double q = v[lmIdx];
    if (e.hasCam) q -= v[camIdx];

    out[lmIdx] += q;
    if (e.hasCam) out[camIdx] -= q;
  }
}

// ------------------------- Core operator: At*(A x - b) -------------------------
//
// e = (x_L[j] - (hasCam ? x_C[i] : 0)) - u
// Then gradient g = At e is:
//   g_L[j] += e
//   if hasCam: g_C[i] += -e
//
static inline void apply_At_residual(
    const std::vector<EdgeObs>& edges,
    int numCams, int numLms,
    const std::vector<double>& x,   // size N
    std::vector<double>& g)         // size N
{
  const int N = numCams + numLms;
  assert((int)x.size() == N);
  g.assign(N, 0.0);

  for (const auto& e : edges) {
    const int camIdx = e.cam;
    const int lmIdx  = numCams + e.lm;

    double pred = x[lmIdx];
    if (e.hasCam) pred -= x[camIdx];

    const double err = pred - e.u;

    g[lmIdx] += err;
    if (e.hasCam) g[camIdx] -= err;
  }
}

// Optional: compute RMS of Ax-b without forming Ax (for reporting only)
static inline double rms_residual(
    const std::vector<EdgeObs>& edges,
    int numCams, int numLms,
    const std::vector<double>& x)
{
  double sse = 0.0;
  for (const auto& e : edges) {
    const int camIdx = e.cam;
    const int lmIdx  = numCams + e.lm;
    double pred = x[lmIdx];
    if (e.hasCam) pred -= x[camIdx];
    double r = pred - e.u;
    sse += r * r;
  }
  return edges.empty() ? 0.0 : std::sqrt(sse / (double)edges.size());
}

// ------------------------- Conjugate Gradient on normal equations -------------------------
//
// We solve: (AtA) x = Atb
// but we do it without forming AtA or Atb:
//   r = Atb - AtA x  = -At(Ax - b)
// We compute g = At(Ax - b), then set r = -g.
//
// Stop criterion: sqrt(r·r) <= tolAbs OR relative reduction <= tolRel
//
bool coopers_cg_edge_list(
    const std::vector<EdgeObs>& edges,
    int numCams, int numLms,
    std::vector<double>& x,          // size N, in/out
    int maxIters = 200,
    double tolRel = 1e-8,
    double tolAbs = 1e-12,
    bool verbose = true)
{
  const int N = numCams + numLms;
  assert((int)x.size() == N);

  std::vector<double> g, r, p, Ap;

  // r = -At(Ax - b)
  apply_At_residual(edges, numCams, numLms, x, g);
  r.resize(N);
  for (int i = 0; i < N; ++i) r[i] = -g[i];

  p = r;
  double rsold = norm2(r);
  const double rs0 = rsold;

  if (std::sqrt(rsold) <= tolAbs) {
    if (verbose) std::cout << "CG: already converged (abs)\n";
    return true;
  }

  int iters = 0;
  for (; iters < maxIters; ++iters) {
    apply_AtA(edges, numCams, numLms, p, Ap);

    const double denom = dot(p, Ap);
    if (std::abs(denom) < 1e-30) {
      if (verbose) std::cout << "CG: breakdown (singular / bad gauge)\n";
      return false;
    }

    const double alpha = rsold / denom;

    // x += alpha p
    axpy(x, alpha, p);

    // r -= alpha Ap
    axpy(r, -alpha, Ap);

    const double rsnew = norm2(r);

    // stopping
    const double rel = (rs0 > 0.0) ? std::sqrt(rsnew / rs0) : std::sqrt(rsnew);
    if (std::sqrt(rsnew) <= tolAbs || rel <= tolRel) {
      ++iters; // count this iteration
      break;
    }

    const double beta = rsnew / rsold;

    // p = r + beta p
    for (int i = 0; i < N; ++i) p[i] = r[i] + beta * p[i];

    rsold = rsnew;
  }

  if (verbose) {
    std::cout << "CG ran " << iters << " iterations"
              << ", rel_res=" << ((rs0 > 0.0) ? std::sqrt(rsold / rs0) : std::sqrt(rsold))
              << ", rms=" << rms_residual(edges, numCams, numLms, x)
              << "\n";
  }

  return iters < maxIters;
}

// ------------------------- Building the edge list -------------------------
//
// You’ll adapt these to your types; here’s the shape:
//
// - exactly one root image is fixed => not included in variables
// - cameras are remapped to 0..Nc-1 excluding root
// - landmarks are 0..Nl-1 in track vector order (your track_id is already dense)
//
// The resulting unknown vector per axis is:
//   x[0..Nc-1]          = camera coordinate (non-root cams)
//   x[Nc..Nc+Nl-1]      = landmark coordinate
//
// You run twice: one edge list with u=obs.x, and another with u=obs.y
//

// Forward declare your types (replace with your actual includes)

struct PlanarBAResult {
  int numCams = 0;
  int numLms  = 0;
  std::vector<double> x_x; // solution vector for x axis, size Nc+Nl
  std::vector<double> x_y; // solution vector for y axis, size Nc+Nl
};

void BundleAdjustmentIntegrator::run_coopers_planar_ba_edge_list(
    const std::vector<FeatureTrack>& tracks,
    std::vector<Image*>& images,
    int maxIters,
    double tolRel)
{
  // Map images (excluding root) -> camera variable index
  int rootCount = 0;
  std::unordered_map<long,int> imageToCam;
  imageToCam.reserve(images.size() * 2);

  int camIdx = 0;
  for (auto* img : images) {
    if (img->regInfo->root) { ++rootCount; continue; }
    imageToCam[img->index] = camIdx++;
  }
  assert(rootCount == 1);

  const int Nc = camIdx;
  const int Nl = (int)tracks.size();
  const int N  = Nc + Nl;

  // Build edges for x and y
  size_t totalObs = 0;
  for (auto& t : tracks) totalObs += t.observations.size();

  std::vector<EdgeObs> edgesX;
  std::vector<EdgeObs> edgesY;
  edgesX.reserve(totalObs);
  edgesY.reserve(totalObs);

  for (int lm = 0; lm < Nl; ++lm) {
    for (const auto& obs : tracks[lm].observations) {
      EdgeObs ex;
      ex.lm = lm;
      ex.u  = obs.x;

      if (obs.imgRef->regInfo->root) {
        ex.hasCam = false;
        ex.cam = 0; // unused
      } else {
        ex.hasCam = true;
        auto it = imageToCam.find(obs.image_id);
        assert(it != imageToCam.end());
        ex.cam = it->second;
      }
      edgesX.push_back(ex);

      EdgeObs ey = ex;
      ey.u = obs.y;
      edgesY.push_back(ey);
    }
  }

  // Initial guess from existing regInfo + track world coords
  std::vector<double> x0x(N, 0.0), x0y(N, 0.0);

  // cameras
  for (auto* img : images) {
    if (img->regInfo->root) continue;
    int ci = imageToCam.at(img->index);
    x0x[ci] = img->regInfo->absoluteCoords.x;
    x0y[ci] = img->regInfo->absoluteCoords.y;
  }
  // landmarks
  for (int lm = 0; lm < Nl; ++lm) {
    x0x[Nc + lm] = tracks[lm].world_x;
    x0y[Nc + lm] = tracks[lm].world_y;
  }

  // Solve
  bool okx = coopers_cg_edge_list(edgesX, Nc, Nl, x0x, maxIters, tolRel, 1e-12, true);
  bool oky = coopers_cg_edge_list(edgesY, Nc, Nl, x0y, maxIters, tolRel, 1e-12, true);
  (void)okx; (void)oky;

  for (auto* img : images) {
    if (img->regInfo->root) continue;
    int ci = imageToCam.at(img->index);
    img->regInfo->absoluteCoords.x = x0x[ci];
    img->regInfo->absoluteCoords.y = x0y[ci];
  }
}

// After you get PlanarBAResult, write back camera coords:
//   img->regInfo->absoluteCoords.x = res.x_x[camIdx]
// landmarks are in res.x_x[Nc + lm]



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
      poseVertices[img->index] = poseVertex;
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

    //optimizer->setRobustKernels(robustKernelType, deltaMono, cuba::EdgeType::MONOCULAR);

    optimizer->initialize();
    optimizer->setPoseUpdateAllowance(true, true);

    optimizer->optimize(100);
  }
}
