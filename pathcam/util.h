//
//  util.h
//  pathCam
//
//  Created by Brian on 3/16/23.
//

#ifndef util_h
#define util_h

#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <vector>
#include <tuple>
#include <stdexcept>

namespace std {
  template<>
  struct hash<cv::Point2i> {
    std::size_t operator()(const cv::Point2i &p) const noexcept {
      return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
    }
  };
}

namespace pathCam {
  using namespace nvinfer1;

  struct ScaleResult {
    double scale = 0.0;
    double response = -1.0;
    cv::Point2d shift{0.0, 0.0};
    double shiftNorm = 0.0;
    double shiftNormDiag = 0.0;
    cv::Size cropSize;
    bool valid = false;
  };

  class nvLogger : public ILogger {
    void log(Severity s, const char *msg) noexcept override {
      if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
  };

  extern nvLogger nvloger;

  static std::vector<char> readFile(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
      std::cerr << "Open failed: " << p << "\n";
      std::exit(1);
    }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
  }

  inline void catch_ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh,
                                float lowestScale, bool scaleUp) {

    if (int ans = ExtractSift(siftData, img, numOctaves, initBlur, thresh, lowestScale, scaleUp); ans != 0) {
      int k = 0;
    }
  }



/*
  Header-only image graph that supports:
    - Nodes identified by ImgId (default: long)
    - Each node is either member or non-member
    - Each undirected edge is either ORB or SIFT (exactly one kind per edge)
    - No duplicate nodes
    - No duplicate edges of the same kind between the same node pair
      (ORB and SIFT can both exist for the same unordered pair)
    - Compute a small set of non-member nodes to promote so that the
      induced subgraph on member nodes is connected.
      Preference: ORB-only first; only if impossible, allow SIFT edges as a last resort.
*/

class ImageGraph {
public:
  using ImgId = long;

  enum class EdgeKind : uint8_t { ORB = 0, SIFT = 1 };

  struct PromotionResult {
    bool success = false;                 // connected achieved?
    bool used_sift = false;               // whether we had to allow SIFT to succeed
    std::vector<ImgId> promoted_nodes;    // nodes that were non-member and must become member
  };

public:
  ImageGraph() = default;

  // ----------------------------
  // Node API
  // ----------------------------
  int ensureNode(ImgId id) {
    auto it = id_to_idx_.find(id);
    if (it != id_to_idx_.end()) return it->second;

    const int idx = static_cast<int>(idx_to_id_.size());
    id_to_idx_[id] = idx;
    idx_to_id_.push_back(id);
    adj_.emplace_back();
    member_.push_back(false);
    return idx;
  }

  bool hasNode(ImgId id) const {
    return id_to_idx_.find(id) != id_to_idx_.end();
  }

  void setMember(ImgId id, bool is_member) {
    const int u = ensureNode(id);
    member_[u] = is_member;
  }

  bool isMember(ImgId id) const {
    auto it = id_to_idx_.find(id);
    return (it != id_to_idx_.end()) ? member_[it->second] : false;
  }

  int numNodes() const { return static_cast<int>(adj_.size()); }

  // ----------------------------
  // Edge API (undirected)
  // ----------------------------
  // Returns true iff a NEW edge of that kind was added.
  bool addEdge(ImgId a, ImgId b, EdgeKind kind) {
    const int u = ensureNode(a);
    const int v = ensureNode(b);
    if (u == v) return false;

    const uint64_t key = packEdge(u, v);

    EdgeMask &mask = edge_mask_[key];              // default bits=0 if new
    const bool first_between_pair = (mask.bits == 0);

    const uint8_t bit = uint8_t(1u << static_cast<uint8_t>(kind));
    if (mask.bits & bit) return false;             // duplicate same-kind edge
    mask.bits |= bit;

    // Maintain unique neighbor list: add neighbor only once per unordered pair
    if (first_between_pair) {
      adj_[u].push_back(Neighbor{v, mask.bits});   // store mask snapshot for convenience
      adj_[v].push_back(Neighbor{u, mask.bits});
    } else {
      // Update stored neighbor masks in adjacency lists (so kind checks are O(1) during BFS).
      // This is optional, but keeps adjacency self-contained.
      updateNeighborMask(u, v, mask.bits);
      updateNeighborMask(v, u, mask.bits);
    }

    return true;
  }

  bool hasEdge(ImgId a, ImgId b, EdgeKind kind) const {
    auto ita = id_to_idx_.find(a);
    auto itb = id_to_idx_.find(b);
    if (ita == id_to_idx_.end() || itb == id_to_idx_.end()) return false;

    const int u = ita->second, v = itb->second;
    const auto it = edge_mask_.find(packEdge(u, v));
    if (it == edge_mask_.end()) return false;

    const uint8_t bit = uint8_t(1u << static_cast<uint8_t>(kind));
    return (it->second.bits & bit) != 0;
  }

  // ----------------------------
  // Main solver
  // ----------------------------
  // Returns smallest-ish set of non-members to promote so that all member nodes become mutually connected
  // using walks that traverse only members.
  //
  // Preference:
  //   Phase A: ORB-only edges allowed.
  //   If impossible, Phase B: allow ORB+SIFT, with SIFT treated as last resort.
  //
  // IMPORTANT: This is a greedy Steiner-like heuristic (exact minimum is NP-hard).
  PromotionResult computeMinPromotionsToConnectMembersPreferORB() const {
    // Work on a copy of membership (do NOT mutate caller's graph state).
    std::vector<bool> working_member = member_;

    // Phase A: ORB-only
    {
      auto res = connectByPromotingGreedy(working_member, /*allow_sift=*/false);
      if (res.success) return res;
    }

    // Phase B: allow SIFT (last resort)
    {
      working_member = member_; // reset
      auto res = connectByPromotingGreedy(working_member, /*allow_sift=*/true);
      res.used_sift = res.success; // if it succeeds here, it necessarily used "allow sift" mode
      return res;
    }
  }

private:
  struct EdgeMask { uint8_t bits = 0; }; // bit0=ORB, bit1=SIFT

  struct Neighbor {
    int v = -1;
    uint8_t mask_bits = 0; // available kinds for this unordered pair (1=ORB,2=SIFT,3=both)
  };

private:
  static uint64_t packEdge(int u, int v) {
    if (u > v) std::swap(u, v);
    return (uint64_t(uint32_t(u)) << 32) | uint32_t(v);
  }

  void updateNeighborMask(int u, int v, uint8_t new_mask_bits) {
    auto &nbrs = adj_[u];
    for (auto &nb : nbrs) {
      if (nb.v == v) {
        nb.mask_bits = new_mask_bits;
        return;
      }
    }
    // Should not happen if adjacency is consistent, but keep it safe:
    nbrs.push_back(Neighbor{v, new_mask_bits});
  }

  bool anyMemberExists(const std::vector<bool>& working_member) const {
    for (bool m : working_member) if (m) return true;
    return false;
  }

  // Compute components among *current* members, traversing only member nodes.
  // Edge traversal is restricted by allow_sift.
  std::vector<int> memberComponents(const std::vector<bool>& working_member, bool allow_sift) const {
    const int N = static_cast<int>(adj_.size());
    std::vector<int> comp(N, -1);
    int cid = 0;

    for (int i = 0; i < N; ++i) {
      if (!working_member[i] || comp[i] != -1) continue;

      std::queue<int> q;
      comp[i] = cid;
      q.push(i);

      while (!q.empty()) {
        int u = q.front(); q.pop();
        for (const auto &nb : adj_[u]) {
          const int v = nb.v;
          if (!working_member[v]) continue;

          if (!allow_sift) {
            const bool has_orb = (nb.mask_bits & 0x1u) != 0;
            if (!has_orb) continue;
          } else {
            const bool has_orb_or_sift = (nb.mask_bits & 0x3u) != 0;
            if (!has_orb_or_sift) continue;
          }

          if (comp[v] != -1) continue;
          comp[v] = cid;
          q.push(v);
        }
      }
      cid++;
    }
    return comp;
  }

  int countMemberComponents(const std::vector<bool>& working_member, const std::vector<int>& comp) const {
    int mx = -1;
    for (int i = 0; i < (int)comp.size(); ++i) {
      if (working_member[i]) mx = std::max(mx, comp[i]);
    }
    return mx + 1;
  }

  // Dijkstra bridge search from one member component to any other member component.
  // Allowed to traverse non-members (those would be promoted).
  //
  // Cost:
  //   +1 when entering a non-member node (promotion)
  //   +SIFT_PENALTY when traversing a SIFT edge (Phase B only)
  //
  // Phase A sets allow_sift=false, so SIFT edges are forbidden.
  struct BridgeSearchResult {
    bool found = false;
    int target = -1;
    std::vector<int> parent;
  };

  BridgeSearchResult bestBridgeFromComponent(
      int seed_comp,
      const std::vector<bool>& working_member,
      const std::vector<int>& comp,
      bool allow_sift) const
  {
    const int N = static_cast<int>(adj_.size());
    const int INF = std::numeric_limits<int>::max() / 8;

    // Huge penalty so any ORB-only route is preferred when Phase B is active.
    // Also bounded to avoid overflow.
    const int SIFT_PENALTY = allow_sift ? std::max(1000, (N + 5) * 20) : INF;

    std::vector<int> dist(N, INF);
    std::vector<int> parent(N, -1);

    using PQItem = std::pair<int,int>; // (dist, node)
    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;

    // Multi-source initialization: all member nodes in the seed component at dist=0
    for (int i = 0; i < N; ++i) {
      if (working_member[i] && comp[i] == seed_comp) {
        dist[i] = 0;
        parent[i] = i;
        pq.push({0, i});
      }
    }

    int best_t = -1;

    while (!pq.empty()) {
      auto [du, u] = pq.top(); pq.pop();
      if (du != dist[u]) continue;

      // Reached another member component -> optimal due to Dijkstra
      if (working_member[u] && comp[u] != -1 && comp[u] != seed_comp) {
        best_t = u;
        break;
      }

      for (const auto &nb : adj_[u]) {
        const int v = nb.v;

        // Edge kind filtering / penalties
        if (!allow_sift) {
          if ((nb.mask_bits & 0x1u) == 0) continue; // no ORB edge
        } else {
          if ((nb.mask_bits & 0x3u) == 0) continue;
        }

        int add_edge_cost = 0;
        if (allow_sift) {
          const bool has_orb  = (nb.mask_bits & 0x1u) != 0;
          const bool has_sift = (nb.mask_bits & 0x2u) != 0;

          // If there is an ORB edge for this pair, we treat traversing as "ORB"
          // (zero penalty). Only if ORB is absent but SIFT is present do we add penalty.
          if (!has_orb && has_sift) add_edge_cost = SIFT_PENALTY;
        }

        // Node promotion cost: entering a non-member costs 1
        const int add_node_cost = working_member[v] ? 0 : 1;

        // Relax
        const int nd = du + add_node_cost + add_edge_cost;
        if (nd < dist[v]) {
          dist[v] = nd;
          parent[v] = u;
          pq.push({nd, v});
        }
      }
    }

    BridgeSearchResult out;
    out.found = (best_t != -1);
    out.target = best_t;
    out.parent = std::move(parent);
    return out;
  }

  // Reconstruct a node list from multi-source Dijkstra parent pointers.
  std::vector<int> reconstructPathNodes(int target, const std::vector<int>& parent) const {
    std::vector<int> path;
    int cur = target;
    while (cur >= 0) {
      path.push_back(cur);
      if (parent[cur] == cur) break;
      cur = parent[cur];
    }
    std::reverse(path.begin(), path.end());
    return path;
  }

  PromotionResult connectByPromotingGreedy(std::vector<bool>& working_member, bool allow_sift) const {
    PromotionResult res;

    if (!anyMemberExists(working_member)) {
      res.success = true;
      return res;
    }

    std::vector<char> promoted(adj_.size(), 0);

    while (true) {
      auto comp = memberComponents(working_member, allow_sift);
      const int k = countMemberComponents(working_member, comp);

      if (k <= 1) {
        res.success = true;
        res.used_sift = allow_sift;
        for (int i = 0; i < (int)promoted.size(); ++i) {
          if (promoted[i]) res.promoted_nodes.push_back(idx_to_id_[i]);
        }
        return res;
      }

      // Pick a seed component (component containing the first member node)
      int seed_comp = -1;
      for (int i = 0; i < (int)working_member.size(); ++i) {
        if (working_member[i]) { seed_comp = comp[i]; break; }
      }
      if (seed_comp < 0) { res.success = false; return res; }

      // Find best bridge out of that component
      auto bridge = bestBridgeFromComponent(seed_comp, working_member, comp, allow_sift);
      if (!bridge.found) {
        // Can't connect further under allowed edge set
        res.success = false;
        res.used_sift = allow_sift;
        res.promoted_nodes.clear();
        return res;
      }

      // Promote all non-member nodes along the path (excluding already-members)
      auto path_nodes = reconstructPathNodes(bridge.target, bridge.parent);
      for (int n : path_nodes) {
        if (!working_member[n]) {
          working_member[n] = true;
          promoted[n] = 1;
        }
      }
      // loop; recompute components
    }
  }

public:
  // ----------------------------
  // Optional helpers / introspection
  // ----------------------------
  // Convert ImgId to internal index; returns -1 if not present
  int nodeIndex(ImgId id) const {
    auto it = id_to_idx_.find(id);
    return (it == id_to_idx_.end()) ? -1 : it->second;
  }

  ImgId nodeId(int idx) const { return idx_to_id_.at(idx); }

private:
  // Node storage
  std::unordered_map<ImgId, int> id_to_idx_;
  std::vector<ImgId> idx_to_id_;

  // Per-node membership
  std::vector<bool> member_;

  // Adjacency; each neighbor stores which kinds exist for the unordered pair
  std::vector<std::vector<Neighbor>> adj_;

  // Global per unordered pair: mask bits
  std::unordered_map<uint64_t, EdgeMask> edge_mask_;
};



  struct PointComparator {
    bool operator()(const cv::Point2i &lhs, const cv::Point2i &rhs) const {
      if (lhs.x != rhs.x) {
        return lhs.x < rhs.x; // Compare x-coordinates
      }
      return lhs.y < rhs.y; // Compare y-coordinates if x-coordinates are equal
    }
  };

  class Vec2 {
  public:
    double x, y;

    Vec2() : x(0.0), y(0.0) {
    };

    Vec2(double x, double y) : x(x), y(y) {
    };

    std::string toString() {
      std::stringstream ss;
      ss << "(" << x << "," << y << ")";
      return ss.str();
    }
  };

  class Bbox {
  public:
    double min_x, min_y, max_x, max_y;

    Bbox(double min_x = std::numeric_limits<double>::infinity(),
         double min_y = std::numeric_limits<double>::infinity(),
         double max_x = -std::numeric_limits<double>::infinity(),
         double max_y = -std::numeric_limits<double>::infinity()) : min_x(min_x), min_y(min_y), max_x(max_x),
                                                                    max_y(max_y) {
    };

    bool intersect(Bbox b) {
      return (min_x <= b.max_x && max_x >= b.min_x) &&
             (min_y <= b.max_y && max_y >= b.min_y);
    }

    cv::Rect as_cvRect() {
      return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
    }

    double area(Bbox b) {
      double Omin_x = fmax(min_x, b.min_x);
      double Omin_y = fmax(min_y, b.min_y);
      double Omax_x = fmin(max_x, b.max_x);
      double Omax_y = fmin(max_y, b.max_y);

      //Assuming both boxes are the same size
      double one_area = (max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y);
      double Oarea = (Omax_x - Omin_x) * (Omax_x - Omin_x) + (Omax_y - Omin_y) * (Omax_y - Omin_y);

      return Oarea / one_area;
    }


    std::string toString() {
      std::stringstream ss;
      ss << "[" << min_x << "," << min_y << "," << max_x << "," << max_y << "]";
      return ss.str();
    }
  };


  template<typename T, typename Hash = std::hash<T> >
  class UniqueQueue {
  private:
    std::queue<T> q; // To store elements in FIFO order
    std::unordered_set<T, Hash> seen; // To track unique elements

  public:
    // Push an element into the queue if it is not already present
    void push(const T &value) {
      if (seen.find(value) == seen.end()) {
        // Check for uniqueness
        q.push(value); // Add to the queue
        seen.insert(value); // Mark as seen
      }
    }

    // Pop an element from the front of the queue
    void pop() {
      if (!q.empty()) {
        T value = q.front();
        q.pop(); // Remove from the queue
        seen.erase(value); // Remove from the set
      }
    }

    // Get the front element of the queue
    T front() const {
      if (!q.empty()) {
        return q.front();
      }
      throw std::runtime_error("Queue is empty!");
    }

    // Check if the queue is empty
    bool empty() const {
      return q.empty();
    }

    // Get the size of the queue
    size_t size() const {
      return q.size();
    }
  };

  struct PairHash {
    template<typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const {
      std::size_t h1 = std::hash<T1>()(p.first); // Hash the first element
      std::size_t h2 = std::hash<T2>()(p.second); // Hash the second element
      return h1 ^ (h2 << 1); // Combine the two hashes
    }
  };

  struct TupleHash {
    template<typename T>
    static void hash_combine(std::size_t &seed, const T &value) {
      seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    std::size_t operator()(const std::tuple<int, int, unsigned> &key) const {
      std::size_t seed = 0;
      hash_combine(seed, std::get<0>(key));
      hash_combine(seed, std::get<1>(key));
      hash_combine(seed, std::get<2>(key));
      return seed;
    }
  };


  class ThreadQueue {
  private:
    Poco::ThreadPool *pool;
    std::queue<Poco::Runnable *> jobQueue;

  public:
    ThreadQueue(int min_threads, int max_threads) {
      pool = new Poco::ThreadPool(min_threads, max_threads, 60, POCO_THREAD_STACK_SIZE);
    }

    void run_jobs(std::vector<Poco::Runnable *> jobs) {
      for (unsigned int i = 0; i < jobs.size(); i++) {
        jobQueue.push(jobs[i]);
      }

      while (!jobQueue.empty()) {
        if (pool->available() > 0) {
          pool->start(*jobQueue.front());
          jobQueue.pop();
        } else {
          Poco::Thread::sleep(100);
        }
      }

      pool->joinAll();
    }
  };
}

#endif /* util_h */
