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
  class Image;
  class Match;

  class ImageGraph {
  public:
    using ImgId = long;

    enum class EdgeKind : uint8_t { ORB = 0, SIFT = 1 };

    struct PromotionResult {
      bool success = false; // connected achieved?
      bool used_sift = false; // whether we had to allow SIFT to succeed
      std::vector<ImgId> promoted_nodes; // nodes that were non-member and must become member
    };

  public:
    ImageGraph() = default;

    static void PromoteMembersForOverlapConnectivityShortestHop(
      std::unordered_set<Image *> &members,
      const std::vector<std::pair<Image *, Image *> > &overlaps,
      const std::vector<std::shared_ptr<Match> > &allRegistrations);

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

      EdgeMask &mask = edge_mask_[key]; // default bits=0 if new
      const bool first_between_pair = (mask.bits == 0);

      const uint8_t bit = uint8_t(1u << static_cast<uint8_t>(kind));
      if (mask.bits & bit) return false; // duplicate same-kind edge
      mask.bits |= bit;

      // Maintain unique neighbor list: add neighbor only once per unordered pair
      if (first_between_pair) {
        adj_[u].push_back(Neighbor{v, mask.bits}); // store mask snapshot for convenience
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
    struct EdgeMask {
      uint8_t bits = 0;
    }; // bit0=ORB, bit1=SIFT

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
      for (auto &nb: nbrs) {
        if (nb.v == v) {
          nb.mask_bits = new_mask_bits;
          return;
        }
      }
      // Should not happen if adjacency is consistent, but keep it safe:
      nbrs.push_back(Neighbor{v, new_mask_bits});
    }

    bool anyMemberExists(const std::vector<bool> &working_member) const {
      for (bool m: working_member) if (m) return true;
      return false;
    }

    // Compute components among *current* members, traversing only member nodes.
    // Edge traversal is restricted by allow_sift.
    std::vector<int> memberComponents(const std::vector<bool> &working_member, bool allow_sift) const {
      const int N = static_cast<int>(adj_.size());
      std::vector<int> comp(N, -1);
      int cid = 0;

      for (int i = 0; i < N; ++i) {
        if (!working_member[i] || comp[i] != -1) continue;

        std::queue<int> q;
        comp[i] = cid;
        q.push(i);

        while (!q.empty()) {
          int u = q.front();
          q.pop();
          for (const auto &nb: adj_[u]) {
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

    int lowestMemberIdxById(const std::vector<bool> &m) const {
      int best = -1;
      ImgId best_id = 0;
      for (int i = 0; i < (int) m.size(); ++i) {
        if (!m[i]) continue;
        ImgId id = idx_to_id_[i];
        if (best == -1 || id < best_id) {
          best = i;
          best_id = id;
        }
      }
      return best;
    }

    int countMemberComponents(const std::vector<bool> &working_member, const std::vector<int> &comp) const {
      int mx = -1;
      for (int i = 0; i < (int) comp.size(); ++i) {
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
      const std::vector<bool> &working_member,
      const std::vector<int> &comp,
      bool allow_sift) const {
      const int N = static_cast<int>(adj_.size());
      const int INF = std::numeric_limits<int>::max() / 8;

      // Huge penalty so any ORB-only route is preferred when Phase B is active.
      // Also bounded to avoid overflow.
      const int SIFT_PENALTY = allow_sift ? std::max(1000, (N + 5) * 20) : INF;

      std::vector<int> dist(N, INF);
      std::vector<int> parent(N, -1);

      using PQItem = std::pair<int, int>; // (dist, node)
      std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem> > pq;

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
        auto [du, u] = pq.top();
        pq.pop();
        if (du != dist[u]) continue;

        // Reached another member component -> optimal due to Dijkstra
        if (working_member[u] && comp[u] != -1 && comp[u] != seed_comp) {
          best_t = u;
          break;
        }

        for (const auto &nb: adj_[u]) {
          const int v = nb.v;

          // Edge kind filtering / penalties
          if (!allow_sift) {
            if ((nb.mask_bits & 0x1u) == 0) continue; // no ORB edge
          } else {
            if ((nb.mask_bits & 0x3u) == 0) continue;
          }

          int add_edge_cost = 0;
          if (allow_sift) {
            const bool has_orb = (nb.mask_bits & 0x1u) != 0;
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
    std::vector<int> reconstructPathNodes(int target, const std::vector<int> &parent) const {
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

    PromotionResult connectByPromotingGreedy(std::vector<bool> &working_member, bool allow_sift) const {
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
          for (int i = 0; i < (int) promoted.size(); ++i) {
            if (promoted[i]) res.promoted_nodes.push_back(idx_to_id_[i]);
          }
          return res;
        }

        // Pick a seed component (component containing the first member node)
        int seed_comp = -1;
        for (int i = 0; i < (int) working_member.size(); ++i) {
          if (working_member[i]) {
            seed_comp = comp[i];
            break;
          }
        }
        if (seed_comp < 0) {
          res.success = false;
          return res;
        }

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
        for (int n: path_nodes) {
          if (!working_member[n]) {
            working_member[n] = true;
            promoted[n] = 1;
          }
        }
        // loop; recompute components
      }
    }

  public:
    static int hopDistanceToRoot(
      ImgId node,
      ImgId root,
      const std::unordered_map<ImgId, ImgId> &parent,
      std::unordered_map<ImgId, int> &memo) {
      if (node == root) return 0;

      if (auto it = memo.find(node); it != memo.end())
        return it->second;

      auto itp = parent.find(node);
      if (itp == parent.end())
        return std::numeric_limits<int>::max() / 8; // unreachable / not in tree

      ImgId p = itp->second;
      if (p == node) // bad parent map (cycle/self but not root)
        return std::numeric_limits<int>::max() / 8;

      int d = hopDistanceToRoot(p, root, parent, memo);
      if (d >= std::numeric_limits<int>::max() / 16)
        return memo[node] = d;

      return memo[node] = d + 1;
    }

    // ----------------------------
    // Optional helpers / introspection
    // ----------------------------
    // Convert ImgId to internal index; returns -1 if not present
    int nodeIndex(ImgId id) const {
      auto it = id_to_idx_.find(id);
      return (it == id_to_idx_.end()) ? -1 : it->second;
    }

    ImgId nodeId(int idx) const { return idx_to_id_.at(idx); }

    // Compute a parent tree from root over FINAL members (original members + promoted),
    // where paths prefer ORB exactly like the solver:
    // - Traversing an edge that has ORB available costs (sift=0, hops=1)
    // - Traversing an edge that is SIFT-only costs (sift=1, hops=1)
    //
    // Minimizes lexicographically: (num_sift_only_edges, hop_count).
    //
    // Returns false if some final member is unreachable even with ORB+SIFT edges.
    bool computePreferredParentsToRootAfterPromotions(
      ImgId &out_root,
      std::unordered_map<ImgId, ImgId> &out_parent) const {
      out_parent.clear();

      // Build final membership
      std::vector<bool> final_member = member_;

      const int root_idx = lowestMemberIdxById(final_member);
      if (root_idx < 0) return false; // no members at all

      out_root = idx_to_id_[root_idx];

      // Lexicographic distance: first minimize siftEdges, then hops.
      struct Dist {
        int sift = std::numeric_limits<int>::max() / 8;
        int hops = std::numeric_limits<int>::max() / 8;
      };
      auto better = [](const Dist &a, const Dist &b) {
        if (a.sift != b.sift) return a.sift < b.sift;
        return a.hops < b.hops;
      };

      const int N = (int) adj_.size();
      std::vector<Dist> dist(N);
      std::vector<int> parent_idx(N, -1);

      dist[root_idx] = Dist{0, 0};
      parent_idx[root_idx] = root_idx;

      // priority queue ordered by (sift, hops)
      struct QItem {
        int sift, hops, v;
      };
      auto cmp = [](const QItem &a, const QItem &b) {
        if (a.sift != b.sift) return a.sift > b.sift;
        if (a.hops != b.hops) return a.hops > b.hops;
        return a.v > b.v;
      };
      std::priority_queue<QItem, std::vector<QItem>, decltype(cmp)> pq(cmp);
      pq.push(QItem{0, 0, root_idx});

      while (!pq.empty()) {
        auto cur = pq.top();
        pq.pop();
        const int u = cur.v;
        if (cur.sift != dist[u].sift || cur.hops != dist[u].hops) continue;

        for (const auto &nb: adj_[u]) {
          const int v = nb.v;
          if (!final_member[v]) continue;

          const bool has_orb = (nb.mask_bits & 0x1u) != 0;
          const bool has_sift = (nb.mask_bits & 0x2u) != 0;
          if (!has_orb && !has_sift) continue; // shouldn't happen, but safe

          // Prefer ORB whenever available; only count SIFT when it's SIFT-only.
          const int add_sift = (!has_orb && has_sift) ? 1 : 0;

          Dist nd{dist[u].sift + add_sift, dist[u].hops + 1};
          if (better(nd, dist[v])) {
            dist[v] = nd;
            parent_idx[v] = u;
            pq.push(QItem{nd.sift, nd.hops, v});
          }
        }
      }

      // Build parent map; check reachability
      bool all_reached = true;
      for (int i = 0; i < N; ++i) {
        if (!final_member[i]) continue;
        if (parent_idx[i] == -1) {
          all_reached = false;
          continue;
        }

        const ImgId id = idx_to_id_[i];
        const ImgId pid = idx_to_id_[parent_idx[i]];
        out_parent[id] = pid;
      }
      out_parent[out_root] = out_root;

      return all_reached;
    }

    // Reconstruct path [root ... node] from parent map (root points to itself).
    static std::vector<ImgId> reconstructPathToRoot(
      ImgId node,
      ImgId root,
      const std::unordered_map<ImgId, ImgId> &parent) {
      std::vector<ImgId> path;
      auto it = parent.find(node);
      if (it == parent.end()) return path;

      ImgId cur = node;
      while (true) {
        path.push_back(cur);
        if (cur == root) break;

        auto jt = parent.find(cur);
        if (jt == parent.end()) {
          path.clear();
          return path;
        }
        ImgId p = jt->second;
        if (p == cur) {
          path.clear();
          return path;
        } // cycle / bad parent map
        cur = p;
      }
      std::reverse(path.begin(), path.end());
      return path;
    }

  private:
    // Node storage
    std::unordered_map<ImgId, int> id_to_idx_;
    std::vector<ImgId> idx_to_id_;

    // Per-node membership
    std::vector<bool> member_;

    // Adjacency; each neighbor stores which kinds exist for the unordered pair
    std::vector<std::vector<Neighbor> > adj_;

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

  namespace poly_union_envelope {

using Clipper2Lib::Point64;
using Clipper2Lib::Path64;
using Clipper2Lib::Paths64;

// ------------------------- helpers -------------------------

static inline int64_t packXY32(int x, int y) {
  return (int64_t(uint32_t(x)) << 32) | uint32_t(y);
}

static inline int64_t cross_ll(const cv::Point2i& a, const cv::Point2i& b) {
  return int64_t(a.x) * int64_t(b.y) - int64_t(a.y) * int64_t(b.x);
}

// Signed area*2. In standard Cartesian coords, + => CCW.
// (In image coords with y down, sign flips, but we only use it for consistency.)
static inline int64_t signedArea2(const std::vector<cv::Point2i>& p) {
  int64_t a2 = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const auto& A = p[i];
    const auto& B = p[(i + 1) % p.size()];
    a2 += int64_t(A.x) * int64_t(B.y) - int64_t(B.x) * int64_t(A.y);
  }
  return a2;
}

static inline void ensureClockwise(std::vector<cv::Point2i>& poly) {
  if (poly.size() < 3) return;
  // If positive area => CCW (cartesian). Reverse to CW.
  if (signedArea2(poly) > 0) std::reverse(poly.begin(), poly.end());
}

static inline void removeDuplicateConsecutive(std::vector<cv::Point2i>& poly) {
  poly.erase(std::unique(poly.begin(), poly.end(),
                         [](const cv::Point2i& a, const cv::Point2i& b){ return a == b; }),
             poly.end());
  // Also remove closing duplicate if present
  if (poly.size() >= 2 && poly.front() == poly.back()) poly.pop_back();
}

static inline void removeCollinear(std::vector<cv::Point2i>& poly) {
  if (poly.size() < 4) return;
  std::vector<cv::Point2i> out;
  out.reserve(poly.size());
  auto sgn = [](int v) { return (v > 0) - (v < 0); };

  const size_t n = poly.size();
  for (size_t i = 0; i < n; ++i) {
    const auto& prev = poly[(i + n - 1) % n];
    const auto& cur  = poly[i];
    const auto& next = poly[(i + 1) % n];

    int dx1 = sgn(cur.x - prev.x), dy1 = sgn(cur.y - prev.y);
    int dx2 = sgn(next.x - cur.x), dy2 = sgn(next.y - cur.y);

    if (dx1 == dx2 && dy1 == dy2) continue;
    out.push_back(cur);
  }
  poly.swap(out);
}

// Segment intersection (including touching) using int64 arithmetic.
static inline bool segIntersects(const cv::Point2i& a, const cv::Point2i& b,
                                 const cv::Point2i& c, const cv::Point2i& d) {
  auto orient = [](const cv::Point2i& p, const cv::Point2i& q, const cv::Point2i& r) -> int64_t {
    return int64_t(q.x - p.x) * int64_t(r.y - p.y) - int64_t(q.y - p.y) * int64_t(r.x - p.x);
  };
  auto onSeg = [](const cv::Point2i& p, const cv::Point2i& q, const cv::Point2i& r) -> bool {
    // q on segment pr
    return std::min(p.x, r.x) <= q.x && q.x <= std::max(p.x, r.x) &&
           std::min(p.y, r.y) <= q.y && q.y <= std::max(p.y, r.y);
  };

  int64_t o1 = orient(a, b, c);
  int64_t o2 = orient(a, b, d);
  int64_t o3 = orient(c, d, a);
  int64_t o4 = orient(c, d, b);

  if ((o1 > 0 && o2 < 0 || o1 < 0 && o2 > 0) &&
      (o3 > 0 && o4 < 0 || o3 < 0 && o4 > 0)) return true;

  if (o1 == 0 && onSeg(a, c, b)) return true;
  if (o2 == 0 && onSeg(a, d, b)) return true;
  if (o3 == 0 && onSeg(c, a, d)) return true;
  if (o4 == 0 && onSeg(c, b, d)) return true;

  return false;
}

// Point in convex polygon (CW or CCW). Treat boundary as inside.
static inline bool pointInConvexPoly(const cv::Point2i& p, const std::vector<cv::Point2i>& poly) {
  if (poly.size() < 3) return false;

  // Use half-plane tests. Works for convex polygon either orientation if we track sign.
  int64_t prev = 0;
  for (size_t i = 0; i < poly.size(); ++i) {
    const auto& a = poly[i];
    const auto& b = poly[(i + 1) % poly.size()];
    cv::Point2i ab{b.x - a.x, b.y - a.y};
    cv::Point2i ap{p.x - a.x, p.y - a.y};
    int64_t c = cross_ll(ab, ap);
    if (c == 0) continue;
    if (prev == 0) prev = c;
    else if ((prev > 0) != (c > 0)) return false;
  }
  return true;
}

// Conservative legality: reject chord if it passes through the interior of ANY input polygon.
// Implementation: if the segment intersects any polygon edge at a non-endpoint, or if the
// midpoint lies inside the polygon. Touching polygon boundaries is allowed.
static bool chordIsLegal(const cv::Point2i& u,
                         const cv::Point2i& v,
                         const std::vector<std::vector<cv::Point2i>>& inputPolys)
{
  if (u == v) return false;
  // Midpoint test to catch "goes through interior" even if it doesn't cross edges (rare but possible).
  // Use integer midpoint (floor); good enough for conservative test.
  cv::Point2i mid{ (u.x + v.x) / 2, (u.y + v.y) / 2 };

  for (const auto& poly : inputPolys) {
    if (poly.size() < 3) continue;

    // If midpoint is inside, chord goes through filled region (unless chord lies exactly on boundary,
    // but then midpoint might still be on boundary; we treat boundary as inside -> conservative).
    if (pointInConvexPoly(mid, poly)) {
      // Allow the case where chord is exactly an existing polygon edge:
      // If u-v is an edge of this poly, accept.
      bool isEdge = false;
      for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];
        if ((a == u && b == v) || (a == v && b == u)) { isEdge = true; break; }
      }
      if (!isEdge) return false;
    }

    // Edge intersection test: disallow proper crossings with polygon edges
    for (size_t i = 0; i < poly.size(); ++i) {
      const auto& a = poly[i];
      const auto& b = poly[(i + 1) % poly.size()];

      // If intersection is only at shared endpoints, that's fine.
      if ((a == u || a == v || b == u || b == v)) continue;

      if (segIntersects(u, v, a, b)) return false;
    }
  }
  return true;
}

// Convert cv::Point2i polygon -> Clipper Path64
static Path64 toPath64(const std::vector<cv::Point2i>& poly) {
  Path64 p;
  p.reserve(poly.size());
  for (const auto& pt : poly) p.push_back(Point64(pt.x, pt.y));
  return p;
}

// Convert Clipper Path64 -> cv::Point2i (dropping 64->32 assuming it fits)
static std::vector<cv::Point2i> fromPath64(const Path64& p) {
  std::vector<cv::Point2i> out;
  out.reserve(p.size());
  for (const auto& pt : p) out.push_back(cv::Point2i((int)pt.x, (int)pt.y));
  return out;
}

// Choose the outer boundary path from union result.
// If union returns multiple outer polygons, pick the one with largest absolute area.
static Path64 pickLargestOuter(const Paths64& sol) {
  if (sol.empty()) return {};
  auto area = [](const Path64& p) -> double { return Clipper2Lib::Area(p); };
  size_t best = 0;
  double bestAbs = std::abs(area(sol[0]));
  for (size_t i = 1; i < sol.size(); ++i) {
    double a = std::abs(area(sol[i]));
    if (a > bestAbs) { bestAbs = a; best = i; }
  }
  return sol[best];
}

// ------------------------- main function -------------------------

std::vector<cv::Point2i> union_boundary_then_chord_simplify_CW(const std::vector<std::vector<cv::Point2i>>& polysCW);

} // namespace poly_union_envelope
}

#endif /* util_h */
