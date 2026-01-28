//
// Created by cooper maira on 1/28/26.
//
#include "pathCam.h"
namespace pathCam {
    void ImageGraph::PromoteMembersForOverlapConnectivityShortestHop(
        std::unordered_set<Image*>& members,
        const std::vector<std::pair<Image*, Image*>>& overlaps,
        const std::vector<std::shared_ptr<Match>>& allRegistrations)
    {
        using Img = Image*;

        // ---------- helpers ----------
        struct PairHash {
            size_t operator()(const std::pair<Img, Img>& p) const noexcept {
                auto a = reinterpret_cast<std::uintptr_t>(p.first);
                auto b = reinterpret_cast<std::uintptr_t>(p.second);
                // A simple, decent combine
                size_t h1 = std::hash<std::uintptr_t>{}(a);
                size_t h2 = std::hash<std::uintptr_t>{}(b);
                return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
            }
        };
        auto canon = [](Img a, Img b) -> std::pair<Img, Img> {
            return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
        };

        // Lexicographic DP state: (promotions, nonMemberCount, hops)
        // hops is implicit by layer; we keep it for clarity/stale checks.
        struct State {
            int promos = std::numeric_limits<int>::max();
            int nonMembers = std::numeric_limits<int>::max();
            int hops = std::numeric_limits<int>::max();
        };
        auto better = [](const State& a, const State& b) -> bool {
            if (a.promos != b.promos) return a.promos < b.promos;
            if (a.nonMembers != b.nonMembers) return a.nonMembers < b.nonMembers;
            return a.hops < b.hops;
        };

        // ---------- build undirected adjacency from registrations ----------
        std::unordered_map<Img, std::vector<Img>> adj;
        adj.reserve(allRegistrations.size() * 2);

        std::unordered_set<std::pair<Img, Img>, PairHash> directEdge;
        directEdge.reserve(allRegistrations.size() * 2);

        for (const auto& msp : allRegistrations) {
            if (!msp) continue;
            Img a = msp->image_1;
            Img b = msp->image_2;
            if (!a || !b || a == b) continue;

            adj[a].push_back(b);
            adj[b].push_back(a);

            directEdge.insert(canon(a, b));
        }

        // ---------- BFS cache per source ----------
        struct DistCacheEntry {
            // dist[node] = hop distance from src
            std::unordered_map<Img, int> dist;
            // buckets[level] = nodes at that hop distance (filled only up to D used per query)
            // Not cached since D differs; we can build per query cheaply.
        };

        std::unordered_map<Img, DistCacheEntry> bfsCache;
        bfsCache.reserve(members.size());

        auto getDistMap = [&](Img src) -> std::unordered_map<Img, int>& {
            auto it = bfsCache.find(src);
            if (it != bfsCache.end()) return it->second.dist;

            DistCacheEntry entry;
            entry.dist.reserve(adj.size() / 2 + 16);

            std::deque<Img> q;
            entry.dist[src] = 0;
            q.push_back(src);

            while (!q.empty()) {
                Img u = q.front();
                q.pop_front();
                int du = entry.dist[u];

                auto aIt = adj.find(u);
                if (aIt == adj.end()) continue;

                for (Img v : aIt->second) {
                    if (!v) continue;
                    if (entry.dist.find(v) != entry.dist.end()) continue;
                    entry.dist[v] = du + 1;
                    q.push_back(v);
                }
            }

            auto [insIt, _] = bfsCache.emplace(src, std::move(entry));
            return insIt->second.dist;
        };

        // Track nodes we promoted during this call (subset of members)
        std::unordered_set<Img> promoted;
        promoted.reserve(256);

        auto isMember = [&](Img n) -> bool { return members.count(n) != 0; };
        auto isActive = [&](Img n) -> bool { return members.count(n) != 0; }; // members already includes promoted as we insert

        // DP for a single pair: shortest-hop only; minimize promotions, then nonMembers, then (hops = D)
        auto choosePathAndPromote = [&](Img src, Img dst, int D, const std::unordered_map<Img, int>& dist) {
            // Build level buckets up to D
            std::vector<std::vector<Img>> level(D + 1);
            level[0].push_back(src);
            for (const auto& [node, dn] : dist) {
                if (dn > 0 && dn <= D) level[dn].push_back(node);
            }

            std::unordered_map<Img, State> bestState;
            std::unordered_map<Img, Img> parent;
            bestState.reserve(level.size() * 16);
            parent.reserve(level.size() * 16);

            bestState[src] = {0, 0, 0};
            parent[src] = nullptr;

            // Layered DP on shortest-hop DAG
            for (int l = 0; l < D; ++l) {
                for (Img u : level[l]) {
                    auto sIt = bestState.find(u);
                    if (sIt == bestState.end()) continue;
                    const State cur = sIt->second;

                    auto aIt = adj.find(u);
                    if (aIt == adj.end()) continue;

                    for (Img v : aIt->second) {
                        if (!v) continue;
                        auto dIt = dist.find(v);
                        if (dIt == dist.end()) continue;
                        if (dIt->second != l + 1) continue; // shortest-hop forward edge

                        State nxt = cur;
                        nxt.hops = l + 1;

                        // Promotion objective: only counts if v is internal and not active
                        if (v != dst && v != src && !isActive(v)) {
                            nxt.promos += 1;
                        }

                        // Tie-break objective: prefer staying within existing members (counts any non-member internal node)
                        // Note: isMember() uses current members set (including already-promoted nodes).
                        if (v != dst && v != src && !isMember(v)) {
                            nxt.nonMembers += 1;
                        }

                        auto bIt = bestState.find(v);
                        if (bIt == bestState.end() || better(nxt, bIt->second)) {
                            bestState[v] = nxt;
                            parent[v] = u;
                        }
                    }
                }
            }

            if (bestState.find(dst) == bestState.end()) return; // no shortest-hop path (shouldn't happen if D known)

            // Reconstruct path
            std::vector<Img> path;
            for (Img cur = dst; cur != nullptr; cur = parent[cur]) path.push_back(cur);
            std::reverse(path.begin(), path.end());

            // Promote internal nodes that were not already members
            for (size_t i = 1; i + 1 < path.size(); ++i) {
                Img n = path[i];
                if (!isMember(n)) {
                    members.insert(n);
                    promoted.insert(n);
                }
            }
        };

        // ---------- prepare overlap items, compute hop distance, order by decreasing D ----------
        struct Item { Img u; Img v; int d; };
        std::vector<Item> items;
        items.reserve(overlaps.size());

        for (const auto& [u, v] : overlaps) {
            if (!u || !v || u == v) continue;
            // both are declared members by contract; we won't enforce but it's fine.
            // Skip if direct registration exists
            if (directEdge.count(canon(u, v))) continue;

            auto& dist = getDistMap(u);
            auto it = dist.find(v);
            if (it == dist.end()) continue; // disconnected in full reg graph -> can't satisfy via promotions
            items.push_back({u, v, it->second});
        }

        std::sort(items.begin(), items.end(),
                  [](const Item& a, const Item& b) { return a.d > b.d; });

        // ---------- main pass ----------
        for (const auto& it : items) {
            Img u = it.u;
            Img v = it.v;
            int D = it.d;

            // Distances cached by source u; note: D computed already from that cache
            const auto& dist = bfsCache.find(u)->second.dist;

            choosePathAndPromote(u, v, D, dist);
        }
    }
}