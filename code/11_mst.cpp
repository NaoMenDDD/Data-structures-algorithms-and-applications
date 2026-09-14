// 11_mst.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 11 章 图算法》
// 主题：最小生成树（MST）—— Prim（贪心长树）+ Kruskal（贪心加边）
//
//   问题：给一张**连通无向带权图**，找一棵**生成树**（连通所有点、恰好 V-1 条边、
//   无环），使得**边权总和最小**。这就是「用最少的线把所有城市连起来」
//   （铺电网、修路、聚类、图像分割都用它）。
//
//   两个经典贪心算法，贪心策略不同但都正确：
//
//   ① Prim（长树）：从任一点出发，每次把「连接已选集合与外部的最小边」加进来。
//        实现像 Dijkstra——用优先队列维护「到已选集合的最小边权」。
//        复杂度 O(E log V)。适合**稠密图**。
//
//   ② Kruskal（加边）：把所有边按权从小到大排序，依次尝试加入，
//        若两端已连通（用并查集判断）就跳过，否则加入。
//        复杂度 O(E log E)。适合**稀疏图**。
//
//   为什么贪心对？核心是**切分性质（cut property）**：
//     对任意把点分成两半的「切」，**横跨这个切的最小边一定在某个 MST 里**。
//     Prim 每次加的就是当前切的最小横跨边；Kruskal 每次加的是「不产生环」的
//     全局最小边（等价于某个切的最小横跨边）。两个算法都只是「实现这条性质」。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 11_mst.cpp -o 11_mst && ./11_mst
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 11_mst.cpp -o mst_san && ./mst_san
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>     // std::greater
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// 并查集（Disjoint Set Union）：Kruskal 用它判「两端是否已连通」。
//   路径压缩 + 按秩合并，均摊近 O(1)。（详见 11_union_find.cpp）
// ---------------------------------------------------------------------------
class DSU {
public:
    explicit DSU(int n) : parent_(static_cast<std::size_t>(n)), rank_(static_cast<std::size_t>(n), 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }
    int find(int x) {
        while (parent_[static_cast<std::size_t>(x)] != x) {              // 路径压缩（迭代版）
            parent_[static_cast<std::size_t>(x)] = parent_[static_cast<std::size_t>(parent_[static_cast<std::size_t>(x)])];
            x = parent_[static_cast<std::size_t>(x)];
        }
        return x;
    }
    bool unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;                                       // 已连通
        if (rank_[static_cast<std::size_t>(ra)] < rank_[static_cast<std::size_t>(rb)]) std::swap(ra, rb);
        parent_[static_cast<std::size_t>(rb)] = ra;                       // 矮树挂到高树上
        if (rank_[static_cast<std::size_t>(ra)] == rank_[static_cast<std::size_t>(rb)]) ++rank_[static_cast<std::size_t>(ra)];
        return true;
    }
private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

struct Edge { int u, v, w; };

// ---------------------------------------------------------------------------
// Kruskal：按边权排序 + 并查集。返回 MST 的总权与所选的边。
// ---------------------------------------------------------------------------
static long long kruskalMST(int n, std::vector<Edge> edges, std::vector<Edge>& chosen) {
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.w < b.w; });
    DSU dsu(n);
    long long total = 0;
    chosen.clear();
    for (const Edge& e : edges) {
        if (dsu.unite(e.u, e.v)) {                                        // 不成环才加
            total += e.w;
            chosen.push_back(e);
            if (static_cast<int>(chosen.size()) == n - 1) break;          // 够了就停
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// Prim：从 start 出发「长树」。用优先队列维护「到已选集合的最小边」。
//   惰性删除：允许过期的边留在堆里，弹出时检查另一端是否已入树。
// ---------------------------------------------------------------------------
static long long primMST(int n, const std::vector<std::vector<std::pair<int,int>>>& adj,
                         int start, std::vector<Edge>& chosen) {
    std::vector<char> inMST(static_cast<std::size_t>(n), 0);
    using P = std::tuple<int,int,int>;                                    // (weight, to, from)
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    inMST[static_cast<std::size_t>(start)] = 1;
    for (auto [v, w] : adj[static_cast<std::size_t>(start)]) pq.push({w, v, start});

    long long total = 0;
    chosen.clear();
    while (!pq.empty()) {
        auto [w, v, u] = pq.top(); pq.pop();
        if (inMST[static_cast<std::size_t>(v)]) continue;                 // 过期边，跳过
        inMST[static_cast<std::size_t>(v)] = 1;
        total += w;
        chosen.push_back({u, v, w});
        for (auto [x, wx] : adj[static_cast<std::size_t>(v)])
            if (!inMST[static_cast<std::size_t>(x)]) pq.push({wx, x, v});
    }
    return total;
}

int main() {
    std::cout << "======== 11 最小生成树（Prim / Kruskal）========\n\n";

    // ---------- 1. 一个直观的小例子 ----------
    std::cout << "=== 1. 经典例子（6 个点）===\n";
    {
        const int n = 6;
        std::vector<Edge> edges = {
            {0,1,4},{0,2,4},{1,2,2},{1,3,6},{2,3,8},{3,4,9},{2,5,7},{4,5,11},{3,5,3},{1,4,5}
        };
        std::vector<Edge> kEdges;
        const long long kTotal = kruskalMST(n, edges, kEdges);
        std::vector<std::vector<std::pair<int,int>>> adj(static_cast<std::size_t>(n));
        for (const Edge& e : edges) { adj[e.u].push_back({e.v, e.w}); adj[e.v].push_back({e.u, e.w}); }
        std::vector<Edge> pEdges;
        const long long pTotal = primMST(n, adj, 0, pEdges);

        std::cout << "  Kruskal 选的边（按权升序）：\n";
        for (const Edge& e : kEdges) std::cout << "     " << e.u << " - " << e.v << "  权 " << e.w << "\n";
        std::cout << "  Kruskal 总权 = " << kTotal << "\n";
        std::cout << "  Prim    总权 = " << pTotal << "（应与 Kruskal 相同）\n";
        assert(kTotal == pTotal && kTotal == 20);
        assert(static_cast<int>(kEdges.size()) == n - 1 && static_cast<int>(pEdges.size()) == n - 1);
        std::cout << "  -> 两算法边数都 = V-1 = 5，总权都 = 20（=2+3+4+5+6）✓（MST 可能不唯一，但总权唯一）\n";
    }

    // ---------- 2. ★ 对拍：随机连通图上 Prim == Kruskal ----------
    std::cout << "\n=== 2. ★ 随机连通图：Prim 总权 == Kruskal 总权 ===\n";
    {
        std::mt19937 rng(20260913);
        int tested = 0;
        for (int trial = 0; trial < 500; ++trial) {
            const int n = 2 + static_cast<int>(rng() % 60);
            std::vector<Edge> edges;
            // 先连一棵随机树保证连通，再加若干随机边
            for (int v = 1; v < n; ++v) {
                const int u = static_cast<int>(rng() % static_cast<unsigned>(v));
                edges.push_back({u, v, 1 + static_cast<int>(rng() % 100)});
            }
            const int extra = static_cast<int>(rng() % (n * 2 + 1));
            for (int i = 0; i < extra; ++i) {
                int u = static_cast<int>(rng() % static_cast<unsigned>(n));
                int v = static_cast<int>(rng() % static_cast<unsigned>(n));
                if (u == v) continue;
                edges.push_back({u, v, 1 + static_cast<int>(rng() % 100)});
            }
            std::vector<std::vector<std::pair<int,int>>> adj(static_cast<std::size_t>(n));
            for (const Edge& e : edges) { adj[e.u].push_back({e.v, e.w}); adj[e.v].push_back({e.u, e.w}); }

            std::vector<Edge> ke, pe;
            const long long k = kruskalMST(n, edges, ke);
            const long long p = primMST(n, adj, 0, pe);
            assert(k == p);                                              // 总权必须一致
            assert(static_cast<int>(ke.size()) == n - 1);                // 恰好 V-1 条边
            // Prim 边数（连通图）也应为 V-1
            assert(static_cast<int>(pe.size()) == n - 1);
            ++tested;
        }
        std::cout << "  " << tested << " 个随机连通图：两算法总权完全一致、边数都是 V-1 ✓\n";
    }

    // ---------- 3. ★ 验证「生成树」：连通且无环（用并查集重查）----------
    std::cout << "\n=== 3. ★ 校验结果确实是一棵「树」===\n";
    {
        const int n = 6;
        std::vector<Edge> edges = {{0,1,1},{1,2,2},{2,3,3},{3,4,4},{4,5,5},{0,5,100},{1,4,100},{2,5,100}};
        std::vector<Edge> chosen;
        kruskalMST(n, edges, chosen);
        DSU dsu(n);
        bool acyclic = true;
        for (const Edge& e : chosen) if (!dsu.unite(e.u, e.v)) acyclic = false;    // 任何一条成环则该假
        int comps = 0;
        for (int i = 0; i < n; ++i) if (dsu.find(i) == i) ++comps;
        std::cout << "  选出的 " << chosen.size() << " 条边：无环 = " << acyclic
                  << "，连通分量 = " << comps << "（都应为真 / 1）\n";
        assert(acyclic && comps == 1);
        std::cout << "  -> V-1 条边 + 无环 + 连通 == 一棵生成树 ✓\n";
    }

    std::cout << "\n=== 4. 复杂度与选择 ===\n"
                 "  * Kruskal：排序 O(E log E) + 并查集近 O(E)；稀疏图 / 只给边表时首选；\n"
                 "  * Prim（堆版）：O(E log V)；稠密图 / 邻接矩阵时更顺手；\n"
                 "  * 二者都基于**切分性质**：横跨任一切的最小边必在某 MST 里；\n"
                 "  * MST 一般不唯一（等权边可互换），但**总权一定唯一**；\n"
                 "  * 求「最大生成树」把权取负再跑即可；「次小生成树」用 MST + 换边。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
