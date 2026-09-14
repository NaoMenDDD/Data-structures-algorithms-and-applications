// 11_dijkstra.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 11 章 图算法》
// 主题：Dijkstra 单源最短路 —— 贪心 + 优先队列 = 导航软件的引擎
//
//   问题：给定带权有向图（边权非负）和一个源点 s，求 s 到其余每点的最短距离。
//   第 10 章的 BFS 只能处理「每条边权重相同」；一旦边上带权，就要 Dijkstra。
//
//   Dijkstra 的思想（贪心）：
//     维护「已确定最短路的集合 S」。每次从未确定的点里挑出**当前距离最小**的
//     那个 u（它此时的 dist[u] 已是最终答案），把它并入 S，然后用 u 去**松弛**
//     （relax）它的所有出边：若 dist[u] + w < dist[v]，就更新 dist[v]。
//
//   为什么贪心是对的？因为边权非负 —— 一旦 dist[u] 是所有人里最小的，
//   「绕远路」只会更大，不可能再找到通往 u 的更短路径。
//   ★ 这条正确性**依赖「非负边权」**。有负权边时贪心失效，要用 Bellman-Ford。
//
//   两种实现：
//     * 朴素 O(V^2)：每轮线性扫未定点找最小 —— 稠密图（或 V 小）够用；
//     * 堆优化 O((V+E) log V)：用优先队列取最小 —— 稀疏图（默认选择）。
//
//   本文件还把 Dijkstra 与 Bellman-Ford 对拍（非负权图上二者结果必须一致）。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 11_dijkstra.cpp -o 11_dijkstra && ./11_dijkstra
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 11_dijkstra.cpp -o dj_san && ./dj_san
// ---------------------------------------------------------------------------

#include <algorithm>      // std::reverse
#include <cassert>
#include <chrono>
#include <cstddef>
#include <functional>     // std::greater
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

static constexpr int INF = std::numeric_limits<int>::max() / 4;   // 防加法溢出

using Adj = std::vector<std::vector<std::pair<int,int>>>;          // adj[u] = {(v, w)}

// ---------------------------------------------------------------------------
// 堆优化的 Dijkstra。返回 dist[]（不可达 = INF）和 parent[]（用于还原路径）。
//   用「惰性删除」：允许同一个点多次入堆，弹出时若它的 dist 已过时就跳过。
// ---------------------------------------------------------------------------
static void dijkstraHeap(int n, const Adj& adj, int src,
                         std::vector<int>& dist, std::vector<int>& parent) {
    dist.assign(static_cast<std::size_t>(n), INF);
    parent.assign(static_cast<std::size_t>(n), -1);
    using P = std::pair<int,int>;                                   // (dist, node)
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    dist[static_cast<std::size_t>(src)] = 0;
    pq.push({0, src});
    while (!pq.empty()) {
        const auto [d, u] = pq.top(); pq.pop();
        if (d > dist[static_cast<std::size_t>(u)]) continue;        // 过期条目，跳过
        for (auto [v, w] : adj[static_cast<std::size_t>(u)]) {
            const int nd = d + w;
            if (nd < dist[static_cast<std::size_t>(v)]) {           // 松弛成功
                dist[static_cast<std::size_t>(v)] = nd;
                parent[static_cast<std::size_t>(v)] = u;
                pq.push({nd, v});
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 朴素 O(V^2) 版 Dijkstra：每轮线性找「未确定点里 dist 最小者」。
//   稠密图（E ~ V^2）时反而比堆版快（没有堆的常数）。
// ---------------------------------------------------------------------------
static void dijkstraNaive(int n, const Adj& adj, int src, std::vector<int>& dist) {
    dist.assign(static_cast<std::size_t>(n), INF);
    std::vector<char> done(static_cast<std::size_t>(n), 0);
    dist[static_cast<std::size_t>(src)] = 0;
    for (int iter = 0; iter < n; ++iter) {
        int u = -1, best = INF;
        for (int i = 0; i < n; ++i)                                 // 线性找最小
            if (!done[static_cast<std::size_t>(i)] && dist[static_cast<std::size_t>(i)] < best) {
                best = dist[static_cast<std::size_t>(i)]; u = i;
            }
        if (u == -1) break;                                         // 剩下的都不可达
        done[static_cast<std::size_t>(u)] = 1;
        for (auto [v, w] : adj[static_cast<std::size_t>(u)])
            if (dist[static_cast<std::size_t>(u)] + w < dist[static_cast<std::size_t>(v)])
                dist[static_cast<std::size_t>(v)] = dist[static_cast<std::size_t>(u)] + w;
    }
}

// ---------------------------------------------------------------------------
// Bellman-Ford（O(VE)）：能处理负权，还能检测负环。这里用来给 Dijkstra 当基准。
//   返回 false 表示存在从 src 可达的负权环（此时最短路无定义）。
// ---------------------------------------------------------------------------
static bool bellmanFord(int n, const std::vector<std::tuple<int,int,int>>& edges, int src,
                        std::vector<int>& dist) {
    dist.assign(static_cast<std::size_t>(n), INF);
    dist[static_cast<std::size_t>(src)] = 0;
    for (int i = 0; i < n - 1; ++i) {
        bool changed = false;
        for (auto [u, v, w] : edges) {
            if (dist[static_cast<std::size_t>(u)] == INF) continue;
            if (dist[static_cast<std::size_t>(u)] + w < dist[static_cast<std::size_t>(v)]) {
                dist[static_cast<std::size_t>(v)] = dist[static_cast<std::size_t>(u)] + w;
                changed = true;
            }
        }
        if (!changed) break;                                        // 提前收敛
    }
    for (auto [u, v, w] : edges) {                                  // 第 n 轮还能松弛 -> 负环
        if (dist[static_cast<std::size_t>(u)] != INF &&
            dist[static_cast<std::size_t>(u)] + w < dist[static_cast<std::size_t>(v)])
            return false;
    }
    return true;
}

static std::vector<int> buildPath(const std::vector<int>& parent, int target) {
    std::vector<int> path;
    for (int v = target; v != -1; v = parent[static_cast<std::size_t>(v)]) path.push_back(v);
    std::reverse(path.begin(), path.end());
    return path;
}

int main() {
    std::cout << "======== 11 Dijkstra 单源最短路 ========\n\n";

    // ---------- 1. 直观例子 ----------
    std::cout << "=== 1. 一个小路网（有向、非负权）===\n";
    {
        //  0 --4--> 1 --1--> 2
        //  |        ^        |
        //  1        2        5
        //  v        |        v
        //  3 --8--> 4 <--2-- 5 ...
        // 下面直接给一组能体现「绕路更短」的边
        const int n = 6;
        Adj adj(static_cast<std::size_t>(n));
        auto add = [&](int u, int v, int w) { adj[static_cast<std::size_t>(u)].push_back({v, w}); };
        add(0, 1, 7); add(0, 2, 9); add(0, 5, 14);
        add(1, 2, 10); add(1, 3, 15);
        add(2, 3, 11); add(2, 5, 2);
        add(3, 4, 6);
        add(5, 4, 9);

        std::vector<int> dist, parent;
        dijkstraHeap(n, adj, 0, dist, parent);
        std::cout << "  从 0 出发到各点最短路：\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "     到 " << i << " : " << dist[static_cast<std::size_t>(i)] << "   路径 ";
            for (int v : buildPath(parent, i)) std::cout << v << (v == i ? "" : " -> ");
            std::cout << "\n";
        }
        assert((dist == std::vector<int>{0, 7, 9, 20, 20, 11}));
        std::cout << "  -> 到 4 的最短路是 0->2->5->4 = 9+2+9 = 20，比直达 0->...->3->4 短 ✓\n";
    }

    // ---------- 2. 堆版 == 朴素版 == Bellman-Ford ----------
    std::cout << "\n=== 2. ★ 三算法对拍（随机非负权图）===\n";
    {
        std::mt19937 rng(20260913);
        for (int trial = 0; trial < 300; ++trial) {
            const int n = 2 + static_cast<int>(rng() % 40);
            Adj adj(static_cast<std::size_t>(n));
            std::vector<std::tuple<int,int,int>> edges;
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < n; ++v)
                    if (u != v && rng() % 5 == 0) {                  // 约 20% 密度
                        const int w = 1 + static_cast<int>(rng() % 20);
                        adj[static_cast<std::size_t>(u)].push_back({v, w});
                        edges.emplace_back(u, v, w);
                    }
            const int src = static_cast<int>(rng() % n);
            std::vector<int> dh, dn, dbf, par;
            dijkstraHeap(n, adj, src, dh, par);
            dijkstraNaive(n, adj, src, dn);
            assert(bellmanFord(n, edges, src, dbf));                 // 非负权无负环
            assert(dh == dn && dn == dbf);                           // 三者必须完全一致
        }
        std::cout << "  300 个随机非负权图：堆版 == 朴素版 == Bellman-Ford ✓\n";
    }

    // ---------- 3. ★ 负权边让 Dijkstra 出错 ----------
    std::cout << "\n=== 3. ★ 为什么 Dijkstra 要求「非负权」===\n";
    {
        // 0->1(1), 0->2(2), 2->1(-2), 1->3(1)
        // 真实：0->2->1 = 2-2 = 0，再到 3 = 1。
        // 但朴素 Dijkstra 会先「确定」1（dist 1），之后即使发现 0 更短，
        // done[1] 已置位 -> 不再用 1 松弛 3 -> dist[3] 永远停在 2（错）。
        const int n = 4;
        Adj adj(static_cast<std::size_t>(n));
        adj[0].push_back({1, 1});
        adj[0].push_back({2, 2});
        adj[2].push_back({1, -2});
        adj[1].push_back({3, 1});
        std::vector<int> d1, d2;
        dijkstraNaive(n, adj, 0, d1);                            // 朴素版：有 done[] 永久标记
        std::vector<std::tuple<int,int,int>> edges = {{0,1,1},{0,2,2},{2,1,-2},{1,3,1}};
        bellmanFord(n, edges, 0, d2);
        std::cout << "  朴素 Dijkstra  到 3 的距离 = " << d1[3] << "（错！1 被过早「确定」，没再松弛 3）\n";
        std::cout << "  Bellman-Ford   到 3 的距离 = " << d2[3] << "（对：0->2->1->3 = 0+1 = 1）\n";
        assert(d1[3] == 2 && d2[3] == 1);
        std::cout << "  -> 负权边让「已确定」的点事后反悔，贪心失效；此时必须用 Bellman-Ford。\n";
    }

    // ---------- 4. 复杂度实测（堆 vs 朴素，稀疏 vs 稠密）----------
    std::cout << "\n=== 4. 复杂度实测：堆版 vs 朴素版 ===\n";
    auto makeGraph = [](int n, int deg, std::mt19937& rng) {
        Adj adj(static_cast<std::size_t>(n));
        for (int u = 0; u < n; ++u)
            for (int k = 0; k < deg; ++k) {
                const int v = static_cast<int>(rng() % static_cast<unsigned>(n));
                if (v != u) adj[static_cast<std::size_t>(u)].push_back({v, 1 + static_cast<int>(rng() % 100)});
            }
        return adj;
    };
    std::cout << "     V       稀疏(deg=3)堆ms  稀疏朴素ms   稠密(deg=V/4)堆ms  稠密朴素ms\n";
    for (int V : {20000, 50000}) {
        std::mt19937 rng(7);
        Adj sparse = makeGraph(V, 3, rng);                    // E ≈ 3V
        Adj dense  = makeGraph(V, V / 4, rng);                // E ≈ V^2/4

        std::vector<int> d, p;
        auto t0 = std::chrono::steady_clock::now(); dijkstraHeap(V, sparse, 0, d, p);
        auto t1 = std::chrono::steady_clock::now(); dijkstraNaive(V, sparse, 0, d);
        auto t2 = std::chrono::steady_clock::now(); dijkstraHeap(V, dense, 0, d, p);
        auto t3 = std::chrono::steady_clock::now(); dijkstraNaive(V, dense, 0, d);
        auto t4 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cout << "     " << V
                  << "         " << ms(t0,t1) << "            " << ms(t1,t2)
                  << "          " << ms(t2,t3) << "          " << ms(t3,t4) << "\n";
    }
    std::cout << "  -> 堆版在两处都更快：稀疏图只碰 O(E) 条边；稠密图里 O(V^2) 的朴素版\n"
                 "     随 V 平方增长（V=5万 -> 25 亿次操作），堆版 O(E log V) 仍占优。\n"
                 "     朴素版只在 V 很小、或配合 O(1) 判边的邻接矩阵时才有优势。\n";

    std::cout << "\n=== 5. 要点 ===\n"
                 "  * Dijkstra 只处理**非负权**；贪心的正确性建立在「边权 >= 0」上；\n"
                 "  * 堆版 O((V+E) log V)（稀疏图默认）；朴素版 O(V^2)（稠密图）；\n"
                 "  * 堆版用「惰性删除」：同一点可多次入堆，弹出时按 dist 判过期；\n"
                 "  * parent[] 可还原整条最短路；求「所有点对」用 Floyd（见另一文件）；\n"
                 "  * 有负权 -> Bellman-Ford O(VE)；有负环 -> 最短路无定义（可检测）。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
