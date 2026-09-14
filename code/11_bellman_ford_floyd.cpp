// 11_bellman_ford_floyd.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 11 章 图算法》
// 主题：Bellman-Ford（可负权、可检测负环）+ Floyd-Warshall（全源最短路）
//
//   ┌─ Bellman-Ford：单源最短路，允许负权边，还能检测「负环」。
//   │    直觉：一条最短路最多用 V-1 条边（否则重复经过某点 = 有环）。
//   │    于是「把所有边松弛 V-1 轮」，dist[] 一定收敛到正确答案。
//   │    第 V 轮若还能松弛 => 存在可达的负环（最短路无定义，可无限变小）。
//   │    复杂度 O(V*E)（比 Dijkstra 慢，但换来「负权」+「负环检测」）。
//   │
//   └─ Floyd-Warshall：全源最短路（所有点对），允许负权。
//        经典的「三维 DP 压成二维」：
//          dp[k][i][j] = 只允许经过前 k 个点作为中转时，i->j 的最短路。
//          转移：dp[k][i][j] = min(dp[k-1][i][j], dp[k-1][i][k] + dp[k-1][k][j])
//          —— 「要么不经过 k，要么 i->k->j」。
//        把 k 提到最外层，二维数组原地更新即可。复杂度 O(V^3)。
//        副产品：若 floyd[i][i] < 0，则 i 在一个负环上。
//
//   为什么 Floyd 的 k 必须放最外层？因为 dp 依赖「上一轮 k-1」的完整结果；
//   把 k 放里层会用到本轮还没算完的状态，结果就错了。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 11_bellman_ford_floyd.cpp -o 11_bff && ./11_bff
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 11_bellman_ford_floyd.cpp -o bff_san && ./bff_san
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

static constexpr long long INF = std::numeric_limits<long long>::max() / 4;

struct Edge { int u, v, w; };

// ---------------------------------------------------------------------------
// Bellman-Ford。返回 true 表示「无（可达的）负环」，dist 有效；
// 返回 false 表示存在负环，此时 dist 不可信。
// ---------------------------------------------------------------------------
static bool bellmanFord(int n, const std::vector<Edge>& edges, int src, std::vector<long long>& dist) {
    dist.assign(static_cast<std::size_t>(n), INF);
    dist[static_cast<std::size_t>(src)] = 0;
    for (int round = 0; round < n - 1; ++round) {
        bool changed = false;
        for (const Edge& e : edges) {
            if (dist[static_cast<std::size_t>(e.u)] == INF) continue;
            if (dist[static_cast<std::size_t>(e.u)] + e.w < dist[static_cast<std::size_t>(e.v)]) {
                dist[static_cast<std::size_t>(e.v)] = dist[static_cast<std::size_t>(e.u)] + e.w;
                changed = true;
            }
        }
        if (!changed) break;                         // 提前收敛
    }
    for (const Edge& e : edges)                      // 第 n 轮还能松弛 => 有负环
        if (dist[static_cast<std::size_t>(e.u)] != INF &&
            dist[static_cast<std::size_t>(e.u)] + e.w < dist[static_cast<std::size_t>(e.v)])
            return false;
    return true;
}

// ---------------------------------------------------------------------------
// Floyd-Warshall：返回 V×V 的距离矩阵（不可达 = INF）。
//   dist[i][i] 初始为 0；若有自环或负环，可能变负。
// ---------------------------------------------------------------------------
static std::vector<std::vector<long long>> floydWarshall(int n, const std::vector<Edge>& edges) {
    std::vector<std::vector<long long>> d(static_cast<std::size_t>(n),
                                          std::vector<long long>(static_cast<std::size_t>(n), INF));
    for (int i = 0; i < n; ++i) d[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 0;
    for (const Edge& e : edges)
        d[static_cast<std::size_t>(e.u)][static_cast<std::size_t>(e.v)] =
            std::min(d[static_cast<std::size_t>(e.u)][static_cast<std::size_t>(e.v)], (long long)e.w);

    for (int k = 0; k < n; ++k)                       // ★ k 必须在最外层！
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (d[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] != INF &&
                    d[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] != INF)
                    d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                        std::min(d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                                 d[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] +
                                 d[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)]);
    return d;
}

// 传递闭包：只关心「可达 / 不可达」，把 Floyd 的 min 换成逻辑或。
static std::vector<std::vector<char>> transitiveClosure(int n, const std::vector<Edge>& edges) {
    std::vector<std::vector<char>> reach(static_cast<std::size_t>(n), std::vector<char>(static_cast<std::size_t>(n), 0));
    for (int i = 0; i < n; ++i) reach[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1;
    for (const Edge& e : edges) reach[static_cast<std::size_t>(e.u)][static_cast<std::size_t>(e.v)] = 1;
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (reach[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] &&
                    reach[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)])
                    reach[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 1;
    return reach;
}

int main() {
    std::cout << "======== 11 Bellman-Ford + Floyd-Warshall ========\n\n";

    // ---------- 1. Bellman-Ford 处理负权 ----------
    std::cout << "=== 1. Bellman-Ford：允许负权边 ===\n";
    {
        // 经典货币套汇 / 汇率图例子
        const int n = 5;
        std::vector<Edge> edges = {{0,1,6},{0,2,7},{1,2,8},{1,3,5},{1,4,-4},{2,3,-3},{2,4,9},{3,1,-2},{4,0,2},{4,3,7}};
        std::vector<long long> dist;
        const bool ok = bellmanFord(n, edges, 0, dist);
        assert(ok);
        std::cout << "  从 0 出发（含负权边）：";
        for (int i = 0; i < n; ++i) std::cout << " d(" << i << ")=" << dist[static_cast<std::size_t>(i)];
        std::cout << "\n";
        assert((dist == std::vector<long long>{0, 2, 7, 4, -2}));
        std::cout << "  -> 到 4 的「最短路」是 0->1->4 = 6+(-4) = 2，负权边被正确利用 ✓\n";
    }

    // ---------- 2. 负环检测 ----------
    std::cout << "\n=== 2. 负环检测（存在负环 -> 最短路无定义）===\n";
    {
        // 1->2->3->1 的总权 = -1 + 3 + (-3) = -1 < 0，是负环
        const int n = 4;
        std::vector<Edge> edges = {{0,1,1},{1,2,-1},{2,3,3},{3,1,-3}};
        std::vector<long long> dist;
        const bool ok = bellmanFord(n, edges, 0, dist);
        std::cout << "  环 1->2->3->1 总权 = -1+3-3 = -1 < 0：\n";
        std::cout << "     Bellman-Ford 返回 " << ok << "（false = 检测到负环）✓\n";
        assert(!ok);

        std::vector<Edge> noNeg = {{0,1,1},{1,2,-1},{2,3,3}};   // 去掉回边 3->1，变 DAG
        assert(bellmanFord(n, noNeg, 0, dist));
        std::cout << "  去掉回边 3->1 后变 DAG：无负环，返回 true ✓\n";
    }

    // ---------- 3. Floyd-Warshall 全源最短路 ----------
    std::cout << "\n=== 3. Floyd-Warshall：一张表的「所有点对最短路」===\n";
    {
        const int n = 4;
        std::vector<Edge> edges = {{0,1,5},{0,3,10},{1,2,3},{2,3,1}};
        auto d = floydWarshall(n, edges);
        std::cout << "       到列:   0    1    2    3\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "    从 " << i << " :  ";
            for (int j = 0; j < n; ++j) {
                const long long x = d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                if (x >= INF) std::cout << " inf  ";
                else          std::cout << " " << x << "   ";
            }
            std::cout << "\n";
        }
        // 0->1->2->3 = 5+3+1 = 9，比 0->3 直达的 10 更短
        assert(d[0][3] == 9);
        std::cout << "  0->3 最短路 = 0->1->2->3 = 5+3+1 = 9 < 直达 10 ✓\n";
    }

    // ---------- 4. ★ 对拍：Floyd 的每一行 == Bellman-Ford ----------
    std::cout << "\n=== 4. ★ 对拍：Floyd 每一行 == 从该点出发的 Bellman-Ford ===\n";
    {
        std::mt19937 rng(20260913);
        int tested = 0;
        for (int trial = 0; trial < 300; ++trial) {
            const int n = 2 + static_cast<int>(rng() % 12);
            std::vector<Edge> edges;
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < n; ++v)
                    if (u != v && rng() % 4 == 0)
                        edges.push_back({u, v, static_cast<int>(rng() % 21) - 6});   // 含负权 -6..14

            // 先确认无负环（用 BF 检测），否则跳过该组
            std::vector<long long> tmp;
            bool anyNeg = false;
            for (int s = 0; s < n && !anyNeg; ++s) if (!bellmanFord(n, edges, s, tmp)) anyNeg = true;
            if (anyNeg) continue;

            auto d = floydWarshall(n, edges);
            for (int s = 0; s < n; ++s) {
                std::vector<long long> bf;
                assert(bellmanFord(n, edges, s, bf));
                for (int t = 0; t < n; ++t) assert(d[static_cast<std::size_t>(s)][static_cast<std::size_t>(t)] == bf[static_cast<std::size_t>(t)]);
            }
            ++tested;
        }
        std::cout << "  " << tested << " 个无负环随机图：Floyd 全对 == Bellman-Ford 逐点 ✓\n";
    }

    // ---------- 5. 传递闭包 ----------
    std::cout << "\n=== 5. 副产品：传递闭包（可达性矩阵）===\n";
    {
        // 0->1, 1->2, 3->2：问 0 能到哪些点
        const int n = 4;
        std::vector<Edge> edges = {{0,1,0},{1,2,0},{3,2,0}};
        auto reach = transitiveClosure(n, edges);
        std::cout << "       可达?    0  1  2  3\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "    从 " << i << " :     ";
            for (int j = 0; j < n; ++j) std::cout << (reach[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] ? "1  " : "0  ");
            std::cout << "\n";
        }
        assert(reach[0][0] && reach[0][1] && reach[0][2] && !reach[0][3]);   // 0 能到 0,1,2
        assert(!reach[3][0] && reach[3][2]);                                  // 3 只能到 2
        std::cout << "  -> 把 Floyd 的 min 换成「或」，就是「谁可达谁」的闭包（数据库 / 编译器用）。\n";
    }

    std::cout << "\n=== 6. 复杂度与选择 ===\n"
                 "  * BFS 无权最短路 O(V+E)；Dijkstra 非负权 O(E log V)；\n"
                 "  * Bellman-Ford 可负权 O(V*E)，还能检测负环（Dijkstra 不行）；\n"
                 "  * Floyd 全源 O(V^3)（实现极简、常数小），稠密图或要「所有点对」时首选；\n"
                 "  * 稀疏图求全源用「从每个点跑一次 Dijkstra」O(V*E log V) 更划算；\n"
                 "  * Bellman-Ford 的 SPFA 优化（队列版）平均更快，但最坏仍是 O(V*E)。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
