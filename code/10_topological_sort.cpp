// 10_topological_sort.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 10 章 图基础与遍历》
// 主题：拓扑排序（Topological Sort）—— 把 DAG 拉成一条「依赖合法的线」
//
//   问题：一堆任务之间有依赖（A 必须先于 B）。如何排一个执行顺序，
//   使得**每条边 u->v 都满足 u 排在 v 前面**？这样的顺序叫「拓扑序」。
//   存在拓扑序  <=>  图是 DAG（有向无环图）。有环则无解（循环依赖）。
//
//   两种等价算法：
//
//   ① Kahn 算法（BFS 版）：
//        先数出每个点的入度；每次取「入度为 0」的点输出，
//        并把它出边指向的点的入度减 1；减到 0 就入队。
//        输出点数 < 总点数  <=>  图里有环。
//
//   ② DFS 版：
//        对每个点 DFS，**在递归返回时（后序）**把点压进结果；
//        最后把结果整体反转，就是拓扑序。
//        原理：一条 u->v 的边，v 一定在 u 之前完成（后序里更早入栈），
//              反转后 u 就在 v 前面。
//
//   拓扑排序是「任务调度 / 编译依赖 / 电子表格重算 / 神经网络数据流」
//   的通用工具。世界模型里，它直接对应「状态转移 DAG 上的因果顺序」。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 10_topological_sort.cpp -o 10_topological_sort && ./10_topological_sort
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 10_topological_sort.cpp -o ts_san && ./ts_san
// ---------------------------------------------------------------------------

#include <algorithm>      // std::reverse
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

struct Edge { int u, v; };

// ---------------------------------------------------------------------------
// Kahn 算法（BFS 版）。返回 true 表示是 DAG 且 order 为拓扑序；
// 返回 false 表示有环（order 只含能排出的部分）。
// ---------------------------------------------------------------------------
static bool topoKahn(int n, const std::vector<Edge>& edges,
                     std::vector<int>& order, std::vector<int>* indegOut = nullptr) {
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));
    std::vector<int> indeg(static_cast<std::size_t>(n), 0);
    for (const Edge& e : edges) { adj[static_cast<std::size_t>(e.u)].push_back(e.v); ++indeg[static_cast<std::size_t>(e.v)]; }

    std::queue<int> q;
    for (int u = 0; u < n; ++u) if (indeg[static_cast<std::size_t>(u)] == 0) q.push(u);

    order.clear();
    while (!q.empty()) {
        const int u = q.front(); q.pop();
        order.push_back(u);
        for (int w : adj[static_cast<std::size_t>(u)])
            if (--indeg[static_cast<std::size_t>(w)] == 0) q.push(w);   // 入度降到 0 才出队
    }
    if (indegOut) *indegOut = indeg;
    return static_cast<int>(order.size()) == n;                        // 排满 n 个 => 无环
}

// ---------------------------------------------------------------------------
// DFS 版拓扑排序：后序 + 整体反转。
// ---------------------------------------------------------------------------
static bool topoDfs(int n, const std::vector<Edge>& edges, std::vector<int>& order) {
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));
    for (const Edge& e : edges) adj[static_cast<std::size_t>(e.u)].push_back(e.v);

    std::vector<int> state(static_cast<std::size_t>(n), 0);           // 0 白 / 1 灰 / 2 黑
    std::vector<int> post;
    bool acyclic = true;
    std::function<void(int)> go = [&](int u) {
        state[static_cast<std::size_t>(u)] = 1;
        for (int w : adj[static_cast<std::size_t>(u)]) {
            if (state[static_cast<std::size_t>(w)] == 1) { acyclic = false; return; }  // 回边
            if (state[static_cast<std::size_t>(w)] == 0) go(w);
        }
        state[static_cast<std::size_t>(u)] = 2;
        post.push_back(u);                                            // 后序：u 在 v 之后
    };
    for (int s = 0; s < n; ++s) if (state[static_cast<std::size_t>(s)] == 0) go(s);

    order.assign(post.rbegin(), post.rend());                         // 反转 => 拓扑序
    return acyclic;
}

// 校验 order 是不是给定 DAG 的合法拓扑序：每个点的「位置」小于它所有后继的位置
static bool isTopoOrder(int n, const std::vector<Edge>& edges, const std::vector<int>& order) {
    if (static_cast<int>(order.size()) != n) return false;
    std::vector<int> pos(static_cast<std::size_t>(n), -1);
    for (int i = 0; i < n; ++i) pos[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] = i;
    for (const Edge& e : edges) if (pos[static_cast<std::size_t>(e.u)] >= pos[static_cast<std::size_t>(e.v)]) return false;
    return true;
}

// ---------------------------------------------------------------------------
// 应用：DAG 上的最长路径（关键路径）—— 用于「项目最早完工时间 / 数据流深度」。
//   在拓扑序上做 DP：dist[v] = max(dist[v], dist[u] + weight(u,v))。
// ---------------------------------------------------------------------------
static int dagLongestPath(int n, const std::vector<Edge>& edges, const std::vector<int>& w,
                          int src, std::vector<int>& dist) {
    std::vector<int> order;
    if (!topoKahn(n, edges, order)) { dist.clear(); return -1; }       // 有环无解
    std::vector<std::vector<std::pair<int,int>>> adj(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < edges.size(); ++i)
        adj[static_cast<std::size_t>(edges[i].u)].push_back({edges[i].v, w[i]});

    const int NEG = -1000000000;
    dist.assign(static_cast<std::size_t>(n), NEG);
    dist[static_cast<std::size_t>(src)] = 0;
    for (int u : order) {                                              // 按拓扑序松弛
        if (dist[static_cast<std::size_t>(u)] == NEG) continue;
        for (auto [v, wt] : adj[static_cast<std::size_t>(u)])
            dist[static_cast<std::size_t>(v)] = std::max(dist[static_cast<std::size_t>(v)],
                                                         dist[static_cast<std::size_t>(u)] + wt);
    }
    int best = NEG;
    for (int d : dist) best = std::max(best, d);
    return best;
}

int main() {
    std::cout << "======== 10 拓扑排序（Topological Sort）========\n\n";

    // ---------- 1. 基本例子 ----------
    std::cout << "=== 1. 课程依赖（经典例子）===\n";
    {
        // 0=高数 1=线代 2=概率 3=数据结构 4=算法 5=机器学习
        // 0->3, 1->3, 2->5, 0->5, 3->4
        const int n = 6;
        std::vector<Edge> edges = {{0,3},{1,3},{2,5},{0,5},{3,4}};
        std::vector<int> kahn, dfs;
        assert(topoKahn(n, edges, kahn));
        assert(topoDfs(n, edges, dfs));
        const char* name[] = {"高数","线代","概率","数据结构","算法","机器学习"};

        std::cout << "  Kahn 拓扑序: ";
        for (std::size_t i = 0; i < kahn.size(); ++i) std::cout << name[kahn[i]] << (i+1<kahn.size()?" -> ":"");
        std::cout << "\n  DFS  拓扑序: ";
        for (std::size_t i = 0; i < dfs.size(); ++i) std::cout << name[dfs[i]] << (i+1<dfs.size()?" -> ":"");
        std::cout << "\n";
        assert(isTopoOrder(n, edges, kahn));
        assert(isTopoOrder(n, edges, dfs));
        std::cout << "  两种算法给出的顺序都满足所有依赖 ✓（拓扑序通常不唯一）\n";
    }

    // ---------- 2. 有环 -> 无解 ----------
    std::cout << "\n=== 2. 有环图：拓扑排序无解 ===\n";
    {
        const int n = 3;
        std::vector<Edge> cyc = {{0,1},{1,2},{2,0}};                    // 循环依赖
        std::vector<int> order;
        const bool ok = topoKahn(n, cyc, order);
        std::cout << "  循环依赖 0->1->2->0：Kahn 返回 " << ok
                  << "（false = 有环），能排出的点 = " << order.size() << " < " << n << " ✓\n";
        assert(!ok && order.size() < static_cast<std::size_t>(n));
        std::vector<int> od;
        assert(!topoDfs(n, cyc, od));
        std::cout << "  DFS 版同样检测到环 ✓\n";
    }

    // ---------- 3. ★ 与暴力对拍（小图全排列）----------
    std::cout << "\n=== 3. ★ 随机小 DAG：拓扑序合法性验证 ===\n";
    {
        std::mt19937 rng(20260913);
        int tested = 0;
        for (int trial = 0; trial < 500; ++trial) {
            const int n = 2 + static_cast<int>(rng() % 7);             // 2..8 个点
            std::vector<Edge> edges;
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < n; ++v)
                    if (u < v && rng() % 3 == 0) edges.push_back({u, v});   // 只连 u<v => 必为 DAG

            std::vector<int> k, d;
            assert(topoKahn(n, edges, k));
            assert(topoDfs(n, edges, d));
            assert(isTopoOrder(n, edges, k));
            assert(isTopoOrder(n, edges, d));
            ++tested;
        }
        std::cout << "  " << tested << " 个随机 DAG：Kahn 与 DFS 的拓扑序都通过合法性校验 ✓\n";
    }

    // ---------- 4. ★ 关键路径（DAG 最长路）----------
    std::cout << "\n=== 4. ★ 应用：项目关键路径（DAG 最长路）===\n";
    {
        // 一个「做菜」流程，边权 = 耗时（分钟）
        //  0 备菜 -> 1 炒菜 -> 3 装盘
        //  0 备菜 -> 2 煮饭 -> 3 装盘
        //  2 煮饭 -> 4 蒸汤 -> 3 装盘
        const int n = 5;
        std::vector<Edge> edges = {{0,1},{0,2},{1,3},{2,3},{2,4},{4,3}};
        std::vector<int> w        = { 10,   5,    8,    3,    20,   2  };
        std::vector<int> dist;
        const int longest = dagLongestPath(n, edges, w, 0, dist);
        std::cout << "  从「备菜」出发，各步最早完工时间：\n";
        const char* task[] = {"备菜","炒菜","煮饭","装盘","蒸汤"};
        for (int i = 0; i < n; ++i) std::cout << "     " << task[i] << " = " << dist[static_cast<std::size_t>(i)] << " 分钟\n";
        std::cout << "  整个流程最短完工时间（关键路径长度）= " << longest << " 分钟\n";
        assert(longest == 27);   // 0->2->4->3 = 5+20+2 = 27，是瓶颈
        std::cout << "  -> 关键路径 = 备菜->煮饭->蒸汤->装盘 = 5+20+2 = 27，决定总工期。\n";
    }

    std::cout << "\n=== 5. 复杂度与要点 ===\n"
                 "  * Kahn：每个点、每条边各处理一次，O(V+E)；顺带还能判环；\n"
                 "  * DFS：O(V+E)，后序反转即拓扑序，同样能判环（回边）；\n"
                 "  * 拓扑序一般不唯一：入度同时为 0 的点谁先谁后都行；\n"
                 "  * 只有 DAG 才有拓扑序；有环 <=> 存在循环依赖 <=> 无解；\n"
                 "  * 应用：任务调度、编译依赖、电子表格重算、DAG 上的 DP（最长/最短路）。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
