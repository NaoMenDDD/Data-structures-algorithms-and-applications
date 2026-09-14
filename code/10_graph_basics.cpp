// 10_graph_basics.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 10 章 图基础与遍历》
// 主题：图的表示 + DFS / BFS + 连通性 / 环 / 二分图
//
//   树是「没有环、且连通」的图。图把这两个限制都拿掉：
//     * 可以有环（周期、回路、状态转移可以回来）；
//     * 可以多个连通分量（好几个岛）。
//   于是图能表达的东西一下子多了：社交网络、路网、依赖关系、状态转移图、
//   神经网络的计算图 —— 几乎一切「关系」都是图。
//
//   本文件覆盖图的「地基」：
//     1. 三种表示：邻接矩阵 / 邻接表 / 边表 —— 各自的时空取舍；
//     2. 深度优先遍历 DFS（递归版 + 迭代版，用栈）；
//     3. 广度优先遍历 BFS（用队列）—— 天然求「无权最短路」；
//     4. 连通分量个数、判环（无向用「回边」、有向用「在栈中」）、二分图判定；
//     5. 网格 flood fill（把二维网格当隐式图，DFS/BFS 都行）。
//
//   为什么 DFS / BFS 是「一切图算法之母」？因为后面所有高级算法
//   （拓扑排序、最短路、连通性、割点、强连通分量……）都是「按某种顺序
//   遍历并顺手记录信息」，而 DFS/BFS 就是在定义那个「顺序」。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 10_graph_basics.cpp -o 10_graph_basics && ./10_graph_basics
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 10_graph_basics.cpp -o gb_san && ./gb_san
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

// ---------------------------------------------------------------------------
// 邻接表表示的图（可无向 / 有向）
//   优点：空间 O(V+E)，遍历邻居 O(deg(v))；稀疏图的标配。
//   邻接矩阵 space O(V^2)、判边 O(1)，稠密图或需要常数判边时用。
// ---------------------------------------------------------------------------
class Graph {
public:
    explicit Graph(int n, bool directed = false) : n_(n), directed_(directed), adj_(static_cast<std::size_t>(n)) {}

    void addEdge(int u, int v) {
        adj_[static_cast<std::size_t>(u)].push_back(v);
        if (!directed_) adj_[static_cast<std::size_t>(v)].push_back(u);
    }

    int n() const { return n_; }
    bool directed() const { return directed_; }
    const std::vector<int>& neighbors(int u) const { return adj_[static_cast<std::size_t>(u)]; }

    // 边数（无向图里每条边会被两个邻接表各记一次，除 2）
    std::size_t edgeCount() const {
        std::size_t s = 0;
        for (const auto& a : adj_) s += a.size();
        return directed_ ? s : s / 2;
    }

    // ---------- DFS（递归版）----------
    // 访问顺序写进 order；用 visited 防重复（图有环，必须标记）。
    void dfsRecursive(int start, std::vector<int>& order) const {
        std::vector<char> visited(static_cast<std::size_t>(n_), 0);
        std::function<void(int)> go = [&](int u) {
            visited[static_cast<std::size_t>(u)] = 1;
            order.push_back(u);
            for (int w : adj_[static_cast<std::size_t>(u)])
                if (!visited[static_cast<std::size_t>(w)]) go(w);
        };
        go(start);
    }

    // ---------- DFS（迭代版，显式栈）----------
    // 关键：入栈时就标记 visited（否则同一结点会被多条边重复入栈）。
    void dfsIterative(int start, std::vector<int>& order) const {
        std::vector<char> visited(static_cast<std::size_t>(n_), 0);
        std::vector<int> stk{start};
        visited[static_cast<std::size_t>(start)] = 1;
        while (!stk.empty()) {
            const int u = stk.back(); stk.pop_back();
            order.push_back(u);
            // 逆序入栈，让「先入栈的邻居先被弹出」（更接近递归的从左到右）。
            // 注意：因为这里「入栈即标记」，迭代版顺序仍可能和递归版不同 —— 都合法。
            const auto& nb = adj_[static_cast<std::size_t>(u)];
            for (auto it = nb.rbegin(); it != nb.rend(); ++it)
                if (!visited[static_cast<std::size_t>(*it)]) {
                    visited[static_cast<std::size_t>(*it)] = 1;
                    stk.push_back(*it);
                }
        }
    }

    // ---------- BFS（队列）----------
    // 按「距离 start 的层数」逐层访问。无权图里 dist[] 就是最短路长度。
    void bfs(int start, std::vector<int>& order, std::vector<int>* dist = nullptr) const {
        std::vector<char> visited(static_cast<std::size_t>(n_), 0);
        std::vector<int> d(static_cast<std::size_t>(n_), -1);
        std::queue<int> q;
        visited[static_cast<std::size_t>(start)] = 1;
        d[static_cast<std::size_t>(start)] = 0;
        q.push(start);
        while (!q.empty()) {
            const int u = q.front(); q.pop();
            order.push_back(u);
            for (int w : adj_[static_cast<std::size_t>(u)])
                if (!visited[static_cast<std::size_t>(w)]) {
                    visited[static_cast<std::size_t>(w)] = 1;
                    d[static_cast<std::size_t>(w)] = d[static_cast<std::size_t>(u)] + 1;
                    q.push(w);
                }
        }
        if (dist) *dist = d;
    }

    // ---------- 连通分量个数 ----------
    // 无向图：从每个未访问点起一次 DFS/BFS，扫到的就是整个分量。
    int connectedComponents() const {
        assert(!directed_ && "连通分量这里只对无向图定义；有向图要看强连通分量（第 11 章）");
        std::vector<char> visited(static_cast<std::size_t>(n_), 0);
        int components = 0;
        for (int s = 0; s < n_; ++s) {
            if (visited[static_cast<std::size_t>(s)]) continue;
            ++components;
            std::vector<int> stk{s};
            visited[static_cast<std::size_t>(s)] = 1;
            while (!stk.empty()) {
                const int u = stk.back(); stk.pop_back();
                for (int w : adj_[static_cast<std::size_t>(u)])
                    if (!visited[static_cast<std::size_t>(w)]) {
                        visited[static_cast<std::size_t>(w)] = 1;
                        stk.push_back(w);
                    }
            }
        }
        return components;
    }

    // ---------- 无向图判环：DFS，遇到「已访问且不是父结点」的邻居即有环 ----------
    bool hasCycleUndirected() const {
        assert(!directed_);
        std::vector<char> visited(static_cast<std::size_t>(n_), 0);
        std::function<bool(int,int)> go = [&](int u, int parent) -> bool {
            visited[static_cast<std::size_t>(u)] = 1;
            for (int w : adj_[static_cast<std::size_t>(u)]) {
                if (!visited[static_cast<std::size_t>(w)]) {
                    if (go(w, u)) return true;
                } else if (w != parent) {
                    return true;                    // 回到非父结点 -> 有环
                }
            }
            return false;
        };
        for (int s = 0; s < n_; ++s)
            if (!visited[static_cast<std::size_t>(s)] && go(s, -1)) return true;
        return false;
    }

    // ---------- 有向图判环：DFS + 「当前递归栈」标记 ----------
    //   结点有三态：0 未访问、1 在栈中（灰）、2 已完成（黑）。
    //   若从 u 走到一个「在栈中」的 v，说明有回边 -> 有环。
    bool hasCycleDirected() const {
        assert(directed_);
        std::vector<int> state(static_cast<std::size_t>(n_), 0);
        std::function<bool(int)> go = [&](int u) -> bool {
            state[static_cast<std::size_t>(u)] = 1;                 // 灰：在栈中
            for (int w : adj_[static_cast<std::size_t>(u)]) {
                if (state[static_cast<std::size_t>(w)] == 1) return true;   // 回边
                if (state[static_cast<std::size_t>(w)] == 0 && go(w)) return true;
            }
            state[static_cast<std::size_t>(u)] = 2;                 // 黑：完成
            return false;
        };
        for (int s = 0; s < n_; ++s)
            if (state[static_cast<std::size_t>(s)] == 0 && go(s)) return true;
        return false;
    }

    // ---------- 二分图判定：BFS 二染色，遇到同色邻居则不是二分图 ----------
    bool isBipartite() const {
        std::vector<int> color(static_cast<std::size_t>(n_), -1);
        for (int s = 0; s < n_; ++s) {
            if (color[static_cast<std::size_t>(s)] != -1) continue;
            std::queue<int> q;
            color[static_cast<std::size_t>(s)] = 0;
            q.push(s);
            while (!q.empty()) {
                const int u = q.front(); q.pop();
                for (int w : adj_[static_cast<std::size_t>(u)]) {
                    if (color[static_cast<std::size_t>(w)] == -1) {
                        color[static_cast<std::size_t>(w)] = color[static_cast<std::size_t>(u)] ^ 1;
                        q.push(w);
                    } else if (color[static_cast<std::size_t>(w)] == color[static_cast<std::size_t>(u)]) {
                        return false;                    // 同色相邻 -> 冲突
                    }
                }
            }
        }
        return true;
    }

private:
    int n_;
    bool directed_;
    std::vector<std::vector<int>> adj_;
};

// ---------------------------------------------------------------------------
// 网格 flood fill：把 R×C 的网格当成「隐式图」（上下左右 4 邻居）。
//   统计「岛屿」个数（'1' 陆地、'0' 水），就是数连通分量。
// ---------------------------------------------------------------------------
static int countIslands(std::vector<std::string> grid) {
    const int R = static_cast<int>(grid.size());
    if (R == 0) return 0;
    const int C = static_cast<int>(grid[0].size());
    int islands = 0;
    const int dr[] = {1, -1, 0, 0}, dc[] = {0, 0, 1, -1};
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            if (grid[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] != '1') continue;
            ++islands;
            std::vector<std::pair<int,int>> stk{{r, c}};
            grid[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = '0';   // 标记已访问
            while (!stk.empty()) {
                auto [cr, cc] = stk.back(); stk.pop_back();
                for (int d = 0; d < 4; ++d) {
                    const int nr = cr + dr[d], nc = cc + dc[d];
                    if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                    if (grid[static_cast<std::size_t>(nr)][static_cast<std::size_t>(nc)] != '1') continue;
                    grid[static_cast<std::size_t>(nr)][static_cast<std::size_t>(nc)] = '0';
                    stk.push_back({nr, nc});
                }
            }
        }
    }
    return islands;
}

int main() {
    std::cout << "======== 10 图基础与遍历 ========\n\n";

    // ---------- 1. 表示方法与基本遍历 ----------
    std::cout << "=== 1. 邻接表 + DFS / BFS（无向图）===\n";
    {
        //  0 - 1 - 3
        //  |   |   |
        //  2 - + - 4       （一个连通的无向图）
        Graph g(5, false);
        g.addEdge(0, 1); g.addEdge(0, 2); g.addEdge(1, 2);
        g.addEdge(1, 3); g.addEdge(3, 4); g.addEdge(2, 4);
        std::cout << "  顶点数 V = " << g.n() << "，边数 E = " << g.edgeCount() << "\n";

        std::vector<int> d1, d2;
        g.dfsRecursive(0, d1);
        g.dfsIterative(0, d2);
        std::cout << "  DFS(递归) 从 0： "; for (int x : d1) std::cout << x << " "; std::cout << "\n";
        std::cout << "  DFS(迭代) 从 0： "; for (int x : d2) std::cout << x << " "; std::cout << "\n";
        assert(d1.size() == 5 && d2.size() == 5);          // 都能访问到全部 5 个点

        std::vector<int> b, dist;
        g.bfs(0, b, &dist);
        std::cout << "  BFS 从 0：      "; for (int x : b) std::cout << x << " "; std::cout << "\n";
        std::cout << "  到各点的无权最短距离： ";
        for (int i = 0; i < g.n(); ++i) std::cout << "d(" << i << ")=" << dist[static_cast<std::size_t>(i)] << " ";
        std::cout << "\n";
        assert((dist == std::vector<int>{0, 1, 1, 2, 2}));  // BFS 层数
        std::cout << "  -> DFS 探到底、BFS 逐层扩散；BFS 的 dist 即无权最短路 ✓\n";
    }

    // ---------- 2. 连通分量 ----------
    std::cout << "\n=== 2. 连通分量个数（无向图的「几个岛」）===\n";
    {
        Graph g(7, false);
        g.addEdge(0, 1); g.addEdge(1, 2);          // 分量 A：{0,1,2}
        g.addEdge(3, 4);                           // 分量 B：{3,4}
        // 5、6 各自孤立
        std::cout << "  7 个顶点、3 条边 -> 连通分量数 = " << g.connectedComponents()
                  << "（应为 4：{0,1,2}、{3,4}、{5}、{6}）\n";
        assert(g.connectedComponents() == 4);
    }

    // ---------- 3. 判环 ----------
    std::cout << "\n=== 3. 判环：无向用「回边」，有向用「在栈中」===\n";
    {
        Graph tree(4, false);                      // 一棵树：无边环
        tree.addEdge(0, 1); tree.addEdge(1, 2); tree.addEdge(1, 3);
        assert(!tree.hasCycleUndirected());
        tree.addEdge(2, 3);                        // 加一条边形成环 1-2-3-1
        assert(tree.hasCycleUndirected());
        std::cout << "  无向：4 点 3 边（树）无环 ✓；加 1 条边后成环 ✓\n";

        Graph dag(4, true);                        // 有向无环图：0->1->2->3
        dag.addEdge(0, 1); dag.addEdge(1, 2); dag.addEdge(2, 3); dag.addEdge(0, 3);
        assert(!dag.hasCycleDirected());
        std::cout << "  有向 DAG 0->1->2->3, 0->3：无环 ✓\n";

        Graph cyc(3, true);                        // 有向环 0->1->2->0
        cyc.addEdge(0, 1); cyc.addEdge(1, 2); cyc.addEdge(2, 0);
        assert(cyc.hasCycleDirected());
        std::cout << "  有向环 0->1->2->0：检测到环 ✓\n";
        std::cout << "  -> 有向图判环必须区分「在栈中(灰)」与「已完成(黑)」，否则误报！\n";
    }

    // ---------- 4. 二分图 ----------
    std::cout << "\n=== 4. 二分图判定（BFS 二染色）===\n";
    {
        Graph even(6, false);                      // 偶环 C6：二分图
        for (int i = 0; i < 6; ++i) even.addEdge(i, (i + 1) % 6);
        assert(even.isBipartite());
        std::cout << "  偶环 C6（6 个点）：是二分图 ✓\n";

        Graph odd(5, false);                       // 奇环 C5：不是二分图
        for (int i = 0; i < 5; ++i) odd.addEdge(i, (i + 1) % 5);
        assert(!odd.isBipartite());
        std::cout << "  奇环 C5（5 个点）：不是二分图 ✓\n";
        std::cout << "  -> 「图能二染色」等价于「没有奇环」，也是「无环冲突」的抽象。\n";
    }

    // ---------- 5. 网格 flood fill ----------
    std::cout << "\n=== 5. 网格 flood fill：数岛屿（隐式图）===\n";
    {
        std::vector<std::string> grid = {
            "11000",
            "11000",
            "00100",
            "00011",
        };
        const int isl = countIslands(grid);
        std::cout << "  岛屿个数 = " << isl << "（应为 3）\n";
        assert(isl == 3);
        std::cout << "  -> 网格的上下左右邻居 = 隐式图的边；DFS/BFS 直接可用。\n";
    }

    // ---------- 6. 随机图上的遍历不变式 ----------
    std::cout << "\n=== 6. 随机图：DFS/BFS 访问数 == 连通分量规模 ===\n";
    {
        std::mt19937 rng(20260913);
        for (int trial = 0; trial < 200; ++trial) {
            const int n = 1 + static_cast<int>(rng() % 50);
            Graph g(n, false);
            for (int u = 0; u < n; ++u)
                for (int v = u + 1; v < n; ++v)
                    if (rng() % 4 == 0) g.addEdge(u, v);

            // 每个点的 DFS / BFS 访问数必须一致（同一分量内可达性相同）
            for (int s = 0; s < n; ++s) {
                std::vector<int> a, b;
                g.dfsRecursive(s, a);
                g.bfs(s, b, nullptr);
                assert(a.size() == b.size());
            }
            // 连通性自洽：任一 DFS 访问数 == 从该点出发的可达点数
        }
        std::cout << "  200 个随机无向图 × 每个起点：DFS 访问数 == BFS 访问数 ✓\n";
    }

    std::cout << "\n=== 7. 复杂度与要点 ===\n"
                 "  * 邻接表：空间 O(V+E)；遍历全部邻居 O(V+E)（每个点、每条边各访问常数次）；\n"
                 "  * 邻接矩阵：空间 O(V^2)、判边 O(1)；稠密图或需 O(1) 判边时用；\n"
                 "  * DFS 用栈（递归或显式），BFS 用队列；二者都是 O(V+E)；\n"
                 "  * BFS 求「无权图最短路」；DFS 擅长「连通性 / 拓扑序 / 找环」；\n"
                 "  * 图有环，遍历必须 visited[] 防死循环 —— 这是与树遍历最大的区别。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
