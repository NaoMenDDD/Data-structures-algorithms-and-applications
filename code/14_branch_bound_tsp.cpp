// ===========================================================================
// 14_branch_bound_tsp.cpp
// 分支限界（Branch and Bound）求解旅行商问题（TSP）
//   - 暴力：枚举所有 (n-1)! 条哈密顿回路
//   - 分支限界：深度优先 + 「下界剪枝」，一旦部分路段的下界 >= 当前最优就回退
//   - 最近邻启发式：快速得到一个「上界」，帮分支限界尽早剪枝
//
// TSP：从某城市出发，走遍每个城市恰好一次、最后回到起点，总路程最短。
//   它是「NP 难」的经典代表：没有已知多项式算法，但分支限界能在很多实例上
//   把 (n-1)! 的暴搜砍到可接受；它也是「精确解」与「近似解」的分水岭。
//
// 下界（必须 <= 真实最优，否则会误剪）：
//   对「还需要离开」的每个城市 v（当前城市 + 所有未访问城市），
//   它至少要贡献一条出边 >= minOut(v)。故
//     LB = 已走代价 + minOut(当前) + Σ_{v 未访问} minOut(v)
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 14_branch_bound_tsp.cpp -o 14_branch_bound_tsp && ./14_branch_bound_tsp
// ===========================================================================
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace tsp {

// 对称度量：城市随机撒在 [0,1)^2 平面上，距离为欧氏距离（满足三角不等式）
std::vector<std::vector<int>> makeCities(int n, std::mt19937& rng) {
    std::vector<double> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < n; ++i) { x[static_cast<std::size_t>(i)] = u(rng); y[static_cast<std::size_t>(i)] = u(rng); }
    std::vector<std::vector<int>> d(static_cast<std::size_t>(n), std::vector<int>(static_cast<std::size_t>(n), 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = static_cast<int>(
                0.5 + 100000.0 * std::hypot(x[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(j)],
                                            y[static_cast<std::size_t>(i)] - y[static_cast<std::size_t>(j)]));
    return d;
}

std::vector<int> minOutOf(const std::vector<std::vector<int>>& d) {
    const int n = static_cast<int>(d.size());
    std::vector<int> m(static_cast<std::size_t>(n), std::numeric_limits<int>::max());
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) m[static_cast<std::size_t>(i)] = std::min(m[static_cast<std::size_t>(i)], d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
    return m;
}

// 暴力枚举
int bruteForce(const std::vector<std::vector<int>>& d, std::vector<int>* bestPath = nullptr) {
    const int n = static_cast<int>(d.size());
    std::vector<int> p(static_cast<std::size_t>(n - 1));
    std::iota(p.begin(), p.end(), 1);                       // 固定从 0 出发
    int best = std::numeric_limits<int>::max();
    do {
        int c = d[0][static_cast<std::size_t>(p[0])];
        for (int i = 1; i < n - 1; ++i)
            c += d[static_cast<std::size_t>(p[static_cast<std::size_t>(i - 1)])][static_cast<std::size_t>(p[static_cast<std::size_t>(i)])];
        c += d[static_cast<std::size_t>(p[static_cast<std::size_t>(n - 2)])][0];
        if (c < best) { best = c; if (bestPath) { *bestPath = std::vector<int>(static_cast<std::size_t>(n), 0); (*bestPath)[0] = 0; for (int i = 0; i < n - 1; ++i) (*bestPath)[static_cast<std::size_t>(i + 1)] = p[static_cast<std::size_t>(i)]; } }
    } while (std::next_permutation(p.begin(), p.end()));
    return best;
}

// 最近邻启发式：每步走最近的未访问城市（给上界）
int nearestNeighbor(const std::vector<std::vector<int>>& d) {
    const int n = static_cast<int>(d.size());
    std::vector<bool> vis(static_cast<std::size_t>(n), false);
    int cur = 0, cost = 0;
    vis[0] = true;
    for (int step = 1; step < n; ++step) {
        int nxt = -1, bd = std::numeric_limits<int>::max();
        for (int v = 0; v < n; ++v)
            if (!vis[static_cast<std::size_t>(v)] && d[static_cast<std::size_t>(cur)][static_cast<std::size_t>(v)] < bd) { bd = d[static_cast<std::size_t>(cur)][static_cast<std::size_t>(v)]; nxt = v; }
        cost += bd;
        vis[static_cast<std::size_t>(nxt)] = true;
        cur = nxt;
    }
    return cost + d[static_cast<std::size_t>(cur)][0];
}

// 分支限界
struct BB {
    const std::vector<std::vector<int>>* d;
    int n;
    std::vector<int> minOut;
    int best = std::numeric_limits<int>::max();
    std::vector<int> bestPath;
    long long nodes = 0;                                    // 展开的结点数

    void dfs(int cur, int visited, int cost, int mask, std::vector<int>& path) {
        ++nodes;
        if (visited == n) {                                 // 回到起点，结算
            const int total = cost + (*d)[static_cast<std::size_t>(cur)][0];
            if (total < best) { best = total; bestPath = path; bestPath.push_back(0); }
            return;
        }
        int lb = cost + minOut[static_cast<std::size_t>(cur)];          // 当前城市还要离开
        for (int v = 0; v < n; ++v)
            if (!(mask & (1 << v))) lb += minOut[static_cast<std::size_t>(v)];   // 未访问城市各自还要离开
        if (lb >= best) return;                             // 下界已不优 -> 剪枝

        // 邻居按距离升序扩展，更早发现好解 -> 更早剪枝
        std::vector<int> order;
        for (int v = 0; v < n; ++v) if (!(mask & (1 << v)) && v != cur) order.push_back(v);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return (*d)[static_cast<std::size_t>(cur)][static_cast<std::size_t>(a)] <
                   (*d)[static_cast<std::size_t>(cur)][static_cast<std::size_t>(b)];
        });
        for (int v : order) {
            path.push_back(v);
            dfs(v, visited + 1, cost + (*d)[static_cast<std::size_t>(cur)][static_cast<std::size_t>(v)],
                mask | (1 << v), path);
            path.pop_back();
        }
    }
};

int branchAndBound(const std::vector<std::vector<int>>& d, long long& nodes, int ub) {
    BB b;
    b.d = &d;
    b.n = static_cast<int>(d.size());
    b.minOut = minOutOf(d);
    b.best = ub;                                            // 用启发式上界初始化，剪枝更狠
    std::vector<int> path;
    path.push_back(0);
    b.dfs(0, 1, 0, 1, path);
    nodes = b.nodes;
    return b.best;
}

} // namespace tsp

static void testCorrectness() {
    std::cout << "===== 1) 对拍：分支限界 == 暴力枚举 =====\n";
    std::mt19937 rng(20260914);
    for (int t = 0; t < 300; ++t) {
        const int n = 3 + static_cast<int>(rng() % 7);       // n = 3..9
        const auto d = tsp::makeCities(n, rng);
        const int bf = tsp::bruteForce(d);
        long long nodes = 0;
        const int bb = tsp::branchAndBound(d, nodes, tsp::nearestNeighbor(d));
        assert(bb == bf);
    }
    std::cout << "  [断言通过] 300 组随机（n=3..9）：分支限界的最优值 == 暴力枚举\n\n";
}

static void testScaling() {
    std::cout << "===== 2) 规模对比：暴力 (n-1)! vs 分支限界展开结点数 =====\n";
    std::mt19937 rng(20260913);
    std::cout << "  n\t(n-1)!\t\t最近邻\t\t分支限界最优\t展开结点数\n";
    for (int n : {8, 9, 10, 11, 12, 13, 14}) {
        const auto d = tsp::makeCities(n, rng);
        const int nn = tsp::nearestNeighbor(d);
        long long nodes = 0;
        const int opt = tsp::branchAndBound(d, nodes, nn);
        long long fact = 1;
        for (int i = 2; i <= n - 1; ++i) fact *= i;          // (n-1)!
        std::cout << "  " << n << "\t" << fact << "\t\t" << nn << "\t\t" << opt << "\t\t" << nodes << "\n";
    }
    std::cout << "  -> 分支限界用「下界 + 当前最优」把 (n-1)! 的暴搜砍掉几个数量级；\n";
    std::cout << "     最近邻是「近似解」，通常比最优差几个百分点，却快得多。\n\n";
}

static void testGap() {
    std::cout << "===== 3) 近似（最近邻）vs 精确（分支限界）的差距 =====\n";
    std::mt19937 rng(555);
    const int n = 11;
    const auto d = tsp::makeCities(n, rng);
    const int nn = tsp::nearestNeighbor(d);
    long long nodes = 0;
    const int opt = tsp::branchAndBound(d, nodes, nn);
    std::cout << "  n=" << n << "：最近邻 = " << nn << "，最优 = " << opt
              << "，多走 " << (100.0 * (nn - opt) / opt) << "%\n";
    std::cout << "  -> 启发式给「上界」，下界给「保证」；分支限界在两者之间夹出最优解。\n\n";
    assert(nn >= opt);                                        // 启发式不可能优于最优
}

int main() {
    std::cout << "########## 分支限界：旅行商问题 ##########\n\n";
    testCorrectness();
    testScaling();
    testGap();
    std::cout << "全部测试通过。\n";
    return 0;
}
