// ===========================================================================
// 15_vertex_cover_approx.cpp
// 近似算法（Approximation Algorithm）：以「顶点覆盖」为例
//   - 顶点覆盖：选最少的顶点，使每条边至少有一个端点被选中（NP 难）
//   - ★ 2-近似：基于「极大匹配」——匹配里每条边选两个端点，
//     结果 <= 2 * 最优（匹配边两两不相邻 -> 最优至少得覆盖其中每条边的一个端点）
//   - 贪心（按度）：实践常更小，但无保证
//   - 暴力枚举：小规模求精确最优，用于验证「近似比 <= 2」与统计实际比值
//
// 近似算法的语言：对一个最小化问题，若算法输出 A 满足
//     A <= rho * OPT
// 就说它是 rho-近似（rho >= 1）。rho 越小越好；顶点覆盖的匹配算法 rho = 2。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 15_vertex_cover_approx.cpp -o 15_vertex_cover_approx && ./15_vertex_cover_approx
// ===========================================================================
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace vc {

struct Graph {
    int n = 0;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::uint32_t> adj;                          // adj[v] 的位 i 表示 v-i 有边（n <= 32）
};

Graph randomGraph(int n, double p, std::mt19937& rng) {
    Graph g;
    g.n = n;
    g.adj.assign(static_cast<std::size_t>(n), 0u);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (u(rng) < p) {
                g.edges.push_back({i, j});
                g.adj[static_cast<std::size_t>(i)] |= (1u << j);
                g.adj[static_cast<std::size_t>(j)] |= (1u << i);
            }
    return g;
}

// 检验一个顶点集合是否是合法覆盖
bool isCover(const Graph& g, const std::vector<char>& in) {
    for (const auto& e : g.edges)
        if (!in[static_cast<std::size_t>(e.first)] && !in[static_cast<std::size_t>(e.second)]) return false;
    return true;
}

// ★ 2-近似：极大匹配（每条匹配边选两个端点）
std::vector<char> approxMatching(const Graph& g) {
    std::vector<char> in(static_cast<std::size_t>(g.n), 0);
    for (const auto& e : g.edges)                                // 任取一条两端都未覆盖的边
        if (!in[static_cast<std::size_t>(e.first)] && !in[static_cast<std::size_t>(e.second)]) {
            in[static_cast<std::size_t>(e.first)] = 1;
            in[static_cast<std::size_t>(e.second)] = 1;
        }
    return in;
}

// 贪心：每次选「覆盖最多未覆盖边」的顶点
std::vector<char> greedyDegree(const Graph& g) {
    const std::size_t m = g.edges.size();
    std::vector<char> edgeCov(m, 0), in(static_cast<std::size_t>(g.n), 0);
    std::size_t remaining = m;
    while (remaining > 0) {
        std::vector<int> deg(static_cast<std::size_t>(g.n), 0);
        for (std::size_t i = 0; i < m; ++i)
            if (!edgeCov[i]) {
                ++deg[static_cast<std::size_t>(g.edges[i].first)];
                ++deg[static_cast<std::size_t>(g.edges[i].second)];
            }
        int best = -1;
        for (int v = 0; v < g.n; ++v)
            if (!in[static_cast<std::size_t>(v)] && (best < 0 || deg[static_cast<std::size_t>(v)] > deg[static_cast<std::size_t>(best)]))
                best = v;
        in[static_cast<std::size_t>(best)] = 1;
        for (std::size_t i = 0; i < m; ++i)
            if (!edgeCov[i] && (g.edges[i].first == best || g.edges[i].second == best)) { edgeCov[i] = 1; --remaining; }
    }
    return in;
}

// 穷举所有子集求最优（n <= 20 用；这里 n <= 18）
int bruteOptimal(const Graph& g) {
    const int n = g.n;
    const std::uint32_t full = (n == 32) ? ~0u : ((1u << n) - 1);
    int best = n;
    for (std::uint32_t S = 0; S <= full; ++S) {
        const int bits = __builtin_popcount(S);
        if (bits >= best) continue;
        bool ok = true;
        for (int v = 0; v < n && ok; ++v)
            if (!((S >> v) & 1u) && (g.adj[static_cast<std::size_t>(v)] & ~S)) ok = false;   // v 未选，却有邻居未选 -> 有边没盖住
        if (ok) best = bits;
    }
    return best;
}

int count(const std::vector<char>& in) {
    int c = 0;
    for (char x : in) c += x ? 1 : 0;
    return c;
}

} // namespace vc

int main() {
    std::cout << "########## 近似算法：顶点覆盖 ##########\n\n";

    std::cout << "===== 1) 单例演示：极大匹配 2-近似 =====\n";
    {
        std::mt19937 rng(20260913);
        const vc::Graph g = vc::randomGraph(12, 0.35, rng);
        const auto a = vc::approxMatching(g);
        const auto gg = vc::greedyDegree(g);
        const int opt = vc::bruteOptimal(g);
        std::cout << "  图：n=12 顶点, " << g.edges.size() << " 条边\n";
        std::cout << "  最优顶点覆盖 |OPT| = " << opt << "\n";
        std::cout << "  极大匹配近似   |A| = " << vc::count(a) << "  （若有匹配需 2 倍上界）\n";
        std::cout << "  贪心(按度)     |G| = " << vc::count(gg) << "\n";
        assert(vc::isCover(g, a) && vc::isCover(g, gg));
        assert(vc::count(a) <= 2 * opt);                          // 2-近似的保证
        assert(vc::count(gg) >= opt);
        std::cout << "  -> 两个结果都是合法覆盖，且匹配近似 <= 2*OPT。\n\n";
    }

    std::cout << "===== 2) 统计：近似比随图变稠密的变化 =====\n";
    {
        std::cout << "  每顶点平均度\t样本\t匹配近似/OPT 均值\t最大\t贪心/OPT 均值\t最大\n";
        std::mt19937 rng(20260915);
        for (double p : {0.1, 0.2, 0.35, 0.5, 0.7}) {
            double sumA = 0, maxA = 0, sumG = 0, maxG = 0;
            const int samples = 120;
            for (int t = 0; t < samples; ++t) {
                const vc::Graph g = vc::randomGraph(16, p, rng);
                const int opt = vc::bruteOptimal(g);
                const double ra = static_cast<double>(vc::count(vc::approxMatching(g))) / opt;
                const double rg = static_cast<double>(vc::count(vc::greedyDegree(g))) / opt;
                sumA += ra; maxA = std::max(maxA, ra);
                sumG += rg; maxG = std::max(maxG, rg);
                assert(ra <= 2.0 + 1e-9);                          // 匹配近似的硬保证
            }
            std::cout << "  " << (p * 15) << "\t\t" << samples << "\t"
                      << (sumA / samples) << "\t\t" << maxA << "\t"
                      << (sumG / samples) << "\t\t" << maxG << "\n";
        }
        std::cout << "  -> 匹配近似稳定地 <= 2*OPT（最坏比就在 2 附近出现）；\n";
        std::cout << "     贪心(按度)平均更小，但**没有理论保证**，个别实例可能更差。\n\n";
    }

    std::cout << "===== 3) 2 倍界是「紧的」：构造最坏情形 =====\n";
    {
        // 完全二部图 K_{m,m} + 每边… 更简单：一堆「互不相邻」的边（匹配）就是最坏比 2
        // 用一个「星形森林」：m 条互不相邻的边 = 完美匹配
        vc::Graph g;
        g.n = 12;
        g.adj.assign(12, 0u);
        for (int i = 0; i < 6; ++i) {                             // 6 条互不相邻的边
            const int u = 2 * i, v = 2 * i + 1;
            g.edges.push_back({u, v});
            g.adj[static_cast<std::size_t>(u)] |= (1u << v);
            g.adj[static_cast<std::size_t>(v)] |= (1u << u);
        }
        const int opt = vc::bruteOptimal(g);
        const int a = vc::count(vc::approxMatching(g));
        std::cout << "  6 条互不相邻的边：OPT = " << opt << "，匹配近似 = " << a
                  << "，比值 = " << (static_cast<double>(a) / opt) << "\n";
        std::cout << "  -> 每条匹配边都要选 2 个端点，而最优每条边选 1 个就够 -> 恰好 2 倍。\n";
        std::cout << "     说明 2-近似的常数 2 无法再降（对一般图而言）。\n\n";
    }

    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
