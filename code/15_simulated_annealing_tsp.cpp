// ===========================================================================
// 15_simulated_annealing_tsp.cpp
// 启发式（Heuristic）：模拟退火（Simulated Annealing, SA）解 TSP
//   - 用「最近邻」得到一个初始解（贪心上界）
//   - 用「2-opt」作为邻域动作：反转一段路径
//   - ★ 模拟退火：以概率 exp(-Δ/T) 接受「变差」的移动，从而跳出局部最优；
//     温度 T 按几何降温 -> 前期乱走探索、后期几乎只接受改进（收敛）
//
// 为什么需要启发式：TSP 是 NP 难，精确解（分支限界）在城市多时不可行；
//   模拟退火 / 遗传算法 / 蚁群等「元启发式」不保证最优，但能在合理时间里
//   给出接近最优的解 —— 这是组合优化在工程里的主流打法。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 15_simulated_annealing_tsp.cpp -o 15_simulated_annealing_tsp && ./15_simulated_annealing_tsp
// ===========================================================================
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace sa {

using Matrix = std::vector<std::vector<int>>;

Matrix makeCities(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) { x[static_cast<std::size_t>(i)] = u(rng); y[static_cast<std::size_t>(i)] = u(rng); }
    Matrix d(static_cast<std::size_t>(n), std::vector<int>(static_cast<std::size_t>(n), 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = static_cast<int>(
                0.5 + 100.0 * std::hypot(x[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(j)],
                                         y[static_cast<std::size_t>(i)] - y[static_cast<std::size_t>(j)]));
    return d;
}

int tourLen(const Matrix& d, const std::vector<int>& t) {
    const int n = static_cast<int>(t.size());
    int s = 0;
    for (int i = 0; i < n; ++i) s += d[static_cast<std::size_t>(t[static_cast<std::size_t>(i)])][static_cast<std::size_t>(t[static_cast<std::size_t>((i + 1) % n)])];
    return s;
}

std::vector<int> nearestNeighbor(const Matrix& d) {
    const int n = static_cast<int>(d.size());
    std::vector<char> vis(static_cast<std::size_t>(n), 0);
    std::vector<int> t;
    t.reserve(static_cast<std::size_t>(n));
    int cur = 0;
    vis[0] = 1;
    t.push_back(0);
    for (int s = 1; s < n; ++s) {
        int nxt = -1, bd = std::numeric_limits<int>::max();
        for (int v = 0; v < n; ++v)
            if (!vis[static_cast<std::size_t>(v)] && d[static_cast<std::size_t>(cur)][static_cast<std::size_t>(v)] < bd) {
                bd = d[static_cast<std::size_t>(cur)][static_cast<std::size_t>(v)];
                nxt = v;
            }
        t.push_back(nxt);
        vis[static_cast<std::size_t>(nxt)] = 1;
        cur = nxt;
    }
    return t;
}

int bruteOptimal(const Matrix& d) {
    const int n = static_cast<int>(d.size());
    std::vector<int> p(static_cast<std::size_t>(n - 1));
    std::iota(p.begin(), p.end(), 1);
    int best = std::numeric_limits<int>::max();
    do {
        int c = d[0][static_cast<std::size_t>(p[0])];
        for (int i = 1; i < n - 1; ++i) c += d[static_cast<std::size_t>(p[static_cast<std::size_t>(i - 1)])][static_cast<std::size_t>(p[static_cast<std::size_t>(i)])];
        c += d[static_cast<std::size_t>(p[static_cast<std::size_t>(n - 2)])][0];
        best = std::min(best, c);
    } while (std::next_permutation(p.begin(), p.end()));
    return best;
}

// 自适应初始温度：采样随机 2-opt 的「变差」幅度，取中位数 D，
// 令 T0 = D/ln2（此时「接受一个典型坏解」的概率约 0.5）
double initialTemperature(const Matrix& d, const std::vector<int>& t, std::mt19937& rng) {
    const int n = static_cast<int>(t.size());
    std::vector<int> deltas;
    for (int s = 0; s < 4000; ++s) {
        int i = static_cast<int>(rng() % static_cast<unsigned>(n - 1));
        int j = static_cast<int>(rng() % static_cast<unsigned>(n - 1));
        if (i > j) std::swap(i, j);
        if (i == 0 && j == n - 1) continue;
        const int prev = t[static_cast<std::size_t>((i - 1 + n) % n)];
        const int a = t[static_cast<std::size_t>(i)];
        const int b = t[static_cast<std::size_t>(j)];
        const int nxt = t[static_cast<std::size_t>((j + 1) % n)];
        const int delta = d[static_cast<std::size_t>(prev)][static_cast<std::size_t>(b)] +
                          d[static_cast<std::size_t>(a)][static_cast<std::size_t>(nxt)] -
                          d[static_cast<std::size_t>(prev)][static_cast<std::size_t>(a)] -
                          d[static_cast<std::size_t>(b)][static_cast<std::size_t>(nxt)];
        if (delta > 0) deltas.push_back(delta);
    }
    if (deltas.empty()) return 1.0;
    std::sort(deltas.begin(), deltas.end());
    const double median = static_cast<double>(deltas[deltas.size() / 2]);
    return median / std::log(2.0);
}

// 模拟退火：2-opt 邻域，几何降温
struct Result { std::vector<int> tour; int length; long long acceptedWorse; long long steps; };

Result anneal(const Matrix& d, std::mt19937& rng, double T0, double alpha, double Tmin,
              int itersPerT, std::vector<int> tour) {
    const int n = static_cast<int>(tour.size());
    int cur = tourLen(d, tour);
    std::vector<int> best = tour;
    int bestLen = cur;
    long long accWorse = 0, steps = 0;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (double T = T0; T > Tmin; T *= alpha) {
        for (int it = 0; it < itersPerT; ++it) {
            ++steps;
            int i = static_cast<int>(rng() % static_cast<unsigned>(n - 1));
            int j = static_cast<int>(rng() % static_cast<unsigned>(n - 1));
            if (i > j) std::swap(i, j);
            if (i == 0 && j == n - 1) continue;               // 整圈反转 = 原样，跳过
            const int prev = tour[static_cast<std::size_t>((i - 1 + n) % n)];
            const int a = tour[static_cast<std::size_t>(i)];
            const int b = tour[static_cast<std::size_t>(j)];
            const int nxt = tour[static_cast<std::size_t>((j + 1) % n)];
            const int delta = d[static_cast<std::size_t>(prev)][static_cast<std::size_t>(b)] +
                              d[static_cast<std::size_t>(a)][static_cast<std::size_t>(nxt)] -
                              d[static_cast<std::size_t>(prev)][static_cast<std::size_t>(a)] -
                              d[static_cast<std::size_t>(b)][static_cast<std::size_t>(nxt)];
            if (delta <= 0 || u(rng) < std::exp(-static_cast<double>(delta) / T)) {
                if (delta > 0) ++accWorse;                    // 记录「接受变差」的次数
                std::reverse(tour.begin() + static_cast<std::ptrdiff_t>(i), tour.begin() + static_cast<std::ptrdiff_t>(j + 1));
                cur += delta;
                if (cur < bestLen) { bestLen = cur; best = tour; }
            }
        }
    }
    // 2-opt 局部收尾（只接受改进，确保是局部最优）
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j) {
                if (i == 0 && j == n - 1) continue;
                const int prev = tour[static_cast<std::size_t>((i - 1 + n) % n)];
                const int a = tour[static_cast<std::size_t>(i)];
                const int b = tour[static_cast<std::size_t>(j)];
                const int nxt = tour[static_cast<std::size_t>((j + 1) % n)];
                const int delta = d[static_cast<std::size_t>(prev)][static_cast<std::size_t>(b)] +
                                  d[static_cast<std::size_t>(a)][static_cast<std::size_t>(nxt)] -
                                  d[static_cast<std::size_t>(prev)][static_cast<std::size_t>(a)] -
                                  d[static_cast<std::size_t>(b)][static_cast<std::size_t>(nxt)];
                if (delta < 0) {
                    std::reverse(tour.begin() + i, tour.begin() + j + 1);
                    improved = true;
                }
            }
    }
    const int finalLen = tourLen(d, tour);
    if (finalLen < bestLen) { bestLen = finalLen; best = tour; }
    return {best, bestLen, accWorse, steps};
}

} // namespace sa

int main() {
    std::cout << "########## 模拟退火：旅行商问题 ##########\n\n";

    std::cout << "===== 1) 小规模：SA vs 精确最优（n=9） =====\n";
    {
        std::mt19937 rng(20260913);
        const sa::Matrix d = sa::makeCities(9, rng);
        const int opt = sa::bruteOptimal(d);
        const auto nn = sa::nearestNeighbor(d);
        const int nnLen = sa::tourLen(d, nn);
        std::mt19937 r2(12345);
        const double T0 = sa::initialTemperature(d, nn, r2);
        const auto r = sa::anneal(d, r2, T0, 0.9995, 0.01 * T0, 300, nn);
        std::cout << "  自适应起始温度 T0 = " << T0 << "\n";
        std::cout << "  精确最优     = " << opt << "\n";
        std::cout << "  最近邻(贪心) = " << nnLen << "  （比最优差 " << (100.0 * (nnLen - opt) / opt) << "%）\n";
        std::cout << "  模拟退火     = " << r.length << "  （" << (r.length == opt ? "达到最优 ✓" : "未达最优") << "）\n";
        std::cout << "  接受「变差」移动次数 = " << r.acceptedWorse << "（正是它帮助跳出局部最优）\n";
        assert(r.length >= opt);
        std::cout << "  -> n 小时退火能稳稳拿到精确最优，且确实靠「接受坏解」逃出了局部最优。\n\n";
    }

    std::cout << "===== 2) 中等规模：SA 显著优于最近邻（n=40） =====\n";
    {
        std::mt19937 rng(20260914);
        const sa::Matrix d = sa::makeCities(40, rng);
        const auto nn = sa::nearestNeighbor(d);
        const int nnLen = sa::tourLen(d, nn);
        std::mt19937 r2(999);
        const double T0 = sa::initialTemperature(d, nn, r2);
        const auto r = sa::anneal(d, r2, T0, 0.9997, 0.01 * T0, 300, nn);
        std::cout << "  自适应起始温度 T0 = " << T0 << "\n";
        std::cout << "  最近邻(贪心) = " << nnLen << "\n";
        std::cout << "  模拟退火     = " << r.length << "  （比最近邻好 "
                  << (100.0 * (nnLen - r.length) / nnLen) << "%）\n";
        std::cout << "  退火步数 = " << r.steps << "，接受变差 = " << r.acceptedWorse << "\n";
        assert(r.length <= nnLen);
    }

    std::cout << "===== 3) 起始温度：太冷「冻住」、太热「乱走」 =====\n";
    {
        std::mt19937 rng(20260915);
        const sa::Matrix d = sa::makeCities(30, rng);
        const auto nn = sa::nearestNeighbor(d);
        std::mt19937 r0(7);
        const double Tstar = sa::initialTemperature(d, nn, r0);     // 自适应参考温度
        std::cout << "  参考温度 T* = " << Tstar << "（由「典型 Δcost」定）\n";
        std::cout << "  T0/T*\t\tT0\t\t终止长度\t接受变差次数\n";
        for (double mult : {0.01, 0.1, 1.0, 10.0, 100.0}) {
            std::mt19937 r2(7);
            const double T0 = mult * Tstar;
            const auto r = sa::anneal(d, r2, T0, 0.998, 0.01 * T0, 250, nn);
            std::cout << "  " << mult << "\t\t" << T0 << "\t\t" << r.length << "\t\t" << r.acceptedWorse << "\n";
        }
        std::cout << "  -> T0 太小：几乎不接受坏解，退化成爬山，困在局部最优；\n";
        std::cout << "     T0 与 Δcost 同量级（T* 附近）：既能逃出局部最优、又来得及收敛。\n\n";
    }

    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
