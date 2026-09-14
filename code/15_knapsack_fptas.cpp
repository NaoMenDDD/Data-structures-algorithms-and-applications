// ===========================================================================
// 15_knapsack_fptas.cpp
// 0/1 背包的 FPTAS（Fully Polynomial-Time Approximation Scheme）
//   - 精确 DP：O(n*cap)，与容量成正比（容量大就慢）
//   - ★ FPTAS：把「价值」按 K = eps*vmax/n 缩放取整，在「缩小的价值域」上做
//     O(n * sum(v')) 的精确 DP，再乘回 K。运行时间 poly(n, 1/eps) —
//     对任意 eps>0，保证 (1-eps)*OPT <= A <= OPT。
//
// 思路要点：背包有两维（价值 / 重量）。精确 DP 在「重量」维上做（O(n*cap)），
//   容量大就崩。FPTAS 改在「价值」维上做（O(n*sum v')），而把价值**粗粒度化**
//   （除以 K 再取整）—— 代价是精度，收益是 sum(v') 被压到 poly(n/eps)。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 15_knapsack_fptas.cpp -o 15_knapsack_fptas && ./15_knapsack_fptas
// ===========================================================================
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace kn {

// 精确 DP（在容量维）：O(n*cap)
int exactDP(const std::vector<int>& w, const std::vector<int>& v, int cap) {
    std::vector<int> f(static_cast<std::size_t>(cap) + 1, 0);
    const int n = static_cast<int>(w.size());
    for (int i = 0; i < n; ++i)
        for (int c = cap; c >= w[static_cast<std::size_t>(i)]; --c)
            f[static_cast<std::size_t>(c)] = std::max(f[static_cast<std::size_t>(c)],
                                                     f[static_cast<std::size_t>(c - w[static_cast<std::size_t>(i)])] + v[static_cast<std::size_t>(i)]);
    return f[static_cast<std::size_t>(cap)];
}

// FPTAS（在「缩放后的价值」维）：返回 (近似值, 缩放价值域大小, 缩放后的整数最优)
struct FptasResult { double value; long long Vprime; int scaledOpt; };

FptasResult fptas(const std::vector<int>& w, const std::vector<int>& v, int cap, double eps) {
    const int n = static_cast<int>(w.size());
    const int vmax = *std::max_element(v.begin(), v.end());
    const double K = std::max(1e-9, eps * vmax / n);                 // 缩放因子
    std::vector<int> vp(static_cast<std::size_t>(n));                // v'_i = floor(v_i / K)
    long long sumVp = 0;
    for (int i = 0; i < n; ++i) { vp[static_cast<std::size_t>(i)] = static_cast<int>(std::floor(v[static_cast<std::size_t>(i)] / K)); sumVp += vp[static_cast<std::size_t>(i)]; }
    const long long INF = std::numeric_limits<long long>::max() / 4;
    std::vector<long long> minW(static_cast<std::size_t>(sumVp) + 1, INF);   // minW[t] = 达到缩放价值 t 的最小重量
    minW[0] = 0;
    for (int i = 0; i < n; ++i)
        for (long long t = sumVp; t >= vp[static_cast<std::size_t>(i)]; --t) {
            const long long prev = minW[static_cast<std::size_t>(t - vp[static_cast<std::size_t>(i)])];
            if (prev + w[static_cast<std::size_t>(i)] < minW[static_cast<std::size_t>(t)])
                minW[static_cast<std::size_t>(t)] = prev + w[static_cast<std::size_t>(i)];
        }
    int scaledOpt = 0;
    for (long long t = sumVp; t >= 0; --t)
        if (minW[static_cast<std::size_t>(t)] <= cap) { scaledOpt = static_cast<int>(t); break; }
    return {scaledOpt * K, sumVp, scaledOpt};
}

} // namespace kn

int main() {
    std::cout << "########## 背包 FPTAS：用精度换时间 ##########\n\n";

    std::mt19937 rng(20260913);
    const int n = 100, cap = 1000;
    std::vector<int> w(static_cast<std::size_t>(n)), v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        w[static_cast<std::size_t>(i)] = 1 + static_cast<int>(rng() % 50);
        v[static_cast<std::size_t>(i)] = 1 + static_cast<int>(rng() % 1000);
    }
    const int opt = kn::exactDP(w, v, cap);
    std::cout << "===== 1) 实例：n=" << n << "，容量=" << cap << "，精确最优 OPT=" << opt << " =====\n\n";

    std::cout << "===== 2) FPTAS：eps 越小越准（但越慢） =====\n";
    std::cout << "  eps\t近似值\t\t比值 A/OPT\t保证下界(1-eps)\t达标?\t耗时(ms)\t缩放价值域\n";
    for (double eps : {0.5, 0.2, 0.1, 0.05, 0.02, 0.01}) {
        const auto t0 = std::chrono::steady_clock::now();
        const kn::FptasResult r = kn::fptas(w, v, cap, eps);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double ratio = r.value / opt;
        const bool ok = (r.value <= opt + 1e-6) && (ratio >= 1.0 - eps - 1e-9);
        std::cout << "  " << eps << "\t" << std::fixed << std::setprecision(1) << r.value
                  << "\t\t" << std::setprecision(4) << ratio << "\t\t" << (1.0 - eps)
                  << "\t\t" << (ok ? "yes" : "NO") << "\t" << std::setprecision(2) << ms
                  << "\t\t" << r.Vprime << "\n";
        assert(ok);                                                  // FPTAS 的硬保证
    }
    std::cout << "\n  -> 缩放价值域 = sum(v') ≈ n*vmax/K = n^2/eps，精确 DP 复杂度 O(n * sum v') = O(n^3/eps)；\n";
    std::cout << "     故运行时间 poly(n, 1/eps)。eps 减半 -> 价值域翻倍 -> 变慢，但精度上升。\n\n";

    std::cout << "===== 3) 与「不用近似」的精确 DP 对比 =====\n";
    {
        std::cout << "  精确 DP 复杂度 O(n*cap) = " << (static_cast<long long>(n) * cap)
                  << "（与容量成正比，容量大就崩）\n";
        // 极端：容量很大时，精确 DP 慢，FPTAS 不变（它只看价值）
        const int bigCap = 100000000;
        std::cout << "  若容量 = 1 亿：精确 DP 需 O(n*cap) = " << (static_cast<long long>(n) * bigCap)
                  << "（~" << (static_cast<double>(n) * bigCap / 1e9) << " G 次）——不可行；\n";
        std::cout << "  而 FPTAS 只看价值域，与容量无关（只要容量够大容得下），仍可在 ms 级求解。\n";
        std::cout << "  -> 这就是「伪多项式（O(n*cap)）」与「多项式（O(n^3/eps)）」的区别。\n\n";
    }

    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
