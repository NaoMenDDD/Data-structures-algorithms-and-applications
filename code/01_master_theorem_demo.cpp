// 01_master_theorem_demo.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 01 章 算法分析基础》
// 主题：Master 定理三种情形的「数值验证」
//
//   T(n) = a·T(n/b) + n^k,   T(1) = 1
//   令 c* = log_b(a)
//     情形 1：k < c*   →  Θ(n^{c*})
//     情形 2：k == c*  →  Θ(n^{c*} log n)
//     情形 3：k > c*   →  Θ(n^k)
//
//   做法：按「递归树逐层求和」精确算出 T(n)：
//     第 d 层有 a^d 个结点，每个结点代价 (n/b^d)^k
//     所以 T(n) = Σ_{d=0}^{D-1} a^d·(n/b^d)^k  +  a^D·T(1)
//   再用两个规模 n1 < n2 估计实际增长指数
//     e = log(T(n2)/T(n1)) / log(n2/n1)
//   把它与 Master 定理预测的指数对照。
//
//   最后附：带计数的归并排序，实测比较次数 ≈ n·log2(n)。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 01_master_theorem_demo.cpp -o master && ./master
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 递归树逐层求和（精确按定义算，不是近似公式）
// ---------------------------------------------------------------------------
static long double treeSum(long double n, int a, int b, double k) {
    long double total  = 0.0L;
    long double nodes  = 1.0L;     // a^d
    long double subN   = n;        // n / b^d
    int         depth  = 0;

    while (subN > 1.0L + 1e-9L && depth < 400) {
        total += nodes * std::pow(subN, k);
        subN  /= static_cast<long double>(b);
        nodes *= static_cast<long double>(a);
        ++depth;
    }
    total += nodes;                // 叶子层：a^depth 个叶子，每个 T(1) = 1
    return total;
}

// ---------------------------------------------------------------------------
// 由实测数据估计增长指数：e ≈ log(T2/T1) / log(n2/n1)
// ---------------------------------------------------------------------------
static double measuredExponent(int a, int b, double k) {
    // n 取 b^8 与 b^24：两端跨度越大，log 因子对实测指数的「污染」越小，
    // 情形 2 的实测指数才会稳定地贴近 c*（理论上仍会略高一点）。
    const int e1 = 8, e2 = 24;                       // n = b^e
    long double n1 = 1.0L, n2 = 1.0L;
    for (int i = 0; i < e1; ++i) n1 *= static_cast<long double>(b);
    for (int i = 0; i < e2; ++i) n2 *= static_cast<long double>(b);

    const long double t1 = treeSum(n1, a, b, k);
    const long double t2 = treeSum(n2, a, b, k);
    return static_cast<double>(std::log(static_cast<double>(t2) / static_cast<double>(t1)) /
                               std::log(static_cast<double>(n2) / static_cast<double>(n1)));
}

static std::string prediction(int a, int b, double k, double* predictedExponent) {
    const double cstar = std::log(static_cast<double>(a)) / std::log(static_cast<double>(b));
    if (k < cstar - 1e-9) {
        *predictedExponent = cstar;
        return "情形 1 → Θ(n^" + std::to_string(cstar).substr(0, 5) + ")";
    }
    if (std::fabs(k - cstar) < 1e-9) {
        *predictedExponent = cstar;
        return "情形 2 → Θ(n^" + std::to_string(cstar).substr(0, 5) + " log n)";
    }
    *predictedExponent = k;
    return "情形 3 → Θ(n^" + std::to_string(k).substr(0, 5) + ")";
}

static void experiment_master() {
    struct Case { int a, b; double k; };
    const Case cases[] = {
        {4, 2, 1.0},    // c* = 2, k < c*   → 情形 1 → Θ(n²)
        {4, 2, 2.0},    // c* = 2, k == c*  → 情形 2 → Θ(n² log n)
        {4, 2, 3.0},    // c* = 2, k > c*   → 情形 3 → Θ(n³)
        {8, 2, 2.0},    // c* = 3, k < c*   → 情形 1 → Θ(n³)
        {2, 2, 1.0},    // c* = 1, k == c*  → 情形 2 → Θ(n log n) —— 归并排序
        {2, 4, 0.5},    // c* = 1/2, k == c*→ 情形 2 → Θ(√n log n)
    };

    std::cout << "=== Master 定理：预测 vs 递归树实测 ===\n\n";
    std::cout << std::left << std::setw(5)  << "a"
              << std::setw(5)  << "b"
              << std::setw(6)  << "k"
              << std::setw(10) << "log_b(a)"
              << std::setw(30) << "Master 预测"
              << std::setw(18) << "实测指数"
              << "结论\n";
    std::cout << std::string(84, '-') << "\n";

    for (const Case& c : cases) {
        double predictedExp = 0.0;
        const std::string text = prediction(c.a, c.b, c.k, &predictedExp);
        const double measured = measuredExponent(c.a, c.b, c.k);
        const double cstar = std::log(static_cast<double>(c.a)) / std::log(static_cast<double>(c.b));

        // 情形 2 的 log 因子会让指数略高于 c*，但只要 log 因子不改变指数，偏差就应很小
        const bool ok = std::fabs(measured - predictedExp) < 0.15;

        std::cout << std::left << std::setw(5)  << c.a
                  << std::setw(5)  << c.b
                  << std::setw(6)  << c.k
                  << std::setw(10) << std::setprecision(4) << cstar
                  << std::setw(30) << text
                  << std::setw(18) << std::setprecision(4) << measured
                  << (ok ? "一致 ✓" : "不一致 ✗") << "\n";
        std::cout.flush();                          // 断言失败时也能看到表格
        assert(ok && "实测指数与 Master 定理预测不符");
    }
    std::cout << "\n说明：情形 2 的实测指数会略高于 c*，因为 T ≈ n^{c*}·log n，\n"
                 "      在有限规模上 log 因子表现为「一点点额外的指数」。\n\n";
}

// ---------------------------------------------------------------------------
// 附：归并排序的比较次数 ≈ n·log2(n)
// ---------------------------------------------------------------------------
static std::uint64_t mergeSortCount(std::vector<int>& a, std::vector<int>& buf, int lo, int hi) {
    if (hi - lo <= 1) return 0;

    const int mid = lo + (hi - lo) / 2;
    std::uint64_t c = mergeSortCount(a, buf, lo, mid) + mergeSortCount(a, buf, mid, hi);

    int i = lo, j = mid, t = lo;
    while (i < mid && j < hi) {
        ++c;                                   // 一次比较
        if (a[i] <= a[j]) buf[t++] = a[i++];
        else              buf[t++] = a[j++];
    }
    while (i < mid) buf[t++] = a[i++];
    while (j < hi)  buf[t++] = a[j++];
    for (int p = lo; p < hi; ++p) a[p] = buf[p];
    return c;
}

static void experiment_merge_sort() {
    std::cout << "=== 归并排序：比较次数 vs n·log2(n) ===\n\n";
    std::cout << std::left << std::setw(10) << "n"
              << std::setw(22) << "实测比较次数"
              << std::setw(16) << "n·log2(n)"
              << "比值\n";
    std::cout << std::string(52, '-') << "\n";

    std::mt19937 rng(20260914);                // 固定种子，结果可复现
    for (int lg = 10; lg <= 18; ++lg) {
        const int n = 1 << lg;
        std::vector<int> a(n);
        std::iota(a.begin(), a.end(), 0);
        std::shuffle(a.begin(), a.end(), rng);

        std::vector<int> buf(n);
        const std::uint64_t cmp = mergeSortCount(a, buf, 0, n);

        // 排序正确性自查
        assert(std::is_sorted(a.begin(), a.end()));

        const double bound = static_cast<double>(n) * std::log2(static_cast<double>(n));
        const double ratio = static_cast<double>(cmp) / bound;

        std::cout << std::left << std::setw(10) << n
                  << std::setw(16) << cmp
                  << std::setw(16) << static_cast<std::uint64_t>(bound)
                  << std::setprecision(4) << ratio << "\n";

        assert(ratio > 0.5 && ratio < 1.2);    // 实测应在 n log n 的常数倍之内
    }
    std::cout << "\n结论：比较次数稳定在 n·log2(n) 的 1 倍以内 → Θ(n log n)。\n\n";
}

int main() {
    experiment_master();
    experiment_merge_sort();
    std::cout << "所有验证与断言通过 ✔\n";
    return 0;
}
