// 12_quickselect_bfprt.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 12 章 排序》
// 主题：第 k 小元素的选择问题 —— 快选（Quickselect，期望 O(n)）与
//       BFPRT（中位数的中位数，最坏 O(n)）
//
//   问题：给 n 个数，找**第 k 小**的那个（中位数就是 k = n/2 的特例）。
//   最朴素：先排序再取第 k 个 —— O(n log n)。
//   但「排序」做了太多无用功：我们只要一个元素，却把 n 个数全排好了。
//
//   ★ 快选（Quickselect）：借用快排的划分，但**每次只往一边递归**。
//     划分一次把数组切成「<= pivot / > pivot」，看第 k 小落在哪边，
//     只递归那一边。期望每次规模减半 ->
//       T(n) = n + n/2 + n/4 + ... = 2n = **O(n)**（期望）。
//     但它和快排一样，**最坏 O(n^2)**（pivot 总取到极值）。
//
//   ★ BFPRT（Blum–Floyd–Pratt–Rivest–Tarjan, 1973）：用「**中位数的中位数**」
//     当 pivot，保证每次都至少切掉 3n/10，于是**最坏也是 O(n)**。
//     代价是常数大（分组、求各组中位数），实践中常比随机快选慢，
//     但它给出了「选择问题线性时间可解」的**理论保证**。
//
//   本文件：正确性对拍 + 「只排序前 k 个」的对比 + 随机快选 / BFPRT / 全排序基准。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 12_quickselect_bfprt.cpp -o 12_quickselect_bfprt && ./12_quickselect_bfprt
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 12_quickselect_bfprt.cpp -o qs_san && ./qs_san
// ---------------------------------------------------------------------------

#include <algorithm>      // std::sort, std::swap, std::min
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

// ---- 三路划分：把 a[lo,hi) 切成 [<pivot][==pivot][>pivot]，返回边界 (lt, gt) ----
template <class T>
static void partition3(std::vector<T>& a, std::size_t lo, std::size_t hi, const T& pivot,
                       std::size_t& lt, std::size_t& gt) {
    lt = lo; gt = hi;
    std::size_t i = lo;
    while (i < gt) {
        if (a[i] < pivot)      std::swap(a[lt++], a[i++]);
        else if (pivot < a[i]) std::swap(a[i], a[--gt]);
        else                   ++i;
    }
}

// ===========================================================================
// 快选（Quickselect）：随机 pivot，期望 O(n)。返回第 k 小（k 从 0 计）。
// ===========================================================================
template <class T>
T quickselect(std::vector<T>& a, std::size_t k, std::mt19937& rng) {
    assert(!a.empty() && k < a.size());
    std::size_t lo = 0, hi = a.size();
    while (true) {
        if (hi - lo == 1) return a[lo];
        const std::size_t p = lo + rng() % (hi - lo);
        const T pivot = a[p];
        std::size_t lt = 0, gt = 0;
        partition3(a, lo, hi, pivot, lt, gt);
        if (k < lt)        hi = lt;                 // 第 k 小在左段
        else if (k >= gt)  lo = gt;                 // 第 k 小在右段
        else               return pivot;            // 落在「== pivot」段
    }
}

// ===========================================================================
// BFPRT / 中位数的中位数：最坏 O(n)。递归求「每组 5 个的中位数的中位数」当 pivot。
// ===========================================================================
template <class T>
static T selectBFPRT(std::vector<T>& a, std::size_t lo, std::size_t hi, std::size_t k) {
    while (true) {
        if (hi - lo <= 5) {                         // 小规模直接排好，直取
            std::sort(a.begin() + static_cast<std::ptrdiff_t>(lo), a.begin() + static_cast<std::ptrdiff_t>(hi));
            return a[k];
        }
        std::vector<T> medians;                     // 每 5 个一组，取组内中位数
        medians.reserve((hi - lo) / 5 + 1);
        for (std::size_t i = lo; i < hi; i += 5) {
            const std::size_t gHi = std::min(i + 5, hi);
            std::sort(a.begin() + static_cast<std::ptrdiff_t>(i), a.begin() + static_cast<std::ptrdiff_t>(gHi));
            medians.push_back(a[i + (gHi - i) / 2]);
        }
        std::vector<T> tmp = medians;               // 用副本递归，避免打乱 a 供后续划分
        const T pivot = selectBFPRT(tmp, 0, tmp.size(), tmp.size() / 2);
        std::size_t lt = 0, gt = 0;
        partition3(a, lo, hi, pivot, lt, gt);
        if (k < lt)        hi = lt;
        else if (k >= gt)  lo = gt;
        else               return pivot;
    }
}
template <class T>
T bfprtSelect(std::vector<T>& a, std::size_t k) {
    assert(!a.empty() && k < a.size());
    return selectBFPRT(a, 0, a.size(), k);
}

static double msSince(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    std::cout << "======== 12 第 k 小：快选 / BFPRT ========\n\n";

    // ---------- 1. 正确性对拍：两者都 == 排序后取第 k 个 ----------
    std::cout << "=== 1. ★ 对拍：快选 / BFPRT == 排序后取第 k 个 ===\n";
    {
        std::mt19937 rng(20260913);
        int tested = 0;
        for (int trial = 0; trial < 3000; ++trial) {
            const std::size_t n = 1 + rng() % 80;
            const int range = 1 + static_cast<int>(rng() % 30);
            std::vector<int> base(n);
            for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<int>(rng() % static_cast<unsigned>(range));
            std::vector<int> sorted = base;
            std::sort(sorted.begin(), sorted.end());
            const std::size_t k = rng() % n;

            std::vector<int> v1 = base;
            std::mt19937 r(999);
            assert(quickselect(v1, k, r) == sorted[k]);
            std::vector<int> v2 = base;
            assert(bfprtSelect(v2, k) == sorted[k]);
            ++tested;
        }
        std::cout << "  " << tested << " 组随机（含大量重复键）：快选、BFPRT 都 == 排序后第 k 个 ✓\n";
    }

    // ---------- 2. 中位数 & 第 k 大 ----------
    std::cout << "\n=== 2. 用法：中位数、第 k 大、Top-k 阈值 ===\n";
    {
        std::vector<int> a = {7, 1, 9, 3, 5, 8, 2, 6, 4, 0};
        std::vector<int> sorted = a;
        std::sort(sorted.begin(), sorted.end());
        std::vector<int> v = a;
        std::mt19937 rng(1);
        const std::size_t n = v.size();
        const int median = quickselect(v, n / 2, rng);              // 第 n/2 小
        std::vector<int> w = a;
        const int kthLargest = bfprtSelect(w, n - 1 - 2);           // 第 3 大 = 第 (n-1-2) 小
        std::cout << "  数组 = 7 1 9 3 5 8 2 6 4 0\n";
        std::cout << "  中位数（第 " << n / 2 << " 小）= " << median << "（排序后 = " << sorted[n / 2] << "）\n";
        std::cout << "  第 3 大 = " << kthLargest << "（排序后 = " << sorted[n - 3] << "）\n";
        std::cout << "  -> 只要一个分位数时，不必 O(n log n) 全排序，O(n) 取出来即可。\n";
        assert(median == sorted[n / 2] && kthLargest == sorted[n - 3]);
    }

    // ---------- 3. ★ 基准：全排序 vs 快选 vs BFPRT ----------
    std::cout << "\n=== 3. ★ 基准：n = 200,000，取中位数 k = n/2 ===\n";
    {
        const std::size_t n = 200000;
        std::mt19937 rng(20260913);
        std::vector<int> base(n);
        for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<int>(rng());

        const int reps = 30;
        double tSort = 0, tQuick = 0, tBfprt = 0;
        int ref = 0;
        for (int r = 0; r < reps; ++r) {
            {   std::vector<int> v = base; auto t0 = std::chrono::steady_clock::now();
                std::sort(v.begin(), v.end()); tSort += msSince(t0); ref = v[n / 2]; }
            {   std::vector<int> v = base; std::mt19937 rg(100 + static_cast<unsigned>(r));
                auto t0 = std::chrono::steady_clock::now();
                int got = quickselect(v, n / 2, rg); tQuick += msSince(t0); assert(got == ref); }
            {   std::vector<int> v = base; auto t0 = std::chrono::steady_clock::now();
                int got = bfprtSelect(v, n / 2); tBfprt += msSince(t0); assert(got == ref); }
        }
        std::cout << "  方法                平均耗时(ms)\n";
        std::cout << "  全排序后取中位数    " << tSort / reps << "\n";
        std::cout << "  快选 Quickselect    " << tQuick / reps << "（期望 O(n)，比全排序快几倍）\n";
        std::cout << "  BFPRT 中位数的中位数 " << tBfprt / reps << "（最坏 O(n)，但常数大）\n";
        std::cout << "  -> 快选通常最快；BFPRT 慢在「分组 + 反复排序小段 + 递归」的常数上，\n";
        std::cout << "     但它的卖点是**最坏也能线性**——对抗性输入下不会退化。\n";
        assert(tQuick < tSort);
    }

    // ---------- 4. ★ 快选也怕「坏人」：有序输入 + 固定 pivot 会退化 ----------
    std::cout << "\n=== 4. ★ 随机快选 vs BFPRT 在最坏输入上的行为 ===\n";
    {
        // 已升序数组：随机 pivot 的快选仍然很快（随机化就破了有序性）；
        // 真正能整死「固定 pivot」的输入，这里用「几乎有序」来近似体感。
        const std::size_t n = 200000;
        std::vector<int> sorted(n);
        for (std::size_t i = 0; i < n; ++i) sorted[i] = static_cast<int>(i);
        std::vector<int> v = sorted;
        std::mt19937 rng(7);
        auto t0 = std::chrono::steady_clock::now();
        const int got = quickselect(v, n / 2, rng);
        const double tq = msSince(t0);
        std::cout << "  已升序 n=200000，随机快选取中位数：耗时 " << tq << " ms，答案 = " << got << "（对）\n";
        std::cout << "  -> 随机化把「有序 / 逆序」这类最坏输入化解掉了；\n";
        std::cout << "     若坚持取首元当 pivot，有序输入会让快选也退化成 O(n^2)（与快排同理）。\n";
        assert(got == static_cast<int>(n / 2));
    }

    std::cout << "\n=== 5. 复杂度与要点 ===\n"
                 "  * 第 k 小（选择问题）：全排序 O(n log n)；快选**期望 O(n)**；BFPRT **最坏 O(n)**；\n"
                 "  * 快选 = 快排划分 + 「只递归有 k 的那一边」，无递归的那侧直接丢弃；\n"
                 "  * BFPRT 用「中位数的中位数」当 pivot，保证每轮至少切掉 3n/10（故最坏线性）；\n"
                 "  * 工程默认随机快选（快、常数小），BFPRT 主要作为「最坏保证」的理论备胎；\n"
                 "  * 只需 Top-k 且 k 很小时，用**大小为 k 的堆** O(n log k) 更省（见第 08 章）。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
