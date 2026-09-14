// 08_heap_sort.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 08 章 优先级队列与堆》
// 主题：堆排序（heapsort）—— 唯一「原地 + 最坏 O(n log n)」的比较排序之一
//
//   两阶段：
//     ① 建堆：把整个数组 O(n) 调整成大顶堆（自底向上 heapify）；
//     ② 反复取最大：把堆顶（全局最大）与「当前堆的最后一个元素」交换，
//        堆大小减一，再对新的堆顶 sift-down 恢复堆序。
//        第 i 次交换后，数组末尾第 i 个位置就被最终确定了。
//
//   复杂度：建堆 O(n) + n 次 sift-down 各 O(log n) = O(n log n)，
//          且是**最坏情况**保证（不像快排会退化）。空间 O(1)（原地）。
//
//   本文件还实测对比 std::sort（内省排序），说明为什么工程上快排通常更快：
//   堆排序的 sift-down 在数组上**跳跃访问**（下标 2i+1、2i+2），cache 局部性差。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 08_heap_sort.cpp -o 08_heap_sort && ./08_heap_sort
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
static volatile long long g_sink = 0;

inline std::size_t lc(std::size_t i) { return 2 * i + 1; }
inline std::size_t rc(std::size_t i) { return 2 * i + 2; }

// sift-down：在下标 [0, n) 范围内把 i 处的元素下沉，维持大顶堆
static void siftDown(std::vector<int>& a, std::size_t n, std::size_t i) {
    for (;;) {
        std::size_t big = i;
        const std::size_t l = lc(i), r = rc(i);
        if (l < n && a[l] > a[big]) big = l;
        if (r < n && a[r] > a[big]) big = r;
        if (big == i) break;
        std::swap(a[big], a[i]);
        i = big;
    }
}

// 原地堆排序（升序）
static void heapSort(std::vector<int>& a) {
    const std::size_t n = a.size();
    // ① 建大顶堆
    if (n >= 2) for (std::size_t i = n / 2; i-- > 0; ) siftDown(a, n, i);
    // ② 逐个把堆顶换到末尾并缩小堆
    for (std::size_t end = n; end-- > 1; ) {
        std::swap(a[0], a[end]);        // 当前最大 -> 末尾
        siftDown(a, end, 0);            // 剩余 [0, end) 重新恢复堆序
    }
}

// 用最直观的「反复 pop 堆顶」实现堆排序（需要额外 O(n) 空间），用作对照
static std::vector<int> heapSortByPop(std::vector<int> a) {
    const std::size_t n = a.size();
    if (n >= 2) for (std::size_t i = n / 2; i-- > 0; ) siftDown(a, n, i);
    std::vector<int> out;
    out.reserve(n);
    for (std::size_t end = n; end > 0; --end) {
        out.push_back(a[0]);            // 堆顶 = 当前最大
        a[0] = a[end - 1];
        siftDown(a, end - 1, 0);
    }
    std::reverse(out.begin(), out.end());   // pop 出来是降序，反转成升序
    return out;
}

template <typename Fn>
static double timeIt(Fn&& fn, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

int main() {
    std::cout << "======== 08 堆排序 ========\n\n";

    // ---------- 正确性 ----------
    std::cout << "=== 1. 正确性（含大量随机与边界用例）===\n";
    std::mt19937 rng(20260913);
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = rng() % 200;
        std::vector<int> a(n);
        for (int& x : a) x = static_cast<int>(rng() % 1000) - 500;   // 含重复与负数
        std::vector<int> expect = a, inplace = a;

        heapSort(inplace);
        std::sort(expect.begin(), expect.end());
        assert(inplace == expect);
        assert(heapSortByPop(a) == expect);
    }
    std::cout << "  300 组随机用例：原地堆排序 == std::sort == O(n) 空间版 ✓\n";

    // 边界
    for (std::size_t n : {0u, 1u, 2u, 3u}) {
        std::vector<int> a(n, 7);
        heapSort(a);
        assert(std::is_sorted(a.begin(), a.end()));
    }
    std::vector<int> allEq(1000, 42);
    heapSort(allEq);
    assert(std::is_sorted(allEq.begin(), allEq.end()));
    std::cout << "  空数组 / 单元素 / 全相等 / 已排序：均正确 ✓\n\n";

    // ---------- 性能对比 ----------
    std::cout << "=== 2. 性能对比（n = 2,000,000）===\n";
    const std::size_t N = 2000000;
    std::vector<int> base(N);
    for (int& x : base) x = static_cast<int>(rng());

    double t_heap = 0, t_std = 0;
    {
        std::vector<int> a = base;
        t_heap = timeIt([&]{ heapSort(a); g_sink += a[0]; }, 3);
        a = base;
        t_std = timeIt([&]{ std::sort(a.begin(), a.end()); g_sink += a[0]; }, 3);
    }
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << t_heap * 1e3 << " ms   <- heapSort    (原地)\n";
    std::cout << "  " << t_std  * 1e3 << " ms   <- std::sort   (内省排序)\n";
    std::cout << "  比值 = " << std::setprecision(2) << (t_heap / t_std) << "x（堆排序更慢）\n\n";

    std::cout << "=== 3. 为什么堆排序「理论更稳」却常更慢？ ===\n"
                 "  * 堆排序：最坏 O(n log n)，原地，无递归、无额外空间 —— 保证最强；\n"
                 "  * 但 sift-down 在数组中跳跃访问（2i+1、2i+2 下标），\n"
                 "    每次比较都可能落在新的 cache line 上，**常数大、局部性差**；\n"
                 "  * std::sort 是「快排 + 堆排 + 插入排序」的内省组合：\n"
                 "    平均走快排（顺序访问、cache 友好），深度过深才回退堆排兜底。\n"
                 "  * 结论：堆排序的定位是「**保证 + 原地**」；追求平均速度则用快排。\n"
                 "    两者互补，这也是 std::sort 把堆排序当作「兜底」的原因。\n\n";

    std::cout << "（汇总 = " << g_sink << "，仅用于阻止优化）\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
