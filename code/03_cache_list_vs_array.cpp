// 03_cache_list_vs_array.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 03 章 链表》
// 主题：为什么「理论一样」的两种遍历，实测能差好几倍？—— 缓存局部性
//
//   两个实验，都把 n 个 int 求和，加法次数完全相同：
//     实验 1：数组遍历（连续内存，顺序访问，硬件预取器友好）
//     实验 2：链表遍历（结点由 new 逐个分配，内存散落，指针追逐）
//   数组 O(n)、链表 O(n)，理论上「一样快」。实测链表往往慢数倍 ——
//   不是算法差，而是缓存不友好：
//     * 数组：读 a[0] 顺带把整条 cache line（16 个 int）拉进 cache，
//             后续 15 次几乎免费；预取器还能提前把后面的线拉进来。
//     * 链表：每个结点的 next 指向一块「随机」地址，每次都要等一次
//             主存访问（~100 ns），预取器无从下手 —— 这就是「指针追逐
//             （pointer chasing）」的代价。
//
//   这解释了三条重要工程事实：
//     1. 为什么 std::vector 往往比 std::list 快，即使中间插入后者才 O(1)；
//     2. 为什么 PyTorch 张量坚持「连续内存」，哪怕要多做一次 copy；
//     3. 为什么 GPU 极度偏好连续、可合并（coalesced）的访存 ——
//        链表的随机访存会把带宽优势彻底葬送。
//
// 编译运行（-O2 下结论才可信）：
//   g++ -std=c++17 -O2 -Wall -Wextra 03_cache_list_vs_array.cpp -o cache && ./cache
// ---------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <vector>

static volatile std::int64_t g_sink = 0;   // 防止死代码消除

struct LNode {
    int    val;
    LNode* next;
};

// 建一条「结点物理位置随机散落」的链表。
// 做法：先整块分配 nodes[n]，再用一个随机排列 order[] 决定遍历顺序，
// 让 next 指向的地址在物理上不连续 —— 这样才能暴露指针追逐的代价。
// 返回值：*baseOut = 整块的基址（用于最终释放），*headOut = 链头。
static LNode* buildShuffledList(std::size_t n, std::uint32_t seed,
                                LNode** headOut) {
    LNode*      nodes = new LNode[n];
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;

    // 简单的线性同余打乱（避免依赖库随机数的实现差异）
    std::uint32_t s = seed;
    for (std::size_t i = n; i > 1; --i) {
        s = s * 1664525u + 1013904223u;
        std::size_t j = s % i;
        std::swap(order[i - 1], order[j]);
    }
    for (std::size_t i = 0; i < n; ++i) nodes[order[i]].val = static_cast<int>(order[i]);
    for (std::size_t i = 0; i + 1 < n; ++i) nodes[order[i]].next = &nodes[order[i + 1]];
    nodes[order[n - 1]].next = nullptr;

    *headOut = &nodes[order[0]];
    return nodes;
}

template <typename Fn>
static double bestOf(Fn&& fn, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) fn();
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

int main() {
    const std::size_t N = std::size_t(1) << 22;     // 约 419 万个 int（16 MB）
    std::cout << "n = " << N << " 个 int（数组约 "
              << (N * sizeof(int)) / (1024 * 1024) << " MB）\n\n";

    // 数组：值 0..n-1
    std::vector<int> a(N);
    for (std::size_t i = 0; i < N; ++i) a[i] = static_cast<int>(i);

    // 链表：结点散落，next 指向随机物理位置
    LNode* head = nullptr;
    LNode* base = buildShuffledList(N, 20260914u, &head);

    // 求和（两者加法次数都是 n）
    auto sumArray  = [&] { std::int64_t s = 0; for (int x : a) s += x; g_sink = s; };
    auto sumList   = [&] { std::int64_t s = 0; for (LNode* p = head; p; p = p->next) s += p->val; g_sink = s; };

    const double tArr = bestOf(sumArray, /*warmup=*/2, /*reps=*/7);
    const double tLst = bestOf(sumList,  /*warmup=*/2, /*reps=*/7);

    auto bandwidth = [&](double sec) {
        const double bytes = static_cast<double>(N) * sizeof(int);
        return bytes / sec / 1e9;                   // GB/s
    };

    std::cout << std::left << std::setw(22) << "实现"
              << std::setw(14) << "时间(ms)"
              << std::setw(20) << "有效带宽(GB/s)" << "说明\n";
    std::cout << std::string(74, '-') << "\n";
    std::cout << std::left << std::setw(22) << "数组 vector<int>"
              << std::setw(14) << std::fixed << std::setprecision(3) << (tArr * 1e3)
              << std::setw(20) << std::setprecision(2) << bandwidth(tArr)
              << "顺序访问，cache 命中率高\n";
    std::cout << std::left << std::setw(22) << "链表 LNode*"
              << std::setw(14) << std::setprecision(3) << (tLst * 1e3)
              << std::setw(20) << std::setprecision(2) << bandwidth(tLst)
              << "指针追逐，几乎全 miss\n";
    std::cout << "\n链表 / 数组 耗时比 = " << std::setprecision(2) << (tLst / tArr) << "x\n";

    // 自查：两种求和的「和」必须一致（否则说明链表建错了）
    std::int64_t sArr = 0, sLst = 0;
    for (int x : a) sArr += x;
    for (LNode* p = head; p; p = p->next) sLst += p->val;
    if (sArr != sLst) { std::cerr << "两实现结果不一致！\n"; return 1; }

    std::cout << "\n结论：加法次数完全相同（都是 n = " << N << "），\n"
                 "      但链表因为「指针追逐 + cache miss」明显更慢。\n"
                 "      这就是 PyTorch 坚持连续内存、GPU 偏好 coalesced 访存的根本原因。\n";

    delete[] base;                   // 释放整块（注意必须用 base，不能用 head）
    return 0;
}
