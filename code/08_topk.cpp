// 08_topk.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 08 章 优先级队列与堆》
// 主题：Top-k 问题 —— 堆最实用的日常应用
//
//   问题：从 n 个数里找出**最大的 k 个**（或第 k 大）。
//
//   方案对比：
//     * 全部排序取前 k：O(n log n)；
//     * 维护一个**大小为 k 的最小堆**：扫一遍，
//         每个元素与堆顶（当前第 k 大）比较，比它大就替换堆顶。
//       复杂度 O(n log k)，当 k << n 时远优于全排序；
//     * 若数据是**流式到达**（不能回头、内存放不下全部），
//       大小为 k 的堆是**唯一**可行的一次扫描方案。
//
//   这是「Beam Search 保留 top-k 候选」「nucleus/top-k 采样」「推荐召回」
//   等一大类 AI 任务的底层操作。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 08_topk.cpp -o 08_topk && ./08_topk
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>      // INT32_MIN
#include <functional>   // std::greater
#include <iomanip>      // std::setprecision
#include <iostream>
#include <queue>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
static volatile long long g_sink = 0;

// 用大小为 k 的最小堆求「最大的 k 个」，返回升序排列的 top-k
static std::vector<int> topK(const std::vector<int>& a, std::size_t k) {
    if (k == 0) return {};
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;   // 小顶堆
    for (int x : a) {
        if (minHeap.size() < k) minHeap.push(x);
        else if (x > minHeap.top()) { minHeap.pop(); minHeap.push(x); }
    }
    std::vector<int> out;
    out.reserve(minHeap.size());
    while (!minHeap.empty()) { out.push_back(minHeap.top()); minHeap.pop(); }
    std::reverse(out.begin(), out.end());        // 堆弹出是升序 -> 反转成降序（最大在前）
    return out;
}

// 流式 Top-k：用一个可复用的「TopKTracker」模拟数据一条条到达
class TopKTracker {
public:
    explicit TopKTracker(std::size_t k) : k_(k) {}
    void add(int x) {
        if (heap_.size() < k_) heap_.push(x);
        else if (x > heap_.top()) { heap_.pop(); heap_.push(x); }
    }
    // 当前第 k 大（堆顶）；若不足 k 个返回 INT_MIN
    int kthLargest() const { return heap_.size() < k_ ? INT32_MIN : heap_.top(); }
    std::size_t size() const { return heap_.size(); }
    std::vector<int> snapshot() const {
        auto h = heap_;
        std::vector<int> out;
        while (!h.empty()) { out.push_back(h.top()); h.pop(); }
        std::reverse(out.begin(), out.end());
        return out;
    }
private:
    std::size_t k_;
    std::priority_queue<int, std::vector<int>, std::greater<int>> heap_;
};

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
    std::cout << "======== 08 Top-k：大小为 k 的堆 ========\n\n";

    // ---------- 正确性 ----------
    std::cout << "=== 1. 正确性（随机对拍）===\n";
    std::mt19937 rng(20260913);
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = 1 + rng() % 500;
        std::vector<int> a(n);
        for (int& x : a) x = static_cast<int>(rng() % 10000);
        const std::size_t k = 1 + rng() % n;

        std::vector<int> got = topK(a, k);
        std::vector<int> s = a;
        std::sort(s.begin(), s.end(), std::greater<int>());
        std::vector<int> expect(s.begin(), s.begin() + static_cast<long>(k));
        assert(got == expect);
    }
    std::cout << "  300 组随机用例：堆求 top-k == 全排序取前 k ✓\n";

    // ---------- 流式 Top-k ----------
    std::cout << "\n=== 2. 流式 Top-k（数据不可回头）===\n";
    TopKTracker trk(3);
    for (int x : {5, 2, 9, 1, 7, 8, 3}) {
        trk.add(x);
        std::cout << "  收到 " << x << " -> 当前 3 个最大: [";
        auto snap = trk.snapshot();
        for (std::size_t i = 0; i < snap.size(); ++i) std::cout << snap[i] << (i + 1 < snap.size() ? "," : "");
        std::cout << "]  第 3 大 = " << (trk.kthLargest() == INT32_MIN ? -1 : trk.kthLargest()) << "\n";
    }
    assert((trk.snapshot() == std::vector<int>{9, 8, 7}));
    std::cout << "  最终 3 个最大 = [9,8,7] ✓（全程只需 O(k) 内存）\n";

    // ---------- 性能：O(n log k) vs O(n log n) ----------
    std::cout << "\n=== 3. 性能对比（n = 20,000,000）===\n";
    const std::size_t N = 20000000;
    std::vector<int> big(N);
    for (int& x : big) x = static_cast<int>(rng());

    for (std::size_t k : {std::size_t(10), std::size_t(1000)}) {
        std::vector<int> got;
        const double t_heap = timeIt([&]{ got = topK(big, k); g_sink += got.front(); }, 3);

        std::vector<int> copy;
        const double t_sort = timeIt([&]{
            copy = big;
            std::partial_sort(copy.begin(), copy.begin() + static_cast<long>(k), copy.end(), std::greater<int>());
            g_sink += copy.front();
        }, 3);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  k = " << k << "：\n";
        std::cout << "     堆 top-k      " << t_heap * 1e3 << " ms\n";
        std::cout << "     partial_sort  " << t_sort * 1e3 << " ms\n";
        std::cout << "     加速比        " << std::setprecision(2) << (t_sort / t_heap) << "x\n";
    }

    std::cout << "\n=== 4. 结论与 AI 关联 ===\n"
                 "  * k 越小，大小为 k 的堆优势越大：每个元素只与堆顶比较一次，\n"
                 "    仅当更大才做一次 O(log k) 替换；n 个元素总体 O(n log k)；\n"
                 "  * 流式场景（不能重扫、内存放不下全部）下，堆是唯一的一次扫描方案；\n"
                 "  * AI 里的落点：\n"
                 "      - **Top-k / nucleus 采样**：从词表 logits 里取前 k 个候选；\n"
                 "      - **Beam Search**：每步保留得分最高的 k 条候选序列；\n"
                 "      - **推荐召回 / 检索**：从海量候选里取 top-k 相似项；\n"
                 "      - **优先经验回放**：按 TD 误差取 top-k 高价值样本。\n\n";

    std::cout << "（汇总 = " << g_sink << "，仅用于阻止优化）\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
