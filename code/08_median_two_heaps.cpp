// 08_median_two_heaps.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 08 章 优先级队列与堆》
// 主题：对顶堆（two-heap）求数据流的中位数
//
//   维护两个堆，把数据流一分为二：
//     * lo：**大顶堆**，存较小的一半（堆顶 = 小半部分的最大值）；
//     * hi：**小顶堆**，存较大的一半（堆顶 = 大半部分的最小值）。
//   不变式：lo.size() == hi.size() 或 lo.size() == hi.size() + 1。
//
//   于是中位数 = lo.size() > hi.size() ? lo.top()
//                                      : (lo.top() + hi.top()) / 2.
//   addNum 只做常数次堆操作 -> O(log n)；median() 是 O(1)。
//
//   相比「每来一个数就排序一次」的 O(n log n) / 次，或者「维护有序数组插入」
//   的 O(n) / 次，对顶堆是数据流中位数的标准解法。
//
//   AI 关联：训练里监控 loss / reward 的**滑动分位数**（P50、P90）就是它的近亲；
//   「流式统计」在监控、采样、异常检测里到处都用得上。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 08_median_two_heaps.cpp -o 08_median_two_heaps && ./08_median_two_heaps
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 08_median_two_heaps.cpp -o md2_san && ./md2_san
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>        // std::fabs
#include <cstddef>
#include <functional>   // std::greater
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

class MedianFinder {
public:
    void addNum(int x) {
        // ① 先塞进"小半部分"（大顶堆）
        lo_.push(x);
        // ② 把 lo 的最大值挪到 hi —— 保证「lo 里每个元素 <= hi 里每个元素」
        hi_.push(lo_.top());
        lo_.pop();
        // ③ 再平衡：让 lo 的 size 要么等于、要么比 hi 大 1
        if (hi_.size() > lo_.size()) {
            lo_.push(hi_.top());
            hi_.pop();
        }
    }

    double median() const {
        if (lo_.empty()) return 0.0;
        if (lo_.size() > hi_.size()) return double(lo_.top());
        return (double(lo_.top()) + double(hi_.top())) / 2.0;
    }

    std::size_t size() const { return lo_.size() + hi_.size(); }

    bool invariantsHold() const {
        if (lo_.size() < hi_.size() || lo_.size() > hi_.size() + 1) return false;
        if (!lo_.empty() && !hi_.empty() && lo_.top() > hi_.top()) return false;   // 小半 <= 大半
        return true;
    }

private:
    std::priority_queue<int> lo_;                                   // 大顶堆：较小的一半
    std::priority_queue<int, std::vector<int>, std::greater<int>> hi_;  // 小顶堆：较大的一半
};

// 朴素中位数：每次都排序（用作对拍基准）
static double naiveMedian(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if (n == 0) return 0.0;
    if (n % 2 == 1) return double(v[n / 2]);
    return (double(v[n / 2 - 1]) + double(v[n / 2])) / 2.0;
}

int main() {
    std::cout << "======== 08 对顶堆：数据流中位数 ========\n\n";

    // ---------- 演示：逐个插入，观察中位数 ----------
    std::cout << "=== 1. 逐个插入，中位数实时更新 ===\n";
    MedianFinder mf;
    std::vector<int> seen;
    for (int x : {6, 10, 2, 6, 5, 0, 6, 3, 1, 0, 0}) {
        mf.addNum(x);
        seen.push_back(x);
        assert(mf.invariantsHold());
        std::cout << "  插入 " << std::setw(2) << x << " -> 中位数 = "
                  << std::fixed << std::setprecision(1) << mf.median()
                  << "   (n = " << mf.size() << ")\n";
    }

    // ---------- 对拍验证 ----------
    std::cout << "\n=== 2. 与「每次排序求中位数」对拍 ===\n";
    std::mt19937 rng(20260913);
    for (int trial = 0; trial < 200; ++trial) {
        MedianFinder m;
        std::vector<int> v;
        const std::size_t n = 1 + rng() % 300;
        for (std::size_t i = 0; i < n; ++i) {
            const int x = static_cast<int>(rng() % 2000) - 1000;
            m.addNum(x);
            v.push_back(x);
            assert(m.invariantsHold());
            // 每步都校验（小规模，可以承受）
            std::vector<int> copy = v;
            assert(std::fabs(m.median() - naiveMedian(copy)) < 1e-9);
        }
    }
    std::cout << "  200 组随机数据流：对顶堆中位数 == 每次排序 == 全程不变式成立 ✓\n";

    // ---------- 极端：单调递增 / 单调递减 ----------
    std::cout << "\n=== 3. 单调序列（最容易暴露平衡 bug）===\n";
    {
        MedianFinder inc;                    // 1,2,3,...,1001
        for (int i = 1; i <= 1001; ++i) { inc.addNum(i); assert(inc.invariantsHold()); }
        std::cout << "  1..1001 递增：中位数 = " << inc.median() << "（应为 501）\n";
        assert(inc.median() == 501.0);
    }
    {
        MedianFinder dec;                    // 1001,1000,...,1
        for (int i = 1001; i >= 1; --i) { dec.addNum(i); assert(dec.invariantsHold()); }
        std::cout << "  1001..1 递减：中位数 = " << dec.median() << "（应为 501）\n";
        assert(dec.median() == 501.0);
    }
    {
        MedianFinder even;                   // 偶数个：中位数取平均
        for (int i = 1; i <= 10; ++i) even.addNum(i);
        std::cout << "  1..10 偶数个：中位数 = " << even.median() << "（应为 5.5）\n";
        assert(even.median() == 5.5);
    }

    std::cout << "\n=== 4. 复杂度与适用场景 ===\n"
                 "  * addNum：O(log n)（几次堆操作）；median：O(1)；空间 O(n)；\n"
                 "  * 对比：每来一个数排序 O(n log n)/次；维护有序数组插入 O(n)/次；\n"
                 "  * 数据流（不可回放、要求实时）场景下，对顶堆几乎是唯一高效解；\n"
                 "  * 变体：求**滑动窗口中位数**时，两个堆要配合「惰性删除」处理过期元素；\n"
                 "  * AI 落点：训练监控里的滑动 P50/P90/P99、流式异常检测分位数。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
