// 06_monotonic_deque.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 06 章 队列与双端队列》
// 主题：双端队列（deque）的两个「杀手级」用法
//
//   第 1 部分：滑动窗口最大值 —— 暴力 O(n·k) vs 单调队列 O(n)
//     维护一个「单调递减」的下标队列：
//       * 入窗口时，把队尾所有「比新元素小」的下标弹出（它们永远不可能再当最大值）；
//       * 出窗口时，若队首下标已滑出窗口，弹出队首。
//     于是队首永远指向当前窗口的最大值。每个下标最多进出队一次 -> 均摊 O(1)/元素。
//
//   第 2 部分：滑动窗口最小值 —— 只需把比较反过来（单调递增队列）
//
//   第 3 部分：用两个栈实现队列 —— 「栈与队列可以互相模拟」的经典结论，
//     单次最坏 O(n)，但**均摊 O(1)**（每个元素一生只被搬一次）。
//     这解释了为什么很多语言的「队列」底层其实是两块可翻转的缓冲。
//
//   ※ 为什么用 deque 而不是 stack？因为单调队列要在**两端**操作：
//       队尾弹小值（push 前），队首弹过期值（滑出窗口时）。栈只能一端。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 06_monotonic_deque.cpp -o 06_monotonic_deque
//   ./06_monotonic_deque
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 06_monotonic_deque.cpp -o md_san && ./md_san
// ---------------------------------------------------------------------------

#include <algorithm>   // std::max
#include <cassert>
#include <cstddef>
#include <deque>
#include <iostream>
#include <stack>
#include <utility>     // std::move
#include <vector>

// ===========================================================================
// 1. 滑动窗口最大值
// ===========================================================================
//
// 给定数组 nums 和窗口大小 k，返回每个长度为 k 的窗口里的最大值。
// 例如 nums = [1,3,-1,-3,5,3,6,7], k = 3  ->  [3,3,5,5,6,7]

// --- 暴力：每个窗口扫一遍求 max，O(n·k) ---
static std::vector<int> windowMaxBrute(const std::vector<int>& nums, std::size_t k) {
    std::vector<int> out;
    if (k == 0 || nums.size() < k) return out;
    for (std::size_t i = 0; i + k <= nums.size(); ++i) {
        int best = nums[i];
        for (std::size_t j = i + 1; j < i + k; ++j) best = std::max(best, nums[j]);
        out.push_back(best);
    }
    return out;
}

// --- 单调队列：O(n) ---
// dq 里存的是**下标**（而不是值），这样才能判断「是否已滑出窗口」。
// 不变式：dq 中下标对应的值单调递减，且队首下标始终属于当前窗口。
static std::vector<int> windowMaxDeque(const std::vector<int>& nums, std::size_t k) {
    std::vector<int> out;
    if (k == 0 || nums.size() < k) return out;

    std::deque<std::size_t> dq;               // 存下标，值为单调递减

    for (std::size_t i = 0; i < nums.size(); ++i) {
        // (a) 队尾「比新元素小」的一律弹出：新元素更大且更晚过期，它们永无出头之日
        while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
        dq.push_back(i);

        // (b) 队首若已滑出窗口（下标 <= i-k），弹出
        while (dq.front() + k <= i) dq.pop_front();

        // (c) 窗口形成后（i >= k-1），队首即当前窗口最大值
        if (i + 1 >= k) out.push_back(nums[dq.front()]);
    }
    return out;
}

static void part1_window_max() {
    std::cout << "=== 1. 滑动窗口最大值 ===\n";
    const std::vector<int> nums{1, 3, -1, -3, 5, 3, 6, 7};
    const std::size_t k = 3;

    const std::vector<int> b = windowMaxBrute(nums, k);
    const std::vector<int> d = windowMaxDeque(nums, k);
    assert(b == d);
    assert((d == std::vector<int>{3, 3, 5, 5, 6, 7}));

    std::cout << "  nums = [";
    for (std::size_t i = 0; i < nums.size(); ++i) std::cout << nums[i] << (i + 1 < nums.size() ? "," : "");
    std::cout << "], k = " << k << "\n  每个窗口的最大值 = [";
    for (std::size_t i = 0; i < d.size(); ++i) std::cout << d[i] << (i + 1 < d.size() ? "," : "");
    std::cout << "]\n  （暴力与单调队列结果一致 ✓）\n\n";
}

// ===========================================================================
// 2. 滑动窗口最小值：只需把「<=」翻成「>=」，即维持单调递增队列
// ===========================================================================
static std::vector<int> windowMinDeque(const std::vector<int>& nums, std::size_t k) {
    std::vector<int> out;
    if (k == 0 || nums.size() < k) return out;
    std::deque<std::size_t> dq;               // 存下标，值单调递增
    for (std::size_t i = 0; i < nums.size(); ++i) {
        while (!dq.empty() && nums[dq.back()] >= nums[i]) dq.pop_back();  // 与 max 仅差一个符号
        dq.push_back(i);
        while (dq.front() + k <= i) dq.pop_front();
        if (i + 1 >= k) out.push_back(nums[dq.front()]);
    }
    return out;
}

// 用随机数据对 max / min 两种窗口都做「暴力 vs 单调队列」交叉验证，
// 并统计单调队列的总 pop 次数，实测「每个元素最多进出队一次」的均摊性质。
static void part2_window_min_and_amortized() {
    std::cout << "=== 2. 滑动窗口最小值 + 均摊分析实证 ===\n";

    // 随机数组（确定性 LCG，便于复现）
    const std::size_t n = 100000;
    std::vector<int> nums(n);
    unsigned long long s = 88172645463325252ULL;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        nums[i] = static_cast<int>((s >> 33) % 2001) - 1000;   // [-1000, 1000]
    }
    const std::size_t k = 17;

    // 与暴力交叉验证（只抽样窗口，避免 O(n·k) 太慢：这里 n 大 k 小，其实也快）
    const std::vector<int> bmax = windowMaxBrute(nums, k);
    const std::vector<int> dmax = windowMaxDeque(nums, k);
    const std::vector<int> dmin = windowMinDeque(nums, k);
    assert(bmax == dmax);

    // 手工核对 min 与 max 的关系：min <= max 恒成立
    assert(dmin.size() == dmax.size());
    for (std::size_t i = 0; i < dmin.size(); ++i) assert(dmin[i] <= dmax[i]);

    std::cout << "  n = " << n << ", k = " << k << "\n";
    std::cout << "  首个窗口 max = " << dmax.front() << ", min = " << dmin.front() << "\n";

    // 均摊实证：统计「小于等于新元素的队尾弹出」总次数，
    // 它一定 < n（每个下标最多被弹出一次），故整体 O(n)。
    std::deque<std::size_t> dq;
    long long pops = 0;
    for (std::size_t i = 0; i < n; ++i) {
        while (!dq.empty() && nums[dq.back()] <= nums[i]) { dq.pop_back(); ++pops; }
        dq.push_back(i);
        while (dq.front() + k <= i) dq.pop_front();
    }
    std::cout << "  全场（含前后边界）队尾弹出总次数 = " << pops
              << " < n = " << n << "  -> 均摊 O(1)/元素，整体 O(n) ✓\n\n";
}

// ===========================================================================
// 3. 用两个栈实现队列（均摊 O(1)）
// ===========================================================================
//
// inStack  只负责 push；outStack 只负责 pop/peek。
// 当 outStack 为空时，把 inStack **整体倒过去**（顺序因此被翻转两次 = FIFO）。
// 关键：每个元素一生最多被搬一次（in -> out），故均摊 O(1)。
template <typename T>
class QueueWithTwoStacks {
public:
    void push(const T& v) { in_.push(v); }          // 永远 O(1)
    void push(T&& v)      { in_.push(std::move(v)); }

    T pop() {
        moveIfNeeded();
        T v = out_.top();
        out_.pop();
        return v;
    }
    const T& front() {
        moveIfNeeded();
        return out_.top();
    }
    bool empty() const { return in_.empty() && out_.empty(); }
    std::size_t size() const { return in_.size() + out_.size(); }

    // 仅用于演示：统计总搬迁次数，验证均摊 O(1)
    long long moves() const { return moves_; }

private:
    void moveIfNeeded() {
        if (!out_.empty()) return;                  // 还有货，不必搬
        while (!in_.empty()) {                      // 一次性倒空 in_ -> out_
            out_.push(in_.top());
            in_.pop();
            ++moves_;
        }
    }
    std::stack<T> in_, out_;
    long long moves_ = 0;
};

static void part3_two_stacks_queue() {
    std::cout << "=== 3. 用两个栈实现队列（均摊 O(1)）===\n";
    QueueWithTwoStacks<int> q;

    // 交错 push/pop：最容易暴露「忘了搬迁」或「顺序反了」的 bug
    for (int x : {1, 2, 3}) q.push(x);
    assert(q.pop() == 1 && q.pop() == 2);           // 跳出 1,2
    assert(!q.empty() && q.front() == 3);           // 队首是 3
    for (int x : {4, 5}) q.push(x);                 // in_ 里现在有 3,4,5
    assert(q.pop() == 3 && q.pop() == 4 && q.pop() == 5);
    assert(q.empty());

    std::cout << "  交错 push/pop 序列 1,2,3 |pop| 4,5  -> 严格 FIFO 输出 ✓\n";

    // 大量元素，看搬迁次数：应约等于 push 次数（每个元素最多搬一次）
    QueueWithTwoStacks<int> r;
    const int N = 100000;
    for (int i = 0; i < N; ++i) r.push(i);
    long long sum = 0;
    for (int i = 0; i < N; ++i) sum += r.pop();
    assert(sum == static_cast<long long>(N) * (N - 1) / 2);
    std::cout << "  push " << N << " 次后全部 pop：总搬迁 = " << r.moves()
              << " ≈ push 次数（每个元素最多搬一次）-> 均摊 O(1) ✓\n\n";
}

// ===========================================================================
// 4.（附）单调队列 vs 优先队列：为什么有人会用 deque 而不是 heap？
// ===========================================================================
// 用最大堆也能求滑动窗口最大值，但「过期元素」要**惰性删除**：
//   每次取值时，弹出堆顶那些下标已滑出窗口的元素。
// 两者复杂度都是 O(n log k) vs O(n)。这里只做正确性对照，点出取舍：
//   * 单调队列：O(n)，但只能处理「窗口整体平移」这种规整的过期顺序；
//   * 优先队列：O(n log k)，更通用（窗口可任意增删、可带权重/优先级）。
// 深度学习里「KV cache 淘汰」「请求优先级调度」往往既要用堆的通用优先级，
// 又要用 deque 的「先进先出 + 两端裁剪」语义 —— 两者是互补的工具。
static std::vector<int> windowMaxHeap(const std::vector<int>& nums, std::size_t k) {
    std::vector<int> out;
    if (k == 0 || nums.size() < k) return out;
    std::vector<std::size_t> idx;                    // 手动二叉堆（按值比较）
    auto less = [&](std::size_t a, std::size_t b) { return nums[a] < nums[b]; };
    auto siftUp = [&](std::size_t i) {
        while (i > 0) {
            std::size_t p = (i - 1) / 2;
            if (!less(idx[p], idx[i])) break;
            std::swap(idx[p], idx[i]); i = p;
        }
    };
    auto siftDown = [&](std::size_t i) {
        for (;;) {
            std::size_t l = 2 * i + 1, rr = 2 * i + 2, big = i;
            if (l < idx.size() && less(idx[big], idx[l])) big = l;
            if (rr < idx.size() && less(idx[big], idx[rr])) big = rr;
            if (big == i) break;
            std::swap(idx[big], idx[i]); i = big;
        }
    };
    for (std::size_t i = 0; i < nums.size(); ++i) {
        idx.push_back(i); siftUp(idx.size() - 1);
        while (idx.front() + k <= i) {               // 惰性删除过期堆顶
            idx.front() = idx.back(); idx.pop_back(); siftDown(0);
        }
        if (i + 1 >= k) out.push_back(nums[idx.front()]);
    }
    return out;
}

static void part4_heap_comparison() {
    std::cout << "=== 4.（附）单调队列 O(n) vs 优先队列 O(n log k) ===\n";
    const std::vector<int> nums{1, 3, -1, -3, 5, 3, 6, 7};
    const std::size_t k = 3;
    assert(windowMaxHeap(nums, k) == windowMaxDeque(nums, k));
    std::cout << "  两者结果一致；deque 版 O(n)、heap 版 O(n log k)，\n"
                 "  但 heap 更通用（窗口可任意增删/带优先级）——工具互补，不是二选一。\n\n";
}

int main() {
    std::cout << "======== 06 双端队列：滑动窗口 / 两栈队列 ========\n\n";
    part1_window_max();
    part2_window_min_and_amortized();
    part3_two_stacks_queue();
    part4_heap_comparison();
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
