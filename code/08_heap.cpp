// 08_heap.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 08 章 优先级队列与堆》
// 主题：二叉堆（binary heap）—— 优先级队列的标准实现
//
//   1. 二叉堆：一棵**完全二叉树**，用数组紧凑存储（无指针），
//      满足「堆序性质」：每个结点都不低于（或不高于）它的孩子。
//      * 下标 i 的左孩子 2i+1、右孩子 2i+2、父 ⌊(i-1)/2⌋（0 基）。
//      * push：放到末尾，然后 sift-up（上浮）  O(log n)
//      * pop ：取出堆顶，把末尾搬到堆顶，然后 sift-down（下沉） O(log n)
//      * top ：O(1)
//
//   2. ★ 建堆 heapify 为什么是 O(n) 而不是 O(n log n)？
//      自底向上从最后一个内部结点开始 sift-down。
//      高度为 h 的结点约 n/2^{h+1} 个，下沉代价 O(h)，
//      总和 Σ n·h/2^{h+1} = n·Σ h/2^{h+1} 收敛到 ≤ 2n —— 所以是 O(n)。
//      本文件用**比较次数实测**把这个常数钉出来（≈ 2n）。
//
//   3. 应用：用堆做「多路归并」（合并 k 个有序数组），呼应 MapReduce / 归并排序。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 08_heap.cpp -o 08_heap && ./08_heap
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 08_heap.cpp -o hp_san && ./hp_san
// ---------------------------------------------------------------------------

#include <algorithm>    // std::sort
#include <cassert>
#include <cmath>        // std::log2
#include <cstddef>
#include <cstdio>       // std::printf
#include <functional>   // std::less, std::greater
#include <iostream>
#include <queue>
#include <random>
#include <utility>
#include <vector>

// ===========================================================================
// 1. 泛型二叉堆
// ===========================================================================
//
// 比较器语义与 std::priority_queue 一致：Compare = std::less<T> 表示
// 「小的优先级低」=> 堆顶是**最大**元素（大顶堆 / max-heap）。
// 想得到小顶堆，传 std::greater<T>。
template <typename T, typename Compare = std::less<T>>
class BinaryHeap {
public:
    BinaryHeap() = default;
    explicit BinaryHeap(Compare comp) : comp_(std::move(comp)) {}

    // 用一批数据 O(n) 建堆（不是逐个 push 的 O(n log n)！）
    explicit BinaryHeap(std::vector<T> data, Compare comp = Compare())
        : data_(std::move(data)), comp_(std::move(comp)) {
        heapify();
    }

    bool empty() const { return data_.empty(); }
    std::size_t size() const { return data_.size(); }
    const T& top() const { return data_.front(); }        // 堆顶，O(1)

    void push(const T& v) { data_.push_back(v); siftUp(data_.size() - 1); }
    void push(T&& v)      { data_.push_back(std::move(v)); siftUp(data_.size() - 1); }

    T pop() {
        T out = std::move(data_.front());
        data_.front() = std::move(data_.back());
        data_.pop_back();
        if (!data_.empty()) siftDown(0);
        return out;
    }

    // 校验堆序性质：每个结点都不低于它的孩子
    bool isValidHeap() const {
        const std::size_t n = data_.size();
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t l = 2 * i + 1, r = 2 * i + 2;
            if (l < n && comp_(data_[i], data_[l])) return false;   // 父 < 子 违反
            if (r < n && comp_(data_[i], data_[r])) return false;
        }
        return true;
    }

    const std::vector<T>& data() const { return data_; }

private:
    // 上浮：把下标 i 的元素往上挪，直到不再比父结点优先级低
    void siftUp(std::size_t i) {
        while (i > 0) {
            const std::size_t p = (i - 1) / 2;
            if (!comp_(data_[p], data_[i])) break;   // 父 >= 子，堆序已满足
            std::swap(data_[p], data_[i]);
            i = p;
        }
    }

    // 下沉：把下标 i 的元素往下挪，直到不低于两个孩子
    void siftDown(std::size_t i) {
        const std::size_t n = data_.size();
        for (;;) {
            const std::size_t l = 2 * i + 1, r = 2 * i + 2;
            std::size_t best = i;                    // 找优先级最高的（comp 意义下最大）孩子
            if (l < n && comp_(data_[best], data_[l])) best = l;
            if (r < n && comp_(data_[best], data_[r])) best = r;
            if (best == i) break;
            std::swap(data_[best], data_[i]);
            i = best;
        }
    }

    // 自底向上建堆：从最后一个内部结点倒着 sift-down
    void heapify() {
        if (data_.size() < 2) return;
        for (std::size_t i = data_.size() / 2; i-- > 0; ) siftDown(i);
    }

    std::vector<T> data_;
    Compare comp_{};
};

// ===========================================================================
// 2. 实测：O(n) 建堆 vs O(n log n) 逐个 push
// ===========================================================================
// 直接数「比较次数」是最干净的证据（操作次数精确、不受机器影响）。

// 逐个 push 建堆：每次上浮代价 O(log i)，总计 O(n log n)
static long long buildByPush(std::vector<int> v) {
    long long cmp = 0;
    auto less = [&](int a, int b) { ++cmp; return a < b; };
    std::vector<int> h;
    h.reserve(v.size());
    for (int x : v) {
        h.push_back(x);
        std::size_t i = h.size() - 1;
        while (i > 0) {                              // sift-up
            std::size_t p = (i - 1) / 2;
            if (!less(h[p], h[i])) break;
            std::swap(h[p], h[i]);
            i = p;
        }
    }
    return cmp;
}

// 自底向上 heapify：总计 O(n)
static long long buildByHeapify(std::vector<int> v) {
    long long cmp = 0;
    auto less = [&](int a, int b) { ++cmp; return a < b; };
    const std::size_t n = v.size();
    auto siftDown = [&](std::size_t i) {
        for (;;) {
            std::size_t l = 2 * i + 1, r = 2 * i + 2, best = i;
            if (l < n && less(v[best], v[l])) best = l;
            if (r < n && less(v[best], v[r])) best = r;
            if (best == i) break;
            std::swap(v[best], v[i]);
            i = best;
        }
    };
    if (n >= 2) for (std::size_t i = n / 2; i-- > 0; ) siftDown(i);
    return cmp;
}

static void demo_build_heap() {
    std::cout << "=== 2. 建堆：O(n) vs O(n log n) —— 比较次数实测 ===\n";
    std::cout << "        n  push(升序)  push(随机)   heapify  heapify/n  push升序/nlog2n\n";
    for (std::size_t n : {4096u, 65536u, 1048576u, 16777216u}) {
        std::vector<int> asc(n), rnd(n);
        std::mt19937 rng(20260913);
        for (std::size_t i = 0; i < n; ++i) { asc[i] = static_cast<int>(i); rnd[i] = static_cast<int>(rng()); }
        const long long cAsc  = buildByPush(asc);     // 最坏输入：每个新元素都顶到根
        const long long cRnd  = buildByPush(rnd);     // 平均输入：上浮距离短
        const long long cHeap = buildByHeapify(rnd);  // 自底向上
        const double nlog2n = double(n) * std::log2(double(n));
        std::printf("%9zu %10lld %11lld %9lld %10.3f %13.3f\n",
                    n, cAsc, cRnd, cHeap,
                    double(cHeap) / double(n), double(cAsc) / nlog2n);
    }
    std::cout << "  -> heapify 的比较次数 ≈ 2n（与 n 成正比 = O(n)）；\n"
                 "     逐次 push 在**最坏输入（升序）**下 ≈ n·log2(n)，最后一列稳定在 1 附近\n"
                 "     证实了 O(n log n) 的上界；而随机输入下上浮距离很短，反而接近 O(n)。\n"
                 "     结论：heapify 把**最坏情况**从 O(n log n) 压到 O(n)——这才是它的价值。\n\n";
}

// ===========================================================================
// 3. 应用：合并 k 个有序数组（多路归并）
// ===========================================================================
// 用小顶堆维护「k 个数组各自当前最小元素」，每次弹出全局最小、再从它所属
// 数组补下一个。总复杂度 O(N log k)（N = 元素总数），远优于「两两归并」的
// O(N log k) 但常数更大，也优于「全部排序」的 O(N log N)。
static std::vector<int> mergeKSorted(const std::vector<std::vector<int>>& lists) {
    using Item = std::pair<int, std::pair<std::size_t, std::size_t>>;  // (值, (第几个数组, 下标))
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;  // 小顶堆
    for (std::size_t i = 0; i < lists.size(); ++i)
        if (!lists[i].empty()) pq.push({lists[i][0], {i, 0}});

    std::vector<int> out;
    while (!pq.empty()) {
        auto [val, pos] = pq.top();
        pq.pop();
        out.push_back(val);
        const std::size_t li = pos.first, idx = pos.second;
        if (idx + 1 < lists[li].size()) pq.push({lists[li][idx + 1], {li, idx + 1}});
    }
    return out;
}

static void test_heap_and_merge() {
    std::cout << "=== 1. 堆的基本操作 ===\n";
    BinaryHeap<int> h;                       // 默认 std::less -> 大顶堆
    for (int x : {5, 1, 9, 3, 7, 2, 8, 6, 4, 0}) { h.push(x); assert(h.isValidHeap()); }
    std::cout << "  push 0..9 后堆顶（最大）= " << h.top() << "\n";
    assert(h.top() == 9);
    std::vector<int> popped;
    while (!h.empty()) { popped.push_back(h.pop()); assert(h.isValidHeap()); }
    std::cout << "  依次弹出：";
    for (int x : popped) std::cout << x << ' ';
    std::cout << "\n";
    assert((popped == std::vector<int>{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));

    // 小顶堆
    BinaryHeap<int, std::greater<int>> mn;
    for (int x : {5, 1, 9, 3, 7}) mn.push(x);
    assert(mn.top() == 1 && mn.isValidHeap());
    std::cout << "  小顶堆堆顶（最小）= " << mn.top() << " ✓\n";

    // O(n) 建堆 + 随机验证
    std::mt19937 rng(7);
    std::vector<int> randv(5000);
    for (int& x : randv) x = static_cast<int>(rng() % 100000);
    BinaryHeap<int> hb(randv);
    assert(hb.isValidHeap());
    std::vector<int> sorted = randv;
    std::sort(sorted.begin(), sorted.end(), std::greater<int>());
    for (int x : sorted) assert(hb.pop() == x);
    std::cout << "  5000 个随机数建堆后依次弹出 == 降序排序 ✓\n";

    std::cout << "\n=== 3. 多路归并：合并 k 个有序数组 ===\n";
    std::vector<std::vector<int>> lists = {
        {1, 4, 7, 10}, {2, 5, 8, 11}, {3, 6, 9, 12, 15}, {0, 13, 14}
    };
    const std::vector<int> merged = mergeKSorted(lists);
    std::cout << "  4 个有序数组合并 -> ";
    for (int x : merged) std::cout << x << ' ';
    std::cout << "\n";
    std::vector<int> expect;
    for (const auto& l : lists) expect.insert(expect.end(), l.begin(), l.end());
    std::sort(expect.begin(), expect.end());
    assert(merged == expect);
    std::cout << "  结果与「全部排序」一致 ✓  （每步 O(log k)，总 O(N log k)）\n\n";
}

int main() {
    std::cout << "======== 08 二叉堆：优先级队列 ========\n\n";
    test_heap_and_merge();
    demo_build_heap();
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
