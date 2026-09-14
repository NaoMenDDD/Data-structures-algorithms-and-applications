// 02_amortized_growth.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 02 章 线性表与数组》
// 主题：扩容因子 2.0 与 1.5 的「全维度」对比
//
//   第 01 章只回答了「总搬移次数」一个问题。本章把扩容因子的取舍算清楚。
//   设扩容因子为 b > 1，容量序列 c_0 = 1, c_{k+1} = b·c_k，最后一次扩容到 c_m ≈ n。
//   于是：
//     * 总搬移 ≈ c_0 + c_1 + ... + c_{m-1} = (c_m - 1)/(b - 1) ≈ n/(b - 1)
//         b = 2.0 → ≈ n；        b = 1.5 → ≈ 2n。      （b 越大越省）
//     * 单次最坏搬移 = c_{m-1} ≈ n / b
//         b = 2.0 → ≈ n/2；      b = 1.5 → ≈ n/1.5 = 0.67n。（同样 b 越大越省）
//     * 最大闲置比 capacity / size ≈ b
//         b = 2.0 → ≈ 2；        b = 1.5 → ≈ 1.5。      （b 越小越省内存）
//
//   ⚠ 一个常见误解：「扩容因子越小，延迟尖峰越低」。实际上恰恰相反 ——
//     单次最坏搬移 ≈ n/b，b 越小它越大。1.5 倍在「总搬移」和「最坏尖峰」两项上
//     都输给 2 倍。真正让工程界偏爱 1.5 倍的是**内存**：
//       (a) 闲置比 ≈ b，1.5 倍更紧凑；
//       (b) 当 b < φ ≈ 1.618 时，新块大小 < 旧块 + 更早的块之和，
//           分配器才有可能**复用**刚刚释放的旧块，显著降低碎片与峰值内存。
//           b ≥ 2 时新块总是 ≥ 2 倍旧块，刚释放的旧块永远「太小」而无法复用。
//
//   所以：2 倍讨「吞吐与低尖峰」，1.5 倍讨「内存与抗碎片」。
//     本文件会把这些结论全部用实测数字钉死（含断言）。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 02_amortized_growth.cpp -o growth && ./growth
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 02_amortized_growth.cpp -o growth_san && ./growth_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <new>       // placement new / ::operator new
#include <utility>   // std::move

// ---------------------------------------------------------------------------
// 会记账的元素类型：每次移动/拷贝都累加计数
// ---------------------------------------------------------------------------
struct Counted {
    int value;

    static std::uint64_t moves;
    static std::uint64_t copies;

    explicit Counted(int v = 0) : value(v) {}
    Counted(const Counted& o) : value(o.value) { ++copies; }
    Counted(Counted&& o) noexcept : value(o.value) { ++moves; o.value = -1; }
};
std::uint64_t Counted::moves  = 0;
std::uint64_t Counted::copies = 0;

// ---------------------------------------------------------------------------
// 记录「扩容档案」的动态数组：除总搬移外，还记录单次最坏搬移与容量轨迹
// ---------------------------------------------------------------------------
template <typename T>
class TrackedArray {
public:
    explicit TrackedArray(double grow)
        : data_(nullptr), size_(0), cap_(0), grow_(grow),
          totalReloc_(0), worstReloc_(0), growEvents_(0) {}

    ~TrackedArray() {
        for (std::size_t i = 0; i < size_; ++i) data_[i].~T();
        ::operator delete(data_);
    }
    TrackedArray(const TrackedArray&)            = delete;
    TrackedArray& operator=(const TrackedArray&) = delete;

    std::size_t   size()       const { return size_; }
    std::size_t   capacity()   const { return cap_; }
    std::uint64_t totalReloc() const { return totalReloc_; }
    std::uint64_t worstReloc() const { return worstReloc_; }  // 单次扩容搬移的最大值
    std::uint64_t growEvents() const { return growEvents_; }  // 扩容发生的次数

    void push_back(const T& x) {
        if (size_ == cap_) grow();
        new (data_ + size_) T(x);
        ++size_;
    }

private:
    void grow() {
        // 新容量至少 +1，保证 grow_ == 1.0 时也能前进
        std::size_t newCap =
            (cap_ == 0) ? 1 : static_cast<std::size_t>(cap_ * grow_);
        if (newCap <= cap_) newCap = cap_ + 1;

        ++growEvents_;
        totalReloc_ += size_;                    // 本次要搬移 size_ 个元素
        if (size_ > worstReloc_) worstReloc_ = size_;

        T* nd = static_cast<T*>(::operator new(newCap * sizeof(T)));
        for (std::size_t i = 0; i < size_; ++i) {
            new (nd + i) T(std::move(data_[i]));
            data_[i].~T();
        }
        ::operator delete(data_);
        data_ = nd;
        cap_  = newCap;
    }

    T*            data_;
    std::size_t   size_;
    std::size_t   cap_;
    double        grow_;
    std::uint64_t totalReloc_;
    std::uint64_t worstReloc_;
    std::uint64_t growEvents_;
};

// ---------------------------------------------------------------------------
// 跑 n 次 push_back，汇总指标
// ---------------------------------------------------------------------------
struct Stats {
    std::size_t   finalCap   = 0;
    std::uint64_t totalReloc = 0;
    std::uint64_t worstReloc = 0;
    std::uint64_t growEvents = 0;
    double        amortized  = 0.0;   // 总搬移 / n
    double        worstOverN = 0.0;   // 单次最坏搬移 / n
    double        maxWaste   = 0.0;   // max(capacity/size)
};

static Stats run(double factor, std::size_t n) {
    TrackedArray<Counted> a(factor);
    double maxWaste = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        a.push_back(Counted(static_cast<int>(i)));
        const double waste = static_cast<double>(a.capacity()) /
                             static_cast<double>(a.size());
        if (waste > maxWaste) maxWaste = waste;
    }
    Stats s;
    s.finalCap   = a.capacity();
    s.totalReloc = a.totalReloc();
    s.worstReloc = a.worstReloc();
    s.growEvents = a.growEvents();
    s.amortized  = static_cast<double>(a.totalReloc()) / static_cast<double>(n);
    s.worstOverN = static_cast<double>(a.worstReloc()) / static_cast<double>(n);
    s.maxWaste   = maxWaste;
    return s;
}

static void experiment_compare() {
    const std::size_t N = 1u << 20;          // 1,048,576 次 push_back
    std::cout << "=== 扩容因子 2.0 vs 1.5：全维度对比（n = " << N << "）===\n\n";

    const Stats s2  = run(2.0, N);
    const Stats s15 = run(1.5, N);

    auto row = [](const char* k, auto v2, auto v15) {
        std::cout << std::left << std::setw(24) << k
                  << std::setw(20) << v2
                  << v15 << "\n";
    };

    std::cout << std::left << std::setw(24) << "指标"
              << std::setw(20) << "因子 2.0" << "因子 1.5\n";
    std::cout << std::string(60, '-') << "\n";
    row("最终容量",           s2.finalCap,   s15.finalCap);
    row("扩容次数",           s2.growEvents, s15.growEvents);
    row("总搬移次数",         s2.totalReloc, s15.totalReloc);
    row("单次最坏搬移",       s2.worstReloc, s15.worstReloc);
    row("摊还代价 搬移/n",    s2.amortized,  s15.amortized);
    row("最坏单次/n",         s2.worstOverN, s15.worstOverN);
    row("最大 capacity/size", s2.maxWaste,   s15.maxWaste);
    std::cout << "\n";

    // ---- 断言：把理论结论钉死 ----
    // (1) 摊还代价都 O(1)。总搬移 ≈ n/(b-1)：2 倍 → n；1.5 倍 → 2n。
    assert(s2.totalReloc  >= N / 2 && s2.totalReloc  < 2 * N);
    assert(s15.totalReloc >= N     && s15.totalReloc < 3 * N);
    assert(s2.amortized  < 2.0);
    assert(s15.amortized < 3.0);

    // (2) 单次最坏搬移 ≈ n/b。故 2 倍(≈n/2) 比 1.5 倍(≈n/1.5) 更小。
    assert(s2.worstReloc  >= N / 4 && s2.worstReloc  <= N / 2 + 1);
    assert(s15.worstReloc >= N / 3 && s15.worstReloc <= N);
    assert(s2.worstReloc < s15.worstReloc);

    // (3) 最大闲置比 ≈ b。1.5 倍内存更紧凑。
    assert(s2.maxWaste  > 1.9 && s2.maxWaste  <= 2.0 + 1e-9);
    assert(s15.maxWaste > 1.4 && s15.maxWaste <= 1.5 + 1e-9);
    assert(s15.maxWaste < s2.maxWaste);

    std::cout << "结论（实测与理论一致）：\n"
                 "  * 总搬移 ≈ n/(b-1)      → 2 倍约 n，1.5 倍约 2n。2 倍更省。\n"
                 "  * 单次最坏搬移 ≈ n/b    → 2 倍约 n/2，1.5 倍约 0.67n。2 倍更低。\n"
                 "  * 最大闲置比 ≈ b        → 2 倍约 2，1.5 倍约 1.5。1.5 倍更省内存。\n"
                 "  故：2 倍讨「吞吐/低尖峰」，1.5 倍讨「内存/抗碎片」（b < φ≈1.618 可复用旧块）。\n\n";
}

// ---------------------------------------------------------------------------
// 附：直接打印每次扩容的搬移量，直观看「尖峰形状」
// ---------------------------------------------------------------------------
static void experiment_spike_shape() {
    std::cout << "=== 每次扩容的搬移量序列（n = 2^20，只看发生扩容的那几次）===\n\n";
    const std::size_t N = 1u << 20;

    for (double factor : {2.0, 1.5}) {
        TrackedArray<Counted> a(factor);
        std::uint64_t last = 0;
        int printed = 0;
        std::cout << "因子 " << factor << "： ";
        for (std::size_t i = 0; i < N; ++i) {
            a.push_back(Counted(static_cast<int>(i)));
            if (a.totalReloc() != last) {
                const std::uint64_t delta = a.totalReloc() - last;
                if (printed < 14) { std::cout << delta << " "; ++printed; }
                last = a.totalReloc();
            }
        }
        std::cout << "...  共 " << a.growEvents() << " 次扩容\n";
    }
    std::cout << "\n（搬移量 = 扩容当时的 size。2 倍时它翻倍增长、更稀但更陡；\n"
                 " 1.5 倍时增长更缓 —— 但注意最后一个尖峰反而更大，因为 n/1.5 > n/2。）\n\n";
}

int main() {
    experiment_compare();
    experiment_spike_shape();
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
