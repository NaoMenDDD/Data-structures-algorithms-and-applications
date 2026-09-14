// 01_amortized_growth.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 01 章 算法分析基础》
// 主题：动态数组扩容的摊还分析（聚合法 / 记账法 / 势能法）
//
//   实验 1：扩容因子对「总搬移次数」的影响
//           - 因子 2.0   → 总搬移 ≈ n      （摊还 O(1)，常数最小）
//           - 因子 1.5   → 总搬移 ≈ 2n     （摊还 O(1)，常数更大）
//           - 因子 1.0   → 总搬移 ≈ n²/2   （摊还 Θ(n)，灾难）
//   实验 2：用势函数 Φ = 2*size − capacity 逐次计算摊还代价 ĉ_i，
//           并断言「倍增时最大摊还代价 == 3」
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 01_amortized_growth.cpp -o amortized && ./amortized
//
// 关键点：本程序从不调用 delete[] 去逐个析构 T，而是用 placement new 手工管理
// 每个槽位的生命周期——这样才能精确地把「搬移」计入统计，而不是被 vector
// 内部的批量优化掩盖掉。
// ---------------------------------------------------------------------------

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <new>        // placement new / ::operator new
#include <utility>    // std::move

// ---------------------------------------------------------------------------
// 一个「会记账」的元素类型：统计自己被移动 / 拷贝了多少次
// ---------------------------------------------------------------------------
struct Counted {
    int value;

    static std::uint64_t moves;    // 移动构造的次数
    static std::uint64_t copies;   // 拷贝构造的次数

    explicit Counted(int v = 0) : value(v) {}

    Counted(const Counted& o) : value(o.value) { ++copies; }
    Counted(Counted&& o) noexcept : value(o.value) { ++moves; o.value = -1; }

    Counted& operator=(const Counted& o) { value = o.value; ++copies; return *this; }
    Counted& operator=(Counted&& o) noexcept { value = o.value; ++moves; o.value = -1; return *this; }
};
std::uint64_t Counted::moves = 0;
std::uint64_t Counted::copies = 0;

// ---------------------------------------------------------------------------
// 可以指定扩容因子的动态数组（教学版，只保留 push_back）
// ---------------------------------------------------------------------------
template <typename T>
class DynArray {
public:
    explicit DynArray(double grow)
        : data_(nullptr), size_(0), cap_(0), grow_(grow), relocations_(0) {}

    ~DynArray() {
        destroyElements();
        ::operator delete(data_);
    }

    DynArray(const DynArray&) = delete;
    DynArray& operator=(const DynArray&) = delete;

    std::size_t size() const { return size_; }
    std::size_t capacity() const { return cap_; }
    std::uint64_t relocations() const { return relocations_; }

    void push_back(const T& x) {
        if (size_ == cap_) grow();
        new (data_ + size_) T(x);      // 在未初始化的槽位上构造
        ++size_;
    }

private:
    void destroyElements() {
        for (std::size_t i = 0; i < size_; ++i) data_[i].~T();
        size_ = 0;
    }

    void grow() {
        // 计算新容量：至少 +1，保证 grow_ == 1.0 时也能前进
        std::size_t newCap = (cap_ == 0) ? 1 : static_cast<std::size_t>(cap_ * grow_);
        if (newCap <= cap_) newCap = cap_ + 1;

        T* nd = static_cast<T*>(::operator new(newCap * sizeof(T)));
        for (std::size_t i = 0; i < size_; ++i) {
            new (nd + i) T(std::move(data_[i]));   // 搬移（移动构造）
            data_[i].~T();
            ++relocations_;
        }
        ::operator delete(data_);
        data_      = nd;
        cap_       = newCap;
    }

    T*            data_;
    std::size_t   size_;
    std::size_t   cap_;
    double        grow_;
    std::uint64_t relocations_;
};

// ---------------------------------------------------------------------------
// 实验 1：扩容因子 → 总搬移次数
// ---------------------------------------------------------------------------
static void experiment_growth_factor() {
    const std::size_t N = 1u << 20;   // 1,048,576 次 push_back

    std::cout << "=== 实验 1：扩容因子对总搬移次数的影响（n = " << N << "）===\n\n";
    std::cout << "扩容因子\t最终容量\t总搬移次数\t搬移/n\n";
    std::cout << "--------------------------------------------------------------\n";

    for (double b : {2.0, 1.5}) {
        DynArray<Counted> a(b);
        for (std::size_t i = 0; i < N; ++i) a.push_back(Counted(static_cast<int>(i)));

        const double ratio = static_cast<double>(a.relocations()) / static_cast<double>(N);
        std::cout << b << "\t\t" << a.capacity() << "\t\t" << a.relocations()
                  << "\t" << ratio << "\n";

        if (b == 2.0) {
            // 倍增：总搬移 = 1+2+4+...+2^(k-1) = n-1 左右
            assert(a.relocations() >= N / 2);
            assert(a.relocations() <  2 * N);
        } else {
            // 1.5 倍：仍是 O(n)，但常数更大
            assert(a.relocations() >= N);
            assert(a.relocations() <  4 * N);
        }
    }

    // 因子 1.0（每次只加 1）会让总搬移变成 n(n-1)/2，只能用很小的 n 演示
    {
        const std::size_t M = 4096;
        DynArray<Counted> a(1.0);
        for (std::size_t i = 0; i < M; ++i) a.push_back(Counted(static_cast<int>(i)));
        const std::uint64_t expected = static_cast<std::uint64_t>(M) * (M - 1) / 2;
        std::cout << "1.0\t\t" << a.capacity() << "\t\t" << a.relocations()
                  << "\t"
                  << static_cast<double>(a.relocations()) / static_cast<double>(M) << "\n";
        std::cout << "（因子 1.0 的理论总搬移 = n(n-1)/2 = " << expected
                  << "，实测一致 → Θ(n²)）\n";
        assert(a.relocations() == expected);
    }

    std::cout << "\n结论：因子 2.0 时总搬移 ≈ n（摊还 O(1)）；因子 1.5 时 ≈ 2n（常数翻倍）；\n";
    std::cout << "      因子 1.0 时退化为 Θ(n²)。这就是「必须倍增」的原因。\n\n";
}

// ---------------------------------------------------------------------------
// 实验 2：势函数法 —— 逐次计算摊还代价
// ---------------------------------------------------------------------------
static void experiment_potential() {
    std::cout << "=== 实验 2：势函数 Φ = 2*size − capacity 下的摊还代价 ===\n\n";
    std::cout << "i\tsize\tcap\t实际代价 c_i\tΦ_i\t摊还代价 ĉ_i\n";
    std::cout << "--------------------------------------------------------------\n";

    const int n = 17;
    std::size_t size = 0, cap = 0;
    long double phiPrev = 0.0L;
    long double maxAmortized = 0.0L;

    for (int i = 1; i <= n; ++i) {
        double c;                              // 本次操作的实际代价
        if (size == cap) {                     // 先判断是否需要扩容
            c   = 1.0 + static_cast<double>(size);   // 插入 1 次 + 搬移 size 个
            cap = (cap == 0) ? 1 : cap * 2;
        } else {
            c = 1.0;
        }
        ++size;

        const long double phi       = 2.0L * static_cast<long double>(size) - static_cast<long double>(cap);
        const long double amortized = static_cast<long double>(c) + phi - phiPrev;
        phiPrev = phi;
        if (amortized > maxAmortized) maxAmortized = amortized;

        std::cout << i << '\t' << size << '\t' << cap << '\t' << c << "\t\t"
                  << static_cast<double>(phi) << '\t' << static_cast<double>(amortized) << '\n';
    }

    std::cout << "\n最大摊还代价 = " << static_cast<double>(maxAmortized)
              << "  （理论值：3）\n";
    assert(maxAmortized <= 3.0L + 1e-9L);
    assert(maxAmortized >= 3.0L - 1e-9L);
    std::cout << "断言通过：倍增策略下，单次 push_back 的摊还代价恒 ≤ 3 = O(1)。\n\n";
}

int main() {
    experiment_growth_factor();
    experiment_potential();

    std::cout << "所有实验与断言通过 ✔\n";
    return 0;
}
