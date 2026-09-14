// 00_template_class.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 0 章 预备：C++ 与开发环境》
// 主题：函数模板、类模板、运算符重载、const 正确性、迭代器与 range-based for
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 00_template_class.cpp -o tmpl && ./tmpl
//
// 目标：从零实现一个「最小可用」的动态数组容器 MiniVector<T>，
//       它模仿 std::vector 的核心接口：push_back / size / operator[] /
//       begin / end / 迭代器 / 范围 for。写完之后再回头用 std::vector，
//       你会真正理解 STL 为什么长这样 —— 因为 STL 全靠模板。
// ---------------------------------------------------------------------------

#include <algorithm>   // std::copy, std::max, std::equal
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>    // std::iterator_traits, std::random_access_iterator_tag
#include <stdexcept>   // std::out_of_range
#include <string>
#include <type_traits>
#include <utility>     // std::move, std::forward
#include <vector>

// ===========================================================================
// 1. 函数模板（function template）：同一段逻辑适配多种类型
// ===========================================================================
//
// 对比：如果用 void* 或运行期多态（虚函数）实现「通用最大值」，
// 要么丢失类型安全，要么有虚函数调用开销。模板在编译期为每个 T
// 生成一份特化代码，既类型安全又零开销 —— 这叫「编译期多态」。

template <typename T>
T maxOf(const T& a, const T& b) {
    return (a < b) ? b : a;
}

// 多个模板参数 + 返回类型推导（C++14 起 auto 返回值）
template <typename A, typename B>
auto addUp(const A& a, const B& b) -> decltype(a + b) {
    return a + b;
}

// 编译期计算：模板 + constexpr 是「元编程」的入口
template <int N>
struct Factorial {
    static constexpr long long value = N * Factorial<N - 1>::value;
};
template <>
struct Factorial<0> {              // 显式特化（full specialization）：递归基
    static constexpr long long value = 1;
};

// 变参模板（variadic template）：C++11 起支持任意个参数
// 这是 std::make_shared / std::thread 等接口的底层机制
template <typename T>
void printAll(const T& value) {
    std::cout << value;
}
template <typename T, typename... Rest>
void printAll(const T& first, const Rest&... rest) {
    std::cout << first << ", ";
    printAll(rest...);             // 递归展开参数包
}

// ===========================================================================
// 2. 类模板 + 运算符重载 + 迭代器：MiniVector<T>
// ===========================================================================

template <typename T>
class MiniVector {
public:
    // ---- 嵌套类型别名：让容器能被泛型算法识别 ----
    using value_type      = T;
    using size_type       = std::size_t;
    using reference        = T&;
    using const_reference  = const T&;
    using pointer          = T*;
    using const_pointer    = const T*;

    // ---- 迭代器：对「裸指针」的一层薄封装 ----
    // std::vector 在多数实现里迭代器就是指针，但我们显式写出来，
    // 便于理解「迭代器 = 指针的泛化」这一概念。
    template <typename Ptr, typename Ref>
    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag; // 随机访问
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = Ptr;
        using reference          = Ref;

        Iterator() : p_(nullptr) {}
        explicit Iterator(Ptr p) : p_(p) {}

        Ref operator*() const { return *p_; }
        Ptr operator->() const { return p_; }
        Iterator& operator++() { ++p_; return *this; }        // 前置 ++
        Iterator operator++(int) { Iterator t(*this); ++p_; return t; } // 后置 ++
        Iterator& operator--() { --p_; return *this; }
        Iterator operator+(difference_type n) const { return Iterator(p_ + n); }
        Iterator operator-(difference_type n) const { return Iterator(p_ - n); }
        difference_type operator-(const Iterator& o) const { return p_ - o.p_; }
        bool operator==(const Iterator& o) const { return p_ == o.p_; }
        bool operator!=(const Iterator& o) const { return p_ != o.p_; }
        bool operator<(const Iterator& o) const { return p_ < o.p_; }
    private:
        Ptr p_;
    };

    using iterator       = Iterator<pointer, reference>;
    using const_iterator = Iterator<const_pointer, const_reference>;

    // ---- 构造函数：默认 / 带容量 / 初始化列表 ----
    MiniVector() = default;

    explicit MiniVector(size_type n) : size_(n), cap_(n), data_(alloc(n)) {
        for (size_type i = 0; i < n; ++i) new (data_ + i) T();  // 定位 new：值初始化
    }

    MiniVector(std::initializer_list<T> init)
        : size_(init.size()), cap_(init.size()), data_(alloc(init.size())) {
        size_type i = 0;
        for (const T& v : init) new (data_ + i++) T(v);
    }

    // ---- 拷贝构造：深拷贝 ----
    MiniVector(const MiniVector& o)
        : size_(o.size_), cap_(o.size_), data_(alloc(o.size_)) {
        for (size_type i = 0; i < size_; ++i) new (data_ + i) T(o.data_[i]);
    }

    // ---- 移动构造：偷指针，noexcept 让 vector<MiniVector> 也能高效扩容 ----
    MiniVector(MiniVector&& o) noexcept
        : size_(o.size_), cap_(o.cap_), data_(o.data_) {
        o.size_ = o.cap_ = 0;
        o.data_ = nullptr;
    }

    // ---- 拷贝赋值（copy-and-swap，强异常安全）----
    MiniVector& operator=(const MiniVector& o) {
        if (this != &o) { MiniVector tmp(o); swap(tmp); }
        return *this;
    }
    // ---- 移动赋值 ----
    MiniVector& operator=(MiniVector&& o) noexcept {
        if (this != &o) {
            destroyAll();
            size_ = o.size_; cap_ = o.cap_; data_ = o.data_;
            o.size_ = o.cap_ = 0; o.data_ = nullptr;
        }
        return *this;
    }

    ~MiniVector() { destroyAll(); }

    void swap(MiniVector& o) noexcept {
        std::swap(size_, o.size_);
        std::swap(cap_, o.cap_);
        std::swap(data_, o.data_);
    }

    // ---- 元素访问 ----
    // 两个重载：非 const 版本返回可写引用，const 版本返回只读引用。
    // 这就是「const 正确性」——const 对象只能调用 const 成员函数。
    reference       operator[](size_type i)       { return data_[i]; }
    const_reference operator[](size_type i) const { return data_[i]; }

    reference at(size_type i) {
        if (i >= size_) throw std::out_of_range("MiniVector::at 越界");
        return data_[i];
    }
    const_reference at(size_type i) const {
        if (i >= size_) throw std::out_of_range("MiniVector::at 越界");
        return data_[i];
    }

    reference       front()       { return data_[0]; }
    const_reference front() const { return data_[0]; }
    reference       back()        { return data_[size_ - 1]; }
    const_reference back()  const { return data_[size_ - 1]; }

    // ---- 容量与大小 ----
    size_type size()     const { return size_; }
    size_type capacity() const { return cap_; }
    bool      empty()    const { return size_ == 0; }

    // ---- 迭代器接口：这是 range-based for 能工作的关键 ----
    // 编译器把 `for (auto& x : v)` 展开成 `for (auto it = v.begin(); it != v.end(); ++it)`
    iterator       begin()        { return iterator(data_); }
    iterator       end()          { return iterator(data_ + size_); }
    const_iterator begin()  const { return const_iterator(data_); }
    const_iterator end()    const { return const_iterator(data_ + size_); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend()   const { return end(); }

    // ---- 尾部追加（第 1 章将证明其摊还复杂度为 O(1)）----
    void push_back(const T& value) {
        ensureCapacity(size_ + 1);
        new (data_ + size_) T(value);   // 定位 new 在已分配内存上构造
        ++size_;
    }
    void push_back(T&& value) {
        ensureCapacity(size_ + 1);
        new (data_ + size_) T(std::move(value));
        ++size_;
    }

    // 就地构造：避免临时对象，std::vector::emplace_back 的同款机制
    template <typename... Args>
    reference emplace_back(Args&&... args) {
        ensureCapacity(size_ + 1);
        new (data_ + size_) T(std::forward<Args>(args)...);
        return data_[size_++];
    }

    void pop_back() {
        assert(size_ > 0);
        --size_;
        data_[size_].~T();
    }

    void reserve(size_type n) {
        if (n > cap_) reallocate(n);
    }
    void clear() {
        destroyAll();
        size_ = 0;
    }

private:
    static pointer alloc(size_type n) {
        return n ? static_cast<pointer>(::operator new(n * sizeof(T))) : nullptr;
    }
    // 扩容策略：容量翻倍。选 2 倍还是 1.5 倍有讲究（见第 1 章摊还分析）。
    void ensureCapacity(size_type need) {
        if (need <= cap_) return;
        size_type newCap = (cap_ == 0) ? 1 : cap_ * 2;
        while (newCap < need) newCap *= 2;
        reallocate(newCap);
    }
    void reallocate(size_type newCap) {
        pointer fresh = alloc(newCap);
        // 用移动构造搬移元素：对 std::string 这类对象比拷贝快得多
        for (size_type i = 0; i < size_; ++i) {
            new (fresh + i) T(std::move(data_[i]));
            data_[i].~T();
        }
        if (data_) ::operator delete(data_);
        data_ = fresh;
        cap_ = newCap;
    }
    void destroyAll() {
        for (size_type i = 0; i < size_; ++i) data_[i].~T();
        if (data_) ::operator delete(data_);
        data_ = nullptr;
        size_ = cap_ = 0;
    }

    size_type size_ = 0;
    size_type cap_ = 0;
    pointer   data_ = nullptr;
};

// 非成员 swap：std::swap 会对自定义类型查找 ADL 找到它
template <typename T>
void swap(MiniVector<T>& a, MiniVector<T>& b) noexcept { a.swap(b); }

// 运算符重载：相等比较
template <typename T>
bool operator==(const MiniVector<T>& a, const MiniVector<T>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!(a[i] == b[i])) return false;
    return true;
}

// 泛型函数可以同时吃 std::vector 和 MiniVector<T>，因为它们接口相同
template <typename Container>
typename Container::value_type sumOf(const Container& c) {
    typename Container::value_type s{};
    for (const auto& x : c) s += x;   // range-based for 只依赖 begin/end
    return s;
}

// ===========================================================================
// 3. 编译期多态 vs 运行期多态（对比演示）
// ===========================================================================

// 运行期多态：虚函数 + 基类指针，有虚表（vtable）间接调用开销
struct ShapeRTP {
    virtual double area() const = 0;
    virtual ~ShapeRTP() = default;
};
struct CircleRTP : ShapeRTP {
    double r;
    explicit CircleRTP(double rr) : r(rr) {}
    double area() const override { return 3.141592653589793 * r * r; }
};
struct SquareRTP : ShapeRTP {
    double s;
    explicit SquareRTP(double ss) : s(ss) {}
    double area() const override { return s * s; }
};

// 编译期多态：模板 + concept（C++20）的雏形这里用普通模板表达
struct CircleCTP { double r; };
struct SquareCTP { double s; };
double area(const CircleCTP& c) { return 3.141592653589793 * c.r * c.r; }
double area(const SquareCTP& s) { return s.s * s.s; }

// ===========================================================================
// 测试与自测
// ===========================================================================
static void test_function_templates() {
    std::cout << "\n=== 函数模板 ===\n";
    static_assert(Factorial<5>::value == 120, "编译期阶乘错误");
    static_assert(Factorial<0>::value == 1, "0! 应为 1");
    std::cout << "maxOf(3, 7) = " << maxOf(3, 7) << "\n";
    std::cout << "maxOf(2.5, 1.5) = " << maxOf(2.5, 1.5) << "\n";
    std::cout << "maxOf(std::string(\"a\"), std::string(\"b\")) = "
              << maxOf(std::string("a"), std::string("b")) << "\n";
    std::cout << "addUp(1, 2.5) = " << addUp(1, 2.5) << "\n";
    std::cout << "Factorial<5> = " << Factorial<5>::value << "\n";
    std::cout << "printAll: ";
    printAll(1, 2.5, "three", std::string("four"));
    std::cout << "\n";
    // 编译期分支：if constexpr 让不同 T 走不同代码路径
    using T = int;
    if constexpr (std::is_integral_v<T>) {
        std::cout << "T 是整型（编译期判定）\n";
    }
}

static void test_minivector() {
    std::cout << "\n=== MiniVector<T> 类模板 ===\n";

    MiniVector<int> v;
    for (int i = 0; i < 10; ++i) v.push_back(i * i);
    std::cout << "size = " << v.size() << ", capacity = " << v.capacity() << "\n";
    assert(v.size() == 10);
    assert(v[3] == 9);
    assert(v.back() == 81);

    // range-based for（可写）
    for (auto& x : v) x += 1;
    assert(v[0] == 1);

    // range-based for（只读），对 const 对象调用 const 版 begin/end
    const MiniVector<int>& cv = v;
    long long total = 0;
    for (const auto& x : cv) total += x;
    std::cout << "元素之和 = " << total << "\n";

    // 迭代器显式使用
    auto it = v.begin();
    std::cout << "第一个元素 = " << *it << ", 第二个 = " << *(it + 1) << "\n";
    std::cout << "迭代器距离 = " << (v.end() - v.begin()) << "\n";

    // at() 越界抛异常
    bool threw = false;
    try { (void)v.at(1000); }
    catch (const std::out_of_range& e) { threw = true; std::cout << "捕获异常: " << e.what() << "\n"; }
    assert(threw);

    // 拷贝 / 移动 / 初始值列表
    MiniVector<int> copy = v;
    assert(copy == v);
    MiniVector<int> moved = std::move(copy);
    assert(moved.size() == 10);
    assert(copy.size() == 0);

    MiniVector<std::string> names{"alpha", "beta", "gamma"};
    names.emplace_back("delta");
    assert(names.size() == 4);
    std::cout << "names: ";
    for (const auto& n : names) std::cout << n << " ";
    std::cout << "\n";

    // pop_back / clear
    names.pop_back();
    assert(names.size() == 3);
    names.clear();
    assert(names.empty());

    // 泛型函数同时适用 std::vector 与 MiniVector
    std::vector<int> sv{1, 2, 3, 4};
    std::cout << "sumOf(std::vector) = " << sumOf(sv) << "\n";
    std::cout << "sumOf(MiniVector)  = " << sumOf(v) << "\n";
    assert(sumOf(sv) == 10);

    // MiniVector<MiniVector<int>>：验证移动 noexcept 的实际价值
    MiniVector<MiniVector<int>> nested;
    for (int i = 0; i < 5; ++i) {
        MiniVector<int> row;
        for (int j = 0; j <= i; ++j) row.push_back(j);
        nested.push_back(std::move(row));
    }
    assert(nested.size() == 5);
    assert(nested[4].size() == 5);
    std::cout << "嵌套容器尾行长度 = " << nested[4].size() << "\n";
}

static void test_polymorphism() {
    std::cout << "\n=== 编译期多态 vs 运行期多态 ===\n";
    CircleRTP c{1.0};
    SquareRTP s{2.0};
    ShapeRTP* shapes[] = {&c, &s};
    for (auto* sp : shapes) std::cout << "RTP area = " << sp->area() << "\n";

    CircleCTP cc{1.0};
    SquareCTP ss{2.0};
    std::cout << "CTP area = " << area(cc) << "\n";
    std::cout << "CTP area = " << area(ss) << "\n";
    // 编译器可直接内联 area()，运行期版本需要一次虚表间接寻址
}

int main() {
    test_function_templates();
    test_minivector();
    test_polymorphism();
    std::cout << "\n全部断言通过，程序正常结束。\n";
    return 0;
}
