// 02_array_list.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 2 章 线性表与数组》
// 主题：抽象数据类型（ADT）、线性表接口、用「可扩容数组」实现的模板类
//       arrayList<T>，含随机访问迭代器、倍增扩容、越界抛异常、
//       强异常安全（扩容失败不破坏原对象）。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 02_array_list.cpp -o arrlist && ./arrlist
// 消毒器版本（推荐用于数据结构代码）：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 02_array_list.cpp -o arrlist_san && ./arrlist_san
//
// 对照：Sahni《数据结构、算法与应用（C++ 版）》第 2 版 第 5 章
//       "Linear Lists — Array Representation"。
//
// 阅读建议：先读懂接口（ADT），再看实现。接口决定「能做什么」，
//           实现决定「怎么做到、代价多少」。二者分离是本课程的核心思想。
// ---------------------------------------------------------------------------

#include <algorithm>    // std::copy, std::copy_backward, std::move_backward, std::max
#include <cassert>      // assert
#include <cstddef>      // std::size_t, std::ptrdiff_t
#include <initializer_list>
#include <iostream>
#include <iterator>     // std::random_access_iterator_tag
#include <new>          // ::operator new / ::operator delete, 定位 new
#include <ostream>
#include <stdexcept>    // std::out_of_range, std::length_error
#include <string>
#include <type_traits>  // std::move_if_noexcept
#include <utility>      // std::move, std::swap

// ===========================================================================
// 1. 抽象数据类型（Abstract Data Type, ADT）与接口/实现分离
// ===========================================================================
//
// 「线性表（linear list）」的数学定义：n（n >= 0）个元素的有穷序列
//        (e_0, e_1, ..., e_{n-1})
// 每个元素有唯一前驱（首元素除外）与唯一后继（末元素除外）。n = 0 时为空表。
//
// ADT 只规定「操作的名字与语义」，不规定「用什么存」。同一个线性表 ADT，
// 可以用数组实现（本章），也可以用链表实现（第 3 章）。
//
// 我们把数组实现命名为 arrayList<T>，把链表实现命名为 chain<T>；
// 二者接口尽量一致，于是「换实现」对上层代码透明 —— 这就是接口/实现分离的价值。
//
// ---------------------------------------------------------------------------
// 关于「索引」的两套说法（初学者最容易混淆的地方）
// ---------------------------------------------------------------------------
//   * 0-based 索引（index）：C/C++/Python 风格，合法范围 [0, n-1]。
//   * 第 k 个元素（1-based 序数, ordinal）：k 从 1 开始，第 1 个 = 索引 0，
//     第 k 个 = 索引 k-1。教材讲「在位置 k 插入」时若 k 从 1 数，
//     对应到本实现的 insert(k-1, x)；在索引 k 处插入，即新元素成为第 k+1 个元素。
//
// 本实现一律采用 0-based 索引，并在接口注释里写清语义，避免歧义。
//
// ---------------------------------------------------------------------------
// 线性表 ADT 接口一览（本文件实现）
// ---------------------------------------------------------------------------
//   size()            当前元素个数 n
//   empty()           n == 0 ?
//   capacity()        已分配的槽位数（>= n），实现细节，ADT 层面可不暴露
//   get(i)            读取索引 i 的元素（越界抛 std::out_of_range）
//   set(i, x)         把索引 i 的元素改为 x
//   indexOf(x)        返回第一个等于 x 的元素的索引；不存在返回 -1
//   insert(i, x)      在索引 i 处插入 x，原 i..n-1 的元素整体后移一格
//   erase(i)          删除索引 i 的元素，原 i+1..n-1 的元素整体前移一格
//   output(os)        按序打印
//   clear()           清空
// ===========================================================================

template <typename T>
class arrayList {
public:
    // ---- 标准容器所需的嵌套类型别名 ----
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;

    // -----------------------------------------------------------------
    // 迭代器 / 游标（iterator / cursor）
    // -----------------------------------------------------------------
    // 迭代器是「指针的泛化」：它支持 *、++、== 等操作，但不要求底层真是指针。
    // 对数组表示而言，迭代器恰好可以是「裸指针的薄封装」；对链表表示
    // （第 3 章）则必须封装节点指针。上层算法只依赖迭代器接口，
    // 于是同一段 std::sort 式代码能同时跑在数组和链表上。
    template <typename Ptr, typename Ref>
    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag; // 随机访问
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = Ptr;
        using reference         = Ref;

        Iterator() : pos_(nullptr) {}
        explicit Iterator(Ptr p) : pos_(p) {}

        Ref operator*()  const { return *pos_; }        // 解引用：取元素
        Ptr operator->() const { return pos_; }

        Iterator& operator++() { ++pos_; return *this; }             // 前置 ++
        Iterator  operator++(int) { Iterator t(*this); ++pos_; return t; } // 后置 ++
        Iterator& operator--() { --pos_; return *this; }
        Iterator  operator--(int) { Iterator t(*this); --pos_; return t; }

        Iterator& operator+=(difference_type n) { pos_ += n; return *this; }
        Iterator& operator-=(difference_type n) { pos_ -= n; return *this; }
        Iterator  operator+(difference_type n) const { return Iterator(pos_ + n); }
        Iterator  operator-(difference_type n) const { return Iterator(pos_ - n); }
        difference_type operator-(const Iterator& o) const { return pos_ - o.pos_; }

        Ref operator[](difference_type n) const { return pos_[n]; }

        bool operator==(const Iterator& o) const { return pos_ == o.pos_; }
        bool operator!=(const Iterator& o) const { return pos_ != o.pos_; }
        bool operator< (const Iterator& o) const { return pos_ <  o.pos_; }
        bool operator> (const Iterator& o) const { return pos_ >  o.pos_; }
        bool operator<=(const Iterator& o) const { return pos_ <= o.pos_; }
        bool operator>=(const Iterator& o) const { return pos_ >= o.pos_; }

    private:
        Ptr pos_;
    };

    using iterator       = Iterator<pointer, reference>;
    using const_iterator = Iterator<const_pointer, const_reference>;

    // ============================ 构造 / 析构 ============================

    // 默认构造：不分配内存（lazy），首次插入才分配，避免空容器浪费。
    arrayList() = default;

    // 指定初始容量。注意是「容量」而非「大小」：size() 仍为 0。
    explicit arrayList(size_type initialCapacity) {
        if (initialCapacity > 0) {
            data_ = allocate(initialCapacity);
            capacity_ = initialCapacity;
        }
    }

    arrayList(std::initializer_list<T> init) {
        if (init.size() > 0) {
            data_ = allocate(init.size());
            capacity_ = init.size();
            for (const T& v : init) constructAt(size_, v), ++size_;
        }
    }

    // 拷贝构造：深拷贝，容量按实际大小给（复制一个「紧凑」的副本）。
    arrayList(const arrayList& other) {
        if (other.size_ > 0) {
            data_ = allocate(other.size_);
            capacity_ = other.size_;
            for (size_type i = 0; i < other.size_; ++i) constructAt(i, other.data_[i]);
            size_ = other.size_;
        }
    }

    // 移动构造：只偷指针，O(1)。必须 noexcept，否则 vector<arrayList> 扩容会退化成拷贝。
    arrayList(arrayList&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = other.capacity_ = 0;
    }

    ~arrayList() { release(); }

    // 拷贝赋值：copy-and-swap，天然强异常安全（构造副本失败则原对象不变）。
    arrayList& operator=(const arrayList& other) {
        if (this != &other) {
            arrayList tmp(other);   // 可能抛异常；抛出时 *this 未被触碰
            swap(tmp);
        }
        return *this;
    }

    arrayList& operator=(arrayList&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_; size_ = other.size_; capacity_ = other.capacity_;
            other.data_ = nullptr; other.size_ = other.capacity_ = 0;
        }
        return *this;
    }

    void swap(arrayList& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    // ============================ 容量 / 大小 ============================

    size_type size()     const { return size_; }      // 元素个数 n
    bool      empty()    const { return size_ == 0; }
    size_type capacity() const { return capacity_; }  // 已分配槽位数（实现细节）

    // 预分配至少 n 个槽位，避免后续多次扩容。对应 std::vector::reserve。
    void reserve(size_type n) {
        if (n > capacity_) reallocate(n);
    }

    // 把容量收缩到恰好等于 size()，释放多余内存。对应 std::vector::shrink_to_fit。
    void shrink_to_fit() {
        if (size_ == capacity_) return;
        if (size_ == 0) { release(); return; }
        reallocate(size_);
    }

    // ============================ 元素访问 ============================

    // get(i)：0-based 索引。越界抛异常（而不是像 operator[] 那样未定义行为）。
    reference get(size_type i) {
        checkIndex(i);
        return data_[i];
    }
    const_reference get(size_type i) const {
        checkIndex(i);
        return data_[i];
    }

    // set(i, x)：把索引 i 处元素设为 x。
    void set(size_type i, const T& value) {
        checkIndex(i);
        data_[i] = value;
    }

    // 下表运算符：不做边界检查，追求速度（与 std::vector 一致）。
    reference       operator[](size_type i)       { return data_[i]; }
    const_reference operator[](size_type i) const { return data_[i]; }

    reference       front()       { assert(size_ > 0); return data_[0]; }
    const_reference front() const { assert(size_ > 0); return data_[0]; }
    reference       back()        { assert(size_ > 0); return data_[size_ - 1]; }
    const_reference back()  const { assert(size_ > 0); return data_[size_ - 1]; }

    // indexOf(x)：顺序查找第一个等于 x 的元素，返回其索引；没有返回 -1。
    // 复杂度：最坏 O(n)。这正是「数组表示」的固有代价之一（无序时无法更快）。
    // 注意返回值类型用 difference_type（有符号），因为 -1 无法用 size_type 表示。
    difference_type indexOf(const T& value) const {
        for (size_type i = 0; i < size_; ++i)
            if (data_[i] == value) return static_cast<difference_type>(i);
        return -1;
    }

    bool contains(const T& value) const { return indexOf(value) != -1; }

    // ============================ 修改操作 ============================

    // insert(index, value)：在索引 index 处插入 value。
    //   * index 的合法范围是 [0, size_]（允许「插到末尾」，即 index == size_）。
    //   * 语义：插入后 value 成为第 (index+1) 个元素（0-based 索引 index）。
    //   * 代价：索引 index..size_-1 的元素整体后移一格，最坏 O(n) 次搬移。
    //     平均（等概率插在任意位置）搬移 n/2 个元素，仍为 O(n)。
    //
    // 异常安全：先扩容（ensureCapacityForInsert 用「先分配新块再搬移」的写法，
    // 扩容本身给出强保证）；随后的后移 + 构造若中途抛异常，容器仍处于「有效但
    // 元素可能已错位」的**基本保证**状态 —— 不是完整的强保证。
    //
    // 这里用「完美转发模板」而不是分别重载 (const T&) 与 (T&&)：
    // 若写成两个重载，`insert(1, "hello")`（实参是 const char[6]）会让编译器
    // 在两个都需要「用户自定义转换」的候选之间无法取舍 —— 二义性编译错误。
    // 模板把 U 推导为 const char(&)[6]，再交给 T 的构造函数，天然正确。
    template <typename U>
    void insert(size_type index, U&& value) {
        if (index > size_) throw std::out_of_range("arrayList::insert 位置越界");
        ensureCapacityForInsert();          // 先确保容量足够；此步可能抛异常
        // 从后往前搬移，为 value 腾出索引 index 这个空位。
        for (size_type i = size_; i > index; --i) {
            constructAt(i, std::move_if_noexcept(data_[i - 1]));
            destroyAt(i - 1);
        }
        constructAt(index, std::forward<U>(value));
        ++size_;
    }

    // 尾部追加：实践中最常用的插入。摊还 O(1)。
    void push_back(const T& value) { insert(size_, value); }
    void push_back(T&& value)      { insert(size_, std::move(value)); }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
        ensureCapacityForInsert();
        new (data_ + size_) T(std::forward<Args>(args)...);   // 就地构造，省一次搬移
        return data_[size_++];
    }

    // erase(index)：删除索引 index 处的元素。
    //   * index 合法范围 [0, size_-1]。
    //   * 代价：索引 index+1..size_-1 的元素整体前移一格，最坏 O(n)。
    void erase(size_type index) {
        checkIndex(index);
        // 从 index 往后，把后继元素依次前移覆盖。
        for (size_type i = index; i + 1 < size_; ++i) {
            data_[i] = std::move_if_noexcept(data_[i + 1]);
        }
        destroyAt(size_ - 1);   // 最后一个元素成为逻辑上的「多余」，析构掉
        --size_;
    }

    void pop_back() {
        assert(size_ > 0);
        destroyAt(size_ - 1);
        --size_;
    }

    // 删除区间 [from, to) 的所有元素，一次搬移完成（比反复 erase 高效）。
    void eraseRange(size_type from, size_type to) {
        if (from > to || to > size_) throw std::out_of_range("arrayList::eraseRange 越界");
        size_type removed = to - from;
        if (removed == 0) return;
        for (size_type i = from; i + removed < size_; ++i) {
            data_[i] = std::move_if_noexcept(data_[i + removed]);
        }
        for (size_type i = size_ - removed; i < size_; ++i) destroyAt(i);
        size_ -= removed;
    }

    void clear() {
        for (size_type i = 0; i < size_; ++i) destroyAt(i);
        size_ = 0;
        // 注意：clear 不释放内存（capacity 保留），重新填充时无需再分配。
    }

    // ============================ 打印 / 迭代器 ============================

    void output(std::ostream& os, const char* sep = " ") const {
        for (size_type i = 0; i < size_; ++i) {
            if (i) os << sep;
            os << data_[i];
        }
    }

    iterator       begin()        { return iterator(data_); }
    iterator       end()          { return iterator(data_ + size_); }
    const_iterator begin()  const { return const_iterator(data_); }
    const_iterator end()    const { return const_iterator(data_ + size_); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend()   const { return end(); }

private:
    // ============================ 内存原语 ============================
    //
    // 关键区分：分配内存（allocate）≠ 构造对象（construct）。
    // arrayList<int> 这类 POD 无所谓，但对 std::string 等类型，
    // 必须「先分配原始内存，再在需要的位置构造，销毁后释放内存」。
    // 这正是 std::allocator / placement new 存在的原因，也是
    // std::vector 能只对「已构造区间」做析构的前提。

    static pointer allocate(size_type n) {
        return static_cast<pointer>(::operator new(n * sizeof(T)));
    }
    static void deallocate(pointer p) noexcept {
        ::operator delete(p);
    }

    // 定位 new（placement new）：在 data_ 的第 i 个槽位构造对象。
    // 传入的是「索引」而非指针，读起来与 get(i)/set(i) 一致。
    void constructAt(size_type i, const T& value) { new (data_ + i) T(value); }
    void constructAt(size_type i, T&& value)      { new (data_ + i) T(std::move(value)); }
    void destroyAt(size_type i) { (data_ + i)->~T(); }

    // 在一块「独立的原始内存」上构造/销毁（reallocate 的临时新缓冲区用这套）。
    static void constructRaw(pointer p, const T& value) { new (p) T(value); }
    static void constructRaw(pointer p, T&& value)      { new (p) T(std::move(value)); }
    static void destroyRaw(pointer p) { p->~T(); }

    void checkIndex(size_type i) const {
        if (i >= size_) throw std::out_of_range("arrayList 索引越界");
    }

    // -----------------------------------------------------------------
    // 扩容：倍增策略（doubling）
    // -----------------------------------------------------------------
    // 为什么是倍增而不是「每次 +1」？设插入 n 次，每次扩容搬移当前全部元素：
    //   * +1 策略：第 k 次搬移 k-1 个，总计 0+1+...+(n-1) = n(n-1)/2 = O(n^2)。
    //   * 倍增策略：容量序列 1,2,4,...,2^m >= n，第 k 次搬移 2^{k-1} 个，
    //     总计 1+2+4+...+2^{m-1} < 2^m <= 2n = O(n)。
    // 于是 n 次插入总代价 O(n)，每次插入摊还（amortized）代价 O(1)。
    // 详见 02_amortized_growth.cpp 的实测。
    //
    // 强异常安全：先在新内存上成功构造全部元素，再释放旧内存并更新成员。
    // 若中途抛异常，释放新内存并原样抛出，旧内存与成员变量纹丝不动。
    void reallocate(size_type newCapacity) {
        pointer fresh = nullptr;
        size_type constructed = 0;
        try {
            fresh = allocate(newCapacity);
            for (; constructed < size_; ++constructed) {
                // move_if_noexcept：若 T 的移动构造可能抛异常且可拷贝，则退化为拷贝，
                // 从而保证「搬移过程本身」不抛异常，强保证才成立。
                constructRaw(fresh + constructed,
                             std::move_if_noexcept(data_[constructed]));
            }
        } catch (...) {
            for (size_type i = 0; i < constructed; ++i) destroyRaw(fresh + i);
            if (fresh) deallocate(fresh);
            throw;                       // 原对象未被修改，强异常安全达成
        }
        // 到这里新缓冲区已就绪，下面都是不抛异常的操作。
        if (data_) {
            for (size_type i = 0; i < size_; ++i) destroyAt(i);
            deallocate(data_);
        }
        data_ = fresh;
        capacity_ = newCapacity;
    }

    void ensureCapacityForInsert() {
        if (size_ < capacity_) return;
        // 倍增；容量为 0 时从 1 起步。对超大对象可改用 1.5 倍（见正文）。
        size_type newCap = (capacity_ == 0) ? 1 : capacity_ * 2;
        reallocate(newCap);
    }

    void release() noexcept {
        if (!data_) return;
        for (size_type i = 0; i < size_; ++i) destroyAt(i);
        deallocate(data_);
        data_ = nullptr;
        size_ = capacity_ = 0;
    }

    pointer   data_     = nullptr;   // 指向已分配内存块首地址
    size_type size_     = 0;         // 已构造元素个数
    size_type capacity_ = 0;         // 已分配槽位数
};

// 非成员 swap（ADL 可找到）
template <typename T>
void swap(arrayList<T>& a, arrayList<T>& b) noexcept { a.swap(b); }

template <typename T>
std::ostream& operator<<(std::ostream& os, const arrayList<T>& l) {
    os << "[";
    l.output(os, ", ");
    os << "]";
    return os;
}

template <typename T>
bool operator==(const arrayList<T>& a, const arrayList<T>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!(a[i] == b[i])) return false;
    return true;
}

// ===========================================================================
// 2. 自测
// ===========================================================================

static void test_adt_basics() {
    std::cout << "\n=== 1. ADT 基础：size/empty/get/set/indexOf ===\n";
    arrayList<int> a;
    assert(a.empty() && a.size() == 0);

    for (int i = 0; i < 5; ++i) a.push_back(i * 10);
    std::cout << "a = " << a << "\n";
    assert(a.size() == 5);
    assert(a.get(0) == 0 && a.get(4) == 40);

    a.set(2, 999);
    assert(a[2] == 999);
    std::cout << "set(2,999) -> " << a << "\n";

    // indexOf：找得到返回索引，找不到返回 -1
    assert(a.indexOf(999) == 2);
    assert(a.indexOf(40) == 4);
    assert(a.indexOf(-1) == -1);
    std::cout << "indexOf(999) = " << a.indexOf(999)
              << ", indexOf(-1) = " << a.indexOf(-1) << "\n";

    // 越界必须抛异常，而不是 UB
    bool threw = false;
    try { (void)a.get(100); } catch (const std::out_of_range& e) {
        threw = true; std::cout << "越界捕获: " << e.what() << "\n";
    }
    assert(threw);
}

static void test_zero_based_vs_ordinal() {
    std::cout << "\n=== 2. 0-based 索引 vs 「第 k 个元素」 ===\n";
    arrayList<std::string> a{"zero", "one", "two", "three"};
    // 第 1 个元素是索引 0
    std::cout << "第 1 个元素 = " << a.get(0)
              << "，第 3 个元素 = " << a.get(2) << "\n";
    assert(a.get(0) == "zero");
    assert(a.get(2) == "two");

    // 「在第 2 个位置插入 X」（1-based 序数 2）=> 0-based 索引 1 => 新元素成为第 2 个
    a.insert(1, "INSERTED");
    std::cout << "在第 2 个位置插入后: " << a << "\n";
    assert(a.get(1) == "INSERTED");
    assert(a.get(2) == "one");   // 原来的 "one" 后移一位
}

static void test_insert_erase() {
    std::cout << "\n=== 3. insert/erase 的搬移语义 ===\n";
    arrayList<char> a{'a', 'b', 'c', 'd'};
    std::cout << "初始        : " << a << "\n";

    a.insert(0, 'X');                 // 头插
    std::cout << "insert(0,X) : " << a << "\n";
    assert((a == arrayList<char>{'X', 'a', 'b', 'c', 'd'}));

    a.insert(a.size(), 'Z');          // 等价 push_back：index == size 合法
    std::cout << "insert(end,Z): " << a << "\n";
    assert(a.back() == 'Z');

    a.erase(0);                       // 删头
    std::cout << "erase(0)    : " << a << "\n";
    assert(a.front() == 'a');

    a.erase(a.size() - 1);            // 删尾
    std::cout << "erase(end)  : " << a << "\n";
    assert(a.back() == 'd');

    // 非法位置：insert 只允许 [0, size]
    bool threw = false;
    try { a.insert(a.size() + 1, '!'); }
    catch (const std::out_of_range&) { threw = true; }
    assert(threw);

    // eraseRange 一次搬移删一段
    arrayList<int> b{0, 1, 2, 3, 4, 5, 6, 7};
    b.eraseRange(2, 5);               // 删除索引 2,3,4
    std::cout << "eraseRange(2,5) 于 [0..7]: " << b << "\n";
    assert((b == arrayList<int>{0, 1, 5, 6, 7}));
}

static void test_growth_and_strong_guarantee() {
    std::cout << "\n=== 4. 倍增扩容 / reserve / shrink_to_fit ===\n";
    arrayList<int> a;
    std::cout << "capacity 增长序列: ";
    for (int i = 0; i < 20; ++i) {
        std::size_t before = a.capacity();
        a.push_back(i);
        if (a.capacity() != before) std::cout << a.capacity() << " ";
    }
    std::cout << "\n";
    assert(a.size() == 20 && a.capacity() >= 20);

    // reserve 只改容量，不改 size
    arrayList<int> b;
    b.reserve(100);
    assert(b.capacity() >= 100 && b.size() == 0);
    std::cout << "reserve(100) 后 capacity = " << b.capacity()
              << ", size = " << b.size() << "\n";

    for (int i = 0; i < 10; ++i) b.push_back(i);
    b.shrink_to_fit();
    std::cout << "shrink_to_fit 后 capacity = " << b.capacity()
              << ", size = " << b.size() << "\n";
    assert(b.capacity() == b.size());

    // clear 保留容量
    b.clear();
    assert(b.empty() && b.capacity() == 10);
    std::cout << "clear 后 size = 0, capacity = " << b.capacity() << "（内存未归还）\n";
}

static void test_copy_move_iterators() {
    std::cout << "\n=== 5. 拷贝/移动语义 与 迭代器 ===\n";
    arrayList<std::string> a{"alpha", "beta", "gamma"};
    arrayList<std::string> copy = a;
    assert(copy == a);
    std::cout << "深拷贝: " << copy << "\n";

    arrayList<std::string> moved = std::move(copy);
    assert(moved.size() == 3 && copy.size() == 0);
    std::cout << "移动后源 size = " << copy.size() << "（已置空）\n";

    // 迭代器：range-based for 会自动展开成 begin()/end() 循环
    std::cout << "遍历: ";
    for (const auto& s : moved) std::cout << s << " ";
    std::cout << "\n";

    for (auto& s : moved) s += "!";
    assert(moved.front() == "alpha!");

    // 迭代器是随机访问的：支持 +、-、下标、距离
    auto it = moved.begin();
    std::cout << "*it = " << *it << ", it[2] = " << it[2]
              << ", end-begin = " << (moved.end() - moved.begin()) << "\n";
    assert(it[1] == "beta!");
    assert((moved.end() - moved.begin()) == 3);

    // 标准算法可与我们的迭代器配合
    arrayList<int> nums{5, 3, 1, 4, 2};
    std::sort(nums.begin(), nums.end());
    std::cout << "std::sort 后: " << nums << "\n";
    assert(std::is_sorted(nums.begin(), nums.end()));

    // 嵌套容器
    arrayList<arrayList<int>> grid;
    for (int i = 0; i < 3; ++i) {
        arrayList<int> row;
        for (int j = 0; j <= i; ++j) row.push_back(j);
        grid.push_back(std::move(row));
    }
    std::cout << "grid 行数 = " << grid.size()
              << ", 末行 = " << grid.back() << "\n";
    assert(grid.size() == 3 && grid.back().size() == 3);
}

static void test_string_resource_safety() {
    std::cout << "\n=== 6. 非平凡类型：析构/搬移都被正确调用 ===\n";
    // string 会分配堆内存，若我们只 memcpy 而不调用构造函数，就会双重释放。
    arrayList<std::string> a;
    for (int i = 0; i < 100; ++i) a.push_back("item-" + std::to_string(i));
    assert(a.size() == 100);
    assert(a.get(99) == "item-99");

    a.erase(0);
    assert(a.front() == "item-1");
    std::cout << "100 个 string 插入 + 删除首元素，无泄漏/无崩溃（ASan 下更严格）\n";
    std::cout << "front = " << a.front() << ", back = " << a.back()
              << ", size = " << a.size() << "\n";
}

int main() {
    std::cout << "======== 02 线性表与数组：arrayList<T> 自测 ========\n";
    test_adt_basics();
    test_zero_based_vs_ordinal();
    test_insert_erase();
    test_growth_and_strong_guarantee();
    test_copy_move_iterators();
    test_string_resource_safety();
    std::cout << "\n全部断言通过，程序正常结束。\n";
    return 0;
}
