// 03_chain.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 03 章 链表》
// 主题：单向链表的完整实现 chain<T>（Sahni 第 6 章 Linear Lists: Linked Representation）
//
//   本文件想讲清三件事：
//     (1) 「数组表示」与「链表表示」的根本差异：链表用「指针」把散落的内存块串起来，
//         于是插入/删除只需改指针、O(1)；代价是失去随机访问，访问第 i 个元素 O(i)。
//     (2) 链表代码 90% 的 bug 来自「边界」：空表、头插、尾删、只有一个结点、
//         以及「先断开还是先重连」的顺序错误。本文件用断言把每个边界钉死。
//     (3) 链表是「手动内存管理」的主战场：必须遵守「三/五法则」，否则浅拷贝会
//         导致双重释放（double free）。这是 C++ 与 Python 最不一样的地方。
//
//   重要工程结论（第 5 节实验会实测）：
//     对「连续内存友好」的现代 CPU/GPU 来说，数组表示往往比链表快得多，
//     哪怕理论复杂度链表更优。这就是为什么 PyTorch 张量坚持连续内存，
//     也是为什么工程上「list 换 vector」常能白赚几倍性能。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 03_chain.cpp -o chain && ./chain
// 消毒器版本（强烈推荐，能抓出所有悬垂指针/泄漏）：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 03_chain.cpp -o chain_san && ./chain_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>      // std::forward_iterator_tag
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>       // std::move, std::swap

// ===========================================================================
// 单向链表 chain<T>
// ===========================================================================
//
// 结点（node）存 value 与 next 指针。整条链由 head_ 出发，最后一个结点的
// next == nullptr 表示「链尾」。size_ 单独维护，使 size() 为 O(1)。
//
// 为什么不用「哑结点（dummy/sentinel head）」？因为很多教材先说清「裸头指针」
// 版本的边界麻烦，再引入哑结点消除它们。本文件保留裸头指针版本；
// 03_doubly_linked_list.cpp 则用「循环哑结点」展示优雅写法，两相对照。
template <typename T>
class chain {
private:
    struct node {
        T     value;
        node* next;
        node(const T& v, node* n) : value(v), next(n) {}
        node(T&& v, node* n) : value(std::move(v)), next(n) {}
    };

public:
    // -----------------------------------------------------------------
    // 迭代器：把 node* 包一层，支持 *、->、前置/后置 ++、==、!=
    // 有了它，range-based for 与 <algorithm> 里的算法都能直接用。
    // 分类是 forward_iterator（只能前进，不能 -- 或随机跳）。
    // -----------------------------------------------------------------
    template <bool IsConst>
    class Iterator {
        friend class chain;
        using NodePtr = std::conditional_t<IsConst, const node*, node*>;
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = std::conditional_t<IsConst, const T*, T*>;
        using reference         = std::conditional_t<IsConst, const T&, T&>;

        Iterator() : p_(nullptr) {}
        explicit Iterator(NodePtr p) : p_(p) {}

        reference operator*()  const { return p_->value; }
        pointer   operator->() const { return &p_->value; }

        Iterator& operator++() { p_ = p_->next; return *this; }        // 前置
        Iterator  operator++(int) { Iterator t(*this); p_ = p_->next; return t; }

        bool operator==(const Iterator& o) const { return p_ == o.p_; }
        bool operator!=(const Iterator& o) const { return p_ != o.p_; }

    private:
        NodePtr p_;
    };

    using iterator       = Iterator<false>;
    using const_iterator = Iterator<true>;

    // ============================ 构造 / 析构 ============================

    chain() = default;

    chain(std::initializer_list<T> init) {
        node** tail = &head_;                 // 「指向 next 指针的指针」——尾插利器
        for (const T& v : init) { *tail = new node(v, nullptr); tail = &(*tail)->next; }
        size_ = init.size();
    }

    // 拷贝构造：深拷贝。若只做浅拷贝（复制 head_），两条链共用同一批结点，
    // 析构时就会「双重释放」——这是链表最经典的灾难。
    chain(const chain& other) {
        node** tail = &head_;
        for (const node* p = other.head_; p; p = p->next) {
            *tail = new node(p->value, nullptr);
            tail = &(*tail)->next;
            ++size_;
        }
    }

    // 移动构造：只偷走头指针，O(1)，并把源置空（否则源析构会释放这批结点）。
    chain(chain&& other) noexcept
        : head_(other.head_), size_(other.size_) {
        other.head_ = nullptr;
        other.size_ = 0;
    }

    ~chain() { clear(); }

    chain& operator=(const chain& other) {
        if (this != &other) {
            chain tmp(other);      // 先深拷贝；失败则 *this 不变（强异常安全）
            swap(tmp);
        }
        return *this;
    }

    chain& operator=(chain&& other) noexcept {
        if (this != &other) {
            clear();
            head_ = other.head_; size_ = other.size_;
            other.head_ = nullptr; other.size_ = 0;
        }
        return *this;
    }

    void swap(chain& other) noexcept {
        std::swap(head_, other.head_);
        std::swap(size_, other.size_);
    }

    // ============================ 容量 / 访问 ============================

    std::size_t size()  const { return size_; }
    bool        empty() const { return head_ == nullptr; }

    T&       front()       { assert(head_); return head_->value; }
    const T& front() const { assert(head_); return head_->value; }

    // back() 需要走到链尾，O(n)：这是单链表没有尾指针的固有代价。
    T&       back()       { assert(head_); node* p = head_; while (p->next) p = p->next; return p->value; }
    const T& back() const { assert(head_); const node* p = head_; while (p->next) p = p->next; return p->value; }

    // get(i)：走到第 i 个结点，O(i)。链表无法 O(1) 随机访问 —— 关键 trade-off。
    T& get(std::size_t i) {
        assert(i < size_);
        node* p = head_;
        while (i--) p = p->next;
        return p->value;
    }
    const T& get(std::size_t i) const {
        assert(i < size_);
        const node* p = head_;
        while (i--) p = p->next;
        return p->value;
    }

    // indexOf：第一个等于 x 的元素位置；无则返回 -1。O(n)。
    std::ptrdiff_t indexOf(const T& x) const {
        std::ptrdiff_t i = 0;
        for (const node* p = head_; p; p = p->next, ++i)
            if (p->value == x) return i;
        return -1;
    }

    // ============================ 修改操作 ============================
    //
    // 链表插入/删除的核心套路：先找到「目标位置的前驱」，再改指针。
    //   插入：newNode->next = prev->next;  prev->next = newNode;
    //         —— 顺序不能反！若先写 prev->next = newNode，就再也找不到原后继了。
    //   删除：prev->next = cur->next;  delete cur;
    //         —— 必须先保存 cur->next（这里直接读 cur->next 再删），再 delete。

    // 头插：O(1)。注意必须「先让新结点指向旧头，再改 head_」。
    void push_front(const T& v) { head_ = new node(v, head_); ++size_; }
    void push_front(T&& v)      { head_ = new node(std::move(v), head_); ++size_; }

    // 尾插：没有尾指针时是 O(n)，因为要走到链尾。
    void push_back(const T& v) {
        node** tail = &head_;
        while (*tail) tail = &(*tail)->next;
        *tail = new node(v, nullptr);
        ++size_;
    }
    void push_back(T&& v) {
        node** tail = &head_;
        while (*tail) tail = &(*tail)->next;
        *tail = new node(std::move(v), nullptr);
        ++size_;
    }

    // insert(pos, v)：在 0-based 位置 pos 插入；pos == size() 表示追加到末尾。
    // 合法范围 [0, size]。
    template <typename U>
    void insert(std::size_t pos, U&& v) {
        if (pos > size_) throw std::out_of_range("chain::insert 位置越界");
        node** link = &head_;                 // link 指向「要接上新结点的那个 next 指针」
        for (std::size_t i = 0; i < pos; ++i) link = &(*link)->next;
        *link = new node(std::forward<U>(v), *link);   // 新结点接上原后继，再改 link
        ++size_;
    }

    // erase(pos)：删除 0-based 位置 pos 的结点；合法范围 [0, size)。
    void erase(std::size_t pos) {
        if (pos >= size_) throw std::out_of_range("chain::erase 位置越界");
        node** link = &head_;
        for (std::size_t i = 0; i < pos; ++i) link = &(*link)->next;
        node* doomed = *link;
        *link = doomed->next;                 // 跨过被删结点
        delete doomed;                        // 释放内存（链表必须手动 delete）
        --size_;
    }

    void pop_front() { assert(head_); erase(0); }

    // 删除「第一个等于 x」的结点；返回是否删掉了。
    bool remove(const T& x) {
        node** link = &head_;
        while (*link && !((*link)->value == x)) link = &(*link)->next;
        if (!*link) return false;
        node* doomed = *link;
        *link = doomed->next;
        delete doomed;
        --size_;
        return true;
    }

    // 就地反转：把每个结点的 next 掉头。经典三指针法，O(n) 时间、O(1) 空间。
    // 迭代写法务必「先存 next，再掉头，再前进」，否则会丢链。
    void reverse() {
        node* prev = nullptr;
        node* cur  = head_;
        while (cur) {
            node* nxt = cur->next;   // 1. 先记住后继
            cur->next = prev;        // 2. 掉头
            prev = cur;              // 3. 前进
            cur  = nxt;
        }
        head_ = prev;                // 新的头是原来的尾
    }

    void clear() noexcept {
        while (head_) {
            node* nxt = head_->next;
            delete head_;
            head_ = nxt;
        }
        size_ = 0;
    }

    // ============================ 迭代器 ============================

    iterator       begin()        { return iterator(head_); }
    iterator       end()          { return iterator(nullptr); }
    const_iterator begin()  const { return const_iterator(head_); }
    const_iterator end()    const { return const_iterator(nullptr); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend()   const { return end(); }

    void output(std::ostream& os, const char* sep = " ") const {
        bool first = true;
        for (const node* p = head_; p; p = p->next) {
            if (!first) os << sep;
            os << p->value;
            first = false;
        }
    }

private:
    node*       head_ = nullptr;
    std::size_t size_ = 0;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const chain<T>& c) {
    os << "[";
    c.output(os, " -> ");
    os << "]";
    return os;
}

template <typename T>
bool operator==(const chain<T>& a, const chain<T>& b) {
    if (a.size() != b.size()) return false;
    auto ia = a.begin();
    auto ib = b.begin();
    for (; ia != a.end(); ++ia, ++ib)
        if (!(*ia == *ib)) return false;
    return true;
}

// ===========================================================================
// 自测
// ===========================================================================

static void test_basics() {
    std::cout << "\n=== 1. 基本操作 ===\n";
    chain<int> c;
    assert(c.empty() && c.size() == 0);
    for (int i = 1; i <= 5; ++i) c.push_back(i * 10);
    std::cout << "尾插 10..50 : " << c << "\n";
    assert(c.size() == 5 && c.front() == 10 && c.back() == 50);

    c.push_front(0);
    std::cout << "头插 0      : " << c << "\n";
    assert(c.front() == 0);

    assert(c.get(3) == 30);                                  // O(i) 随机访问
    assert(c.indexOf(30) == 3 && c.indexOf(999) == -1);
    std::cout << "get(3) = " << c.get(3) << ", indexOf(30) = " << c.indexOf(30) << "\n";

    c.insert(0, -1);                                         // 头插（等价 push_front）
    c.insert(c.size(), 60);                                  // 尾追（等价 push_back）
    c.insert(3, 999);                                        // 中间插
    std::cout << "insert 后    : " << c << "\n";
    assert(c.front() == -1 && c.back() == 60 && c.get(3) == 999);

    c.erase(3);                                              // 删中间
    c.pop_front();                                           // 删头
    std::cout << "erase/pop 后 : " << c << "\n";
    assert(c.front() == 0 && c.size() == 7);

    bool threw = false;
    try { c.insert(c.size() + 1, 1); } catch (const std::out_of_range&) { threw = true; }
    assert(threw);
    std::cout << "越界 insert 已正确抛异常\n";
}

static void test_sentinel_edges() {
    std::cout << "\n=== 2. 边界：空表 / 单元素 / 头尾 ===\n";
    chain<int> c;
    c.push_back(42);                       // 空表 -> 单元素
    assert(c.size() == 1 && c.front() == 42 && c.back() == 42);
    c.erase(0);                            // 删到空
    assert(c.empty() && c.size() == 0);
    c.push_front(7);                       // 空表头插
    assert(c.front() == 7 && c.back() == 7);
    c.clear();
    assert(c.empty());

    // remove：删头 / 删尾 / 删不存在
    chain<int> d{1, 2, 3, 2, 1};
    assert(d.remove(1));      // 删第一个 1（头）
    std::cout << "remove(1) 后 : " << d << "\n";
    assert(d.remove(1));      // 删最后一个 1（尾）
    std::cout << "再 remove(1) : " << d << "\n";
    assert(!d.remove(1));     // 已无 1
    assert((d == chain<int>{2, 3, 2}));
}

static void test_rule_of_five() {
    std::cout << "\n=== 3. 三/五法则：深拷贝 vs 移动 ===\n";
    chain<std::string> a{"alpha", "beta", "gamma"};
    chain<std::string> b = a;                       // 深拷贝
    assert(b == a);
    b.push_back("delta");
    assert(a.size() == 3 && b.size() == 4);         // 互不影响 —— 证明是深拷贝
    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "（改动 b 不影响 a）\n";

    chain<std::string> c = std::move(a);            // 移动：偷指针
    assert(c.size() == 3 && a.empty());             // 源被置空
    std::cout << "移动后 c = " << c << ", a 已置空 size = " << a.size() << "\n";

    // 自赋值必须安全（copy-and-swap 天然处理）
    chain<std::string> d{"x"};
    d = d;
    assert((d == chain<std::string>{"x"}));
    std::cout << "自赋值 a = a 安全\n";
}

static void test_iterators_and_algorithms() {
    std::cout << "\n=== 4. 迭代器 / range-for / 反转 ===\n";
    chain<int> c{5, 3, 1, 4, 2};
    std::cout << "初始       : " << c << "\n";

    chain<int> copy = c;
    copy.reverse();
    std::cout << "reverse 后 : " << copy << "\n";
    assert((copy == chain<int>{2, 4, 1, 3, 5}));

    // 迭代器遍历（forward iterator）
    int sum = 0;
    for (int v : c) sum += v;
    assert(sum == 15);
    std::cout << "range-for 求和 = " << sum << "\n";

    // 通过迭代器就地修改
    for (int& v : copy) v *= 10;
    std::cout << "迭代器改值 : " << copy << "\n";
    assert(copy.front() == 20);

    // 反转两次回到原样（幂等性自查）
    chain<int> e{1, 2, 3};
    e.reverse(); e.reverse();
    assert((e == chain<int>{1, 2, 3}));
    std::cout << "反转两次复原 ✓\n";
}

int main() {
    std::cout << "======== 03 链表：chain<T> 自测 ========";
    test_basics();
    test_sentinel_edges();
    test_rule_of_five();
    test_iterators_and_algorithms();
    std::cout << "\n全部断言通过，程序正常结束。\n";
    return 0;
}
