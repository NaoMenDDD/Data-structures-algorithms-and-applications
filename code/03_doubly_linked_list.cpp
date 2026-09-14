// 03_doubly_linked_list.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 03 章 链表》
// 主题：带「循环哑结点（circular sentinel）」的双向链表
//
//   为什么单链表写起来到处是边界判断，而双向 + 哑结点写起来几乎没有分支？
//   因为哑结点让「空表」不再特殊：
//
//         +-----------+      +-------+      +-------+      +-------+
//         | sentinel  | <--> | node0 | <--> | node1 | <--> | node2 | --+
//         +-----------+      +-------+      +-------+      +-------+   |
//              ^                                                        |
//              +--------------------------------------------------------+
//   整条链是一个「环」：sentinel.next 是首结点，sentinel.prev 是尾结点；
//   空表时 sentinel.next == sentinel.prev == &sentinel。
//
//   于是插入/删除只需写「在 p 之前插入」和「把 p 摘掉」两个原子操作，
//   头/尾/中间/空表全部走同一条路径 —— 没有 if (空表) / if (头) / if (尾)。
//   这正是 Linux 内核 list_head、C++ std::list 的做法。
//
//   每个结点同时持有 prev 与 next，代价是：每结点多一个指针；插入/删除要改 4 个
//   指针（且顺序不能错）。收益是：删除一个已知结点 O(1)（单链表做不到，除非给前驱）
//   —— 这正是 LRU 缓存（哈希表 + 双向链表）能 O(1) 的关键。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 03_doubly_linked_list.cpp -o dlist && ./dlist
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 03_doubly_linked_list.cpp -o dlist_san && ./dlist_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>      // std::bidirectional_iterator_tag
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// ---------------------------------------------------------------------------
// DList<T>：双向链表。T 需可默认构造（只为给哑结点一个占位对象）。
// ---------------------------------------------------------------------------
template <typename T>
class DList {
private:
    struct Node {
        T     value;
        Node* prev;
        Node* next;
        // 用完美转发构造函数，避免 (const T&)/(T&&) 双重重载在
        // 「实参是字符串字面量」时产生二义性（与 02_array_list.cpp 同一个坑）。
        template <typename U>
        Node(U&& v, Node* p, Node* n)
            : value(std::forward<U>(v)), prev(p), next(n) {}
    };

public:
    // 双向迭代器：支持 ++ 与 --
    template <bool IsConst>
    class Iterator {
        friend class DList;
        using NodePtr = std::conditional_t<IsConst, const Node*, Node*>;
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = std::conditional_t<IsConst, const T*, T*>;
        using reference         = std::conditional_t<IsConst, const T&, T&>;

        Iterator() : p_(nullptr) {}
        explicit Iterator(NodePtr p) : p_(p) {}

        reference operator*()  const { return p_->value; }
        pointer   operator->() const { return &p_->value; }

        Iterator& operator++() { p_ = p_->next; return *this; }
        Iterator  operator++(int) { Iterator t(*this); p_ = p_->next; return t; }
        Iterator& operator--() { p_ = p_->prev; return *this; }   // 双向：可后退
        Iterator  operator--(int) { Iterator t(*this); p_ = p_->prev; return t; }

        bool operator==(const Iterator& o) const { return p_ == o.p_; }
        bool operator!=(const Iterator& o) const { return p_ != o.p_; }
    private:
        NodePtr p_;
    };
    using iterator       = Iterator<false>;
    using const_iterator = Iterator<true>;

    // ============================ 构造 / 析构 ============================

    DList() { initSentinel(); }

    DList(std::initializer_list<T> init) {
        initSentinel();
        for (const T& v : init) push_back(v);
    }

    DList(const DList& other) {
        initSentinel();
        for (const Node* p = other.sentinel_->next; p != other.sentinel_; p = p->next)
            push_back(p->value);
    }

    DList(DList&& other) noexcept {
        initSentinel();          // 先给自己造一个空环
        spliceAllFrom(other);    // 无论 other 空不空，splice 都安全（空则什么都不接）
    }

    ~DList() { clear(); delete sentinel_; }

    DList& operator=(const DList& other) {
        if (this != &other) { DList tmp(other); swap(tmp); }
        return *this;
    }
    DList& operator=(DList&& other) noexcept {
        if (this != &other) { clear(); spliceAllFrom(other); }
        return *this;
    }

    void swap(DList& other) noexcept { std::swap(sentinel_, other.sentinel_); std::swap(size_, other.size_); }

    // ============================ 观察 ============================

    std::size_t size()  const { return size_; }
    bool        empty() const { return sentinel_->next == sentinel_; }

    T&       front()       { assert(!empty()); return sentinel_->next->value; }
    const T& front() const { assert(!empty()); return sentinel_->next->value; }
    T&       back()        { assert(!empty()); return sentinel_->prev->value; }
    const T& back()  const { assert(!empty()); return sentinel_->prev->value; }

    std::ptrdiff_t indexOf(const T& x) const {
        std::ptrdiff_t i = 0;
        for (const Node* p = sentinel_->next; p != sentinel_; p = p->next, ++i)
            if (p->value == x) return i;
        return -1;
    }

    // ============================ 修改（全部 O(1)）============================

    void push_front(const T& v) { insertBefore(sentinel_->next, v); }
    void push_front(T&& v)      { insertBefore(sentinel_->next, std::move(v)); }
    void push_back(const T& v)  { insertBefore(sentinel_, v); }   // 插在哑结点前 = 尾
    void push_back(T&& v)       { insertBefore(sentinel_, std::move(v)); }

    void pop_front() { assert(!empty()); unlink(sentinel_->next); }
    void pop_back()  { assert(!empty()); unlink(sentinel_->prev); }

    // 在「第 pos 个」之前插入（0-based，pos == size() 即尾插）
    template <typename U>
    void insert(std::size_t pos, U&& v) {
        if (pos > size_) throw std::out_of_range("DList::insert 越界");
        Node* at = pos == size_ ? sentinel_ : nodeAt(pos);
        insertBefore(at, std::forward<U>(v));
    }

    void erase(std::size_t pos) {
        if (pos >= size_) throw std::out_of_range("DList::erase 越界");
        unlink(nodeAt(pos));
    }

    // 反转：双向链表反转只需交换每个结点的 prev/next，最后再换哑结点的。
    // 比单链表的三指针法更直观，O(n) 时间、O(1) 空间。
    //
    // 遍历方向的巧妙之处：交换 p 的 prev/next 后，p->prev 恰好等于「原来的 next」，
    // 所以用 `p = p->prev` 就能继续沿原方向前进，一圈走完所有真实结点。
    void reverse() {
        for (Node* p = sentinel_->next; p != sentinel_; p = p->prev)
            std::swap(p->prev, p->next);
        std::swap(sentinel_->prev, sentinel_->next);
    }

    void clear() noexcept {
        Node* p = sentinel_->next;
        while (p != sentinel_) {
            Node* n = p->next;
            delete p;              // 必须 delete（会跑 ~Node -> ~T），不能用 ::operator delete
            p = n;
        }
        sentinel_->next = sentinel_->prev = sentinel_;
        size_ = 0;
    }

    // 把当前「首个」结点挪到链表最前（若已在前则不动）。LRU 缓存的核心动作。
    void moveToFront(std::size_t pos) {
        if (pos == 0 || pos >= size_) return;
        Node* p = nodeAt(pos);
        p->prev->next = p->next;      // 从原位置摘下
        p->next->prev = p->prev;
        p->prev = sentinel_;          // 插到最前
        p->next = sentinel_->next;
        sentinel_->next->prev = p;
        sentinel_->next = p;
    }

    // ============================ 迭代器 ============================

    iterator       begin()        { return iterator(sentinel_->next); }
    iterator       end()          { return iterator(sentinel_); }
    const_iterator begin()  const { return const_iterator(sentinel_->next); }
    const_iterator end()    const { return const_iterator(sentinel_); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend()   const { return end(); }

    void output(std::ostream& os, const char* sep = " <-> ") const {
        bool first = true;
        for (const Node* p = sentinel_->next; p != sentinel_; p = p->next) {
            if (!first) os << sep;
            os << p->value;
            first = false;
        }
    }

private:
    void initSentinel() {
        sentinel_ = new Node(T(), nullptr, nullptr);
        sentinel_->next = sentinel_;
        sentinel_->prev = sentinel_;
    }

    Node* nodeAt(std::size_t i) {
        // 双向链表可选从近端走：i 靠前从头，靠后从尾 —— 平均省一半。
        if (i < size_ / 2) {
            Node* p = sentinel_->next;
            while (i--) p = p->next;
            return p;
        } else {
            Node* p = sentinel_;
            std::size_t k = size_ - i;
            while (k--) p = p->prev;
            return p;
        }
    }

    // 在 before 之前插入一个新结点（4 条指针写全，顺序无所谓，只要都写对）
    template <typename U>
    void insertBefore(Node* before, U&& v) {
        Node* p = new Node(std::forward<U>(v), before->prev, before);
        before->prev->next = p;
        before->prev = p;
        ++size_;
    }

    // 把结点 p 摘下并释放（p 必须属于本链表且非哑结点）
    void unlink(Node* p) {
        p->prev->next = p->next;
        p->next->prev = p->prev;
        delete p;                  // 必须 delete（会跑 ~Node -> ~T）
        --size_;
    }

    // 把 other 的所有结点整体接到本链表尾部，并清空 other（用于移动语义）
    void spliceAllFrom(DList& other) {
        if (other.empty()) return;
        Node* first = other.sentinel_->next;
        Node* last  = other.sentinel_->prev;
        Node* tail  = sentinel_->prev;          // 本链表的尾

        tail->next  = first;  first->prev = tail;
        last->next  = sentinel_;  sentinel_->prev = last;

        size_ += other.size_;
        other.sentinel_->next = other.sentinel_->prev = other.sentinel_;
        other.size_ = 0;
    }

    Node*       sentinel_ = nullptr;
    std::size_t size_     = 0;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const DList<T>& l) {
    os << "<-> [";
    l.output(os, " <-> ");
    os << "]";
    return os;
}

template <typename T>
bool operator==(const DList<T>& a, const DList<T>& b) {
    if (a.size() != b.size()) return false;
    auto ia = a.begin(); auto ib = b.begin();
    for (; ia != a.end(); ++ia, ++ib) if (!(*ia == *ib)) return false;
    return true;
}

// ===========================================================================
// 自测
// ===========================================================================

static void test_all_positions() {
    std::cout << "\n=== 1. 头/尾/中/空表：每个位置都不特殊 ===\n";
    DList<int> d;
    assert(d.empty());
    d.push_front(2);                 // 空表头插
    d.push_front(1);
    d.push_back(3);                  // 空表非空后尾插
    d.push_back(4);
    std::cout << "构造 1..4   : " << d << "\n";
    assert((d == DList<int>{1, 2, 3, 4}));

    d.pop_front();
    d.pop_back();
    std::cout << "去头去尾后  : " << d << "\n";
    assert((d == DList<int>{2, 3}));

    d.insert(1, 99);                 // 中间插
    d.insert(0, 0);                  // 头插
    d.insert(d.size(), 9);           // 尾插
    std::cout << "插 0/99/9   : " << d << "\n";
    assert((d == DList<int>{0, 2, 99, 3, 9}));

    d.erase(0);
    d.erase(d.size() - 1);
    d.erase(1);
    std::cout << "erase 后    : " << d << "\n";
    assert((d == DList<int>{2, 3}));

    // 删到空再从空重建 —— 最容易出 bug 的路径
    d.pop_front(); d.pop_front();
    assert(d.empty() && d.size() == 0);
    d.push_back(7);
    std::cout << "删空后重建  : " << d << "\n";
    assert(d.front() == 7 && d.back() == 7);
}

static void test_bidirectional_iterator() {
    std::cout << "\n=== 2. 双向迭代器：可前进也可后退 ===\n";
    DList<std::string> d{"a", "b", "c", "d"};
    std::cout << "正向 : ";
    for (const auto& s : d) std::cout << s << " ";
    std::cout << "\n反向 : ";
    for (auto it = d.end(); it != d.begin(); ) { --it; std::cout << *it << " "; }
    std::cout << "\n";

    auto it = d.begin();
    ++it; ++it;
    assert(*it == "c" && *(--it) == "b");       // 前进两步、后退一步
    std::cout << "++ 到 c，-- 到 b：验证通过\n";
}

static void test_move_and_reverse() {
    std::cout << "\n=== 3. 移动语义 / 反转 / moveToFront ===\n";
    DList<int> a{1, 2, 3, 4, 5};
    DList<int> b = std::move(a);
    assert(b.size() == 5 && a.empty());
    std::cout << "移动后 b = " << b << ", a.size = " << a.size() << "\n";

    b.reverse();
    std::cout << "reverse  : " << b << "\n";
    assert((b == DList<int>{5, 4, 3, 2, 1}));

    b.moveToFront(4);                // 把末尾的 1 提到最前
    std::cout << "moveToFront(4): " << b << "\n";
    assert(b.front() == 1);
    b.moveToFront(0);                // 已在前，不动
    assert(b.front() == 1);

    DList<int> c{1, 2, 3};
    c.reverse(); c.reverse();
    assert((c == DList<int>{1, 2, 3}));   // 反转两次复原
    std::cout << "反转两次复原 ✓\n";
}

int main() {
    std::cout << "======== 03 双向链表：DList<T> 自测 ========";
    test_all_positions();
    test_bidirectional_iterator();
    test_move_and_reverse();
    std::cout << "\n全部断言通过，程序正常结束。\n";
    return 0;
}
