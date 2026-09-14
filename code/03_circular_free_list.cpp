// 03_circular_free_list.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 03 章 链表》
// 主题：循环链表与「自由表（free list）」
//
//   第 1 部分：循环单向链表 —— 尾结点的 next 指回头结点，形成环。
//     * 天然的「轮询（round-robin）」结构：转一圈回到起点。
//     * 经典应用：约瑟夫环（Josephus problem）。
//     * AI 关联：调度器里「轮流取下一个数据源/下一个 expert」的轮询队列、
//       MoE 里 expert 的轮询选择、多进程 DataLoader 的轮询 worker。
//
//   第 2 部分：自由表 —— 把「已删除的结点」回收进一个链表，下次需要结点时
//     直接复用，而不是每次 new/delete（系统分配器有锁、有元数据开销）。
//     * 这就是「对象池（object pool）」/「内存池（memory pool）」的最小原型。
//     * AI 关联：PyTorch 的 Caching Allocator 就是显存版的自由表 ——
//       释放的张量显存不还给驱动，而是挂进 free list，下次同尺寸分配直接复用，
//       从而把「分配/释放显存」从昂贵的 cudaMalloc 降为 O(1) 的链表操作。
//       训练循环里成千上万次中间张量的申请，正是靠它才不卡。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 03_circular_free_list.cpp -o circ && ./circ
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 03_circular_free_list.cpp -o circ_san && ./circ_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

// ===========================================================================
// 1. 循环单向链表 + 约瑟夫环
// ===========================================================================
struct CNode {
    int    val;
    CNode* next;
    explicit CNode(int v, CNode* n = nullptr) : val(v), next(n) {}
};

// 用 1..n 建一条循环链，返回指向「1」的指针
static CNode* buildCircle(int n) {
    CNode* head = new CNode(1);
    CNode* tail = head;
    for (int i = 2; i <= n; ++i) { tail->next = new CNode(i); tail = tail->next; }
    tail->next = head;                 // 关键：尾巴指回头，成环
    return head;
}

// 释放整条循环链：先断环，再按普通链表删
static void destroyCircle(CNode* head) {
    if (!head) return;
    CNode* p = head;
    while (p->next != head) p = p->next;   // 找到尾
    p->next = nullptr;                     // 断环
    while (head) { CNode* n = head->next; delete head; head = n; }
}

// 约瑟夫环：n 个人围成一圈，从 1 开始报数，报到第 k 个的人出局，
// 然后从下一个人重新报 1，直到剩最后一人。返回幸存者编号。
// 模拟复杂度 O(n·k)（每轮走 k 步），直观但不最快；
// 数学法（递推 J(n) = (J(n-1)+k) mod n）可 O(n)，见练习。
static int josephus(int n, int k) {
    if (n <= 0) return -1;
    CNode* cur = buildCircle(n);
    CNode* prev = cur;
    while (prev->next != cur) prev = prev->next;   // prev 始终是 cur 的前驱

    while (cur->next != cur) {                     // 只剩一个结点时停
        for (int i = 1; i < k; ++i) {              // 报数 1..k-1，cur 前进
            prev = cur;
            cur  = cur->next;
        }
        // 出局的是 cur
        prev->next = cur->next;
        delete cur;
        cur = prev->next;                          // 从下一个人重新开始
    }
    const int survivor = cur->val;
    delete cur;
    return survivor;
}

static void part1_josephus() {
    std::cout << "=== 1. 循环链表 + 约瑟夫环 ===\n";
    struct Case { int n, k, expect; };
    // 手算校验：n=5,k=2 -> 出局顺序 2,4,1,5 -> 幸存 3
    //          n=7,k=3 -> 幸存 4
    //          n=1,k=9 -> 幸存 1
    const Case cases[] = {{5, 2, 3}, {7, 3, 4}, {1, 9, 1}, {10, 3, 4}};
    for (const Case& c : cases) {
        const int s = josephus(c.n, c.k);
        std::cout << "  n=" << c.n << ", k=" << c.k
                  << " -> 幸存者 = " << s << "（期望 " << c.expect << "）\n";
        assert(s == c.expect);
    }
    // 校验 n=5,k=2 的出局顺序
    {
        CNode* cur = buildCircle(5);
        CNode* prev = cur;
        while (prev->next != cur) prev = prev->next;
        std::vector<int> out;
        while (cur->next != cur) {
            for (int i = 1; i < 2; ++i) { prev = cur; cur = cur->next; }
            out.push_back(cur->val);
            prev->next = cur->next; delete cur; cur = prev->next;
        }
        assert((out == std::vector<int>{2, 4, 1, 5}));
        std::cout << "  n=5,k=2 出局顺序 = 2 4 1 5（手算一致）✓\n";
        delete cur;
    }

    // 演示：循环链表「转一圈回到起点」；以及删除前必须先断环
    {
        CNode* c = buildCircle(4);
        std::cout << "  循环链 1->2->3->4->(回 1)，连走 6 步: ";
        CNode* p = c;
        for (int i = 0; i < 6; ++i) { std::cout << p->val << " "; p = p->next; }
        std::cout << "\n";
        // 若按「普通链表」那样 `while (head) { ... head = head->next; }` 去删，
        // 会因为尾巴指回头而不停绕圈、永不结束 —— 所以必须先断环。
        destroyCircle(c);
    }
    std::cout << "\n";
}

// ===========================================================================
// 2. 自由表（对象池）—— 内存池的最小原型
// ===========================================================================
//
// 设计：预先/按需分配一批结点，用一条单链表把它们串起来。
//   * allocate()：若自由表非空，摘下头结点复用（O(1)，无系统分配）；
//                 若为空，才向系统 new 一个新结点。
//   * release(p)：把结点挂回自由表头（O(1)，无系统释放）。
// 关键在于「复用」：反复 allocate/release 时，系统 new 的次数不会无限增长。
struct Slot {
    int   value;
    Slot* next;                       // 复用为自由表的链
};

class FreeList {
public:
    ~FreeList() {
        while (free_) { Slot* n = free_->next; delete free_; free_ = n; }
    }

    Slot* allocate(int v) {
        if (free_) {                  // 命中缓存：直接复用
            Slot* s = free_;
            free_ = free_->next;
            s->value = v;
            ++reused_;
            return s;
        }
        ++systemAllocs_;              // 未命中：才真正找系统要内存
        return new Slot{v, nullptr};
    }

    void release(Slot* s) {
        s->next = free_;              // 挂回自由表头
        free_ = s;
    }

    std::size_t freeCount()   const { return countFree(); }
    std::size_t systemAllocs() const { return systemAllocs_; }
    std::size_t reused()      const { return reused_; }

private:
    std::size_t countFree() const {
        std::size_t n = 0;
        for (Slot* p = free_; p; p = p->next) ++n;
        return n;
    }
    Slot*       free_         = nullptr;
    std::size_t systemAllocs_ = 0;
    std::size_t reused_       = 0;
};

static void part2_freelist() {
    std::cout << "=== 2. 自由表（对象池）：把 new/delete 降为链表操作 ===\n";
    FreeList pool;

    // 申请 100 个，再全部释放 -> 自由表应有 100 个可复用槽
    std::vector<Slot*> a;
    for (int i = 0; i < 100; ++i) a.push_back(pool.allocate(i));
    assert(pool.systemAllocs() == 100 && pool.freeCount() == 0);
    for (Slot* s : a) pool.release(s);
    std::cout << "  申请100+释放100: 系统分配 = " << pool.systemAllocs()
              << ", 自由表 = " << pool.freeCount() << "\n";
    assert(pool.freeCount() == 100);

    // 再申请 100 个：应全部复用，系统分配次数不再增长
    std::vector<Slot*> b;
    for (int i = 0; i < 100; ++i) b.push_back(pool.allocate(1000 + i));
    std::cout << "  再申请100:      系统分配 = " << pool.systemAllocs()
              << ", 复用次数 = " << pool.reused()
              << ", 自由表 = " << pool.freeCount() << "\n";
    assert(pool.systemAllocs() == 100);        // 没有新增系统分配
    assert(pool.reused() == 100);              // 100 次全部复用
    assert(b[0]->value == 1000 && b[99]->value == 1099);
    for (Slot* s : b) pool.release(s);
    std::cout << "  → 100 次申请全部命中复用，未再向系统要内存 ✓\n";
    std::cout << "  这就是 PyTorch Caching Allocator 的核心思想：\n"
                 "    张量显存释放时不还驱动，而是挂进 free list；\n"
                 "    下次同尺寸申请直接复用 -> cudaMalloc 调用次数骤降。\n";
}

int main() {
    std::cout << "======== 03 循环链表 与 自由表 ========\n\n";
    part1_josephus();
    part2_freelist();
    std::cout << "\n全部断言通过，程序正常结束。\n";
    return 0;
}
