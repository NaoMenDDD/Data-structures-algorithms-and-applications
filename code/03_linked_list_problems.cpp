// 03_linked_list_problems.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 03 章 链表》
// 主题：链表的经典算法题与工程结构
//
//   题目清单（每题都标注了复杂度与「考点」）：
//     1. 就地反转（迭代三指针 / 递归）             O(n) / O(1)   —— 指针基本功
//     2. 快慢指针找中点                             O(n) / O(1)   —— Floyd 慢快指针
//     3. 判环 + 找环入口（Floyd 判圈）             O(n) / O(1)   —— 数学推导
//     4. 返回倒数第 k 个结点                        O(n) / O(1)   —— 双指针间隔
//     5. 合并两条有序链表                            O(n+m)/O(1)   —— 归并的链表版
//     6. 删除倒数第 n 个结点                        O(n) / O(1)   —— 哑结点消除边界
//     7. 回文判定                                   O(n) / O(1)   —— 中点+反转+比较
//     8. 两条链表是否相交、交点在哪                 O(n+m)/O(1)   —— 长度差对齐
//     9. LRU 缓存（哈希表 + 双向链表）             get/put O(1)  —— ★ 工程核心
//
//   第 9 题是本章与 AI 世界模型的直接接口：
//     推理服务里的 KV cache、嵌入表缓存、特征存储，都靠「LRU + 哈希」在有限显存/
//     内存里保留最近被用到的条目。它也是「O(1) 淘汰一个已知元素」必须用双向链表的
//     最好例子 —— 单链表做不到 O(1) 删除任意已知结点。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 03_linked_list_problems.cpp -o llp && ./llp
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 03_linked_list_problems.cpp -o llp_san && ./llp_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// 供这些问题使用的极简单链表结点 + 小工具
// ---------------------------------------------------------------------------
struct Node {
    int   val;
    Node* next;
    explicit Node(int v, Node* n = nullptr) : val(v), next(n) {}
};

// 用 initializer_list 建一条链表，返回头指针
static Node* build(std::initializer_list<int> xs) {
    Node* head = nullptr;
    Node** tail = &head;
    for (int v : xs) { *tail = new Node(v); tail = &(*tail)->next; }
    return head;
}

// 把链表转成 vector（便于断言）
static std::vector<int> toVec(Node* head) {
    std::vector<int> v;
    for (Node* p = head; p; p = p->next) v.push_back(p->val);
    return v;
}

// 释放整条链表；把传进来的指针引用置空，避免悬垂
static void destroy(Node*& head) {
    while (head) { Node* n = head->next; delete head; head = n; }
}

// ===========================================================================
// 1. 就地反转
// ===========================================================================
static Node* reverseIter(Node* head) {
    Node* prev = nullptr;
    while (head) {
        Node* nxt = head->next;   // 先存后继
        head->next = prev;        // 掉头
        prev = head;              // 前进
        head = nxt;
    }
    return prev;                  // 新的头
}

// 递归写法：语义「反转以 head 开头的链，返回新头」。注意递归深度 = 链长。
static Node* reverseRec(Node* head) {
    if (!head || !head->next) return head;      // 空或单结点：已反转
    Node* newHead = reverseRec(head->next);
    head->next->next = head;                    // 让「后继」反过来指向自己
    head->next = nullptr;                       // 自己成为尾
    return newHead;
}

static void prob_reverse() {
    std::cout << "\n=== 1. 就地反转 ===\n";
    Node* a = build({1, 2, 3, 4, 5});
    a = reverseIter(a);
    assert((toVec(a) == std::vector<int>{5, 4, 3, 2, 1}));
    std::cout << "迭代反转: "; for (int x : toVec(a)) std::cout << x << " "; std::cout << "\n";
    destroy(a);

    Node* b = build({1, 2, 3, 4, 5});
    b = reverseRec(b);
    assert((toVec(b) == std::vector<int>{5, 4, 3, 2, 1}));
    std::cout << "递归反转: "; for (int x : toVec(b)) std::cout << x << " "; std::cout << "\n";
    destroy(b);

    Node* e = nullptr;                  // 空表
    e = reverseIter(e);
    assert(e == nullptr);
    Node* s = build({9});               // 单结点
    s = reverseIter(s);
    assert(s && s->val == 9 && s->next == nullptr);
    destroy(s);
    std::cout << "空表/单结点边界 ✓\n";
}

// ===========================================================================
// 2. 快慢指针找中点：fast 每次走 2 步，slow 每次走 1 步
//    fast 到头时 slow 恰在中点。偶数长度时返回「偏前」的中点。
// ===========================================================================
static Node* middleNode(Node* head) {
    Node* slow = head;
    Node* fast = head;
    while (fast && fast->next) { slow = slow->next; fast = fast->next->next; }
    return slow;
}

static void prob_middle() {
    std::cout << "\n=== 2. 快慢指针找中点 ===\n";
    Node* a = build({1, 2, 3, 4, 5});
    assert(middleNode(a)->val == 3);
    Node* b = build({1, 2, 3, 4});
    assert(middleNode(b)->val == 3);      // 偶数取偏前
    std::cout << "奇数长度中点 = 3，偶数长度中点(偏前) = 3 ✓\n";
    destroy(a); destroy(b);
}

// ===========================================================================
// 3. Floyd 判圈：slow 走 1 步、fast 走 2 步。
//    * 有环则二者必在环内相遇（fast 每步追近 slow 1 格，不会跳过）。
//    * 找环入口：相遇后把一个指针放回头，两指针同速前进，再次相遇处即入口。
//      推导：设头到入口长 a，入口到相遇点长 b，环长 L。相遇时 slow 走了 a+b，
//      fast 走了 a+b+kL（多转了整数圈），又 fast = 2·slow（速度关系在相遇点成立），
//      于是 2(a+b) = a+b+kL → a+b = kL → a = kL - b = (k-1)L + (L-b)。
//      即「从头走到入口的距离 a」等于「从相遇点继续走到入口的距离 L-b」。
// ===========================================================================
static bool detectCycle(Node* head, Node** entryOut = nullptr) {
    Node* slow = head;
    Node* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {                     // 相遇 → 有环
            if (entryOut) {
                Node* p = head;
                while (p != slow) { p = p->next; slow = slow->next; }
                *entryOut = p;                  // 再遇处即入口
            }
            return true;
        }
    }
    return false;
}

static void prob_cycle() {
    std::cout << "\n=== 3. Floyd 判环 + 找入口 ===\n";
    Node* a = build({1, 2, 3, 4, 5});
    assert(!detectCycle(a));
    std::cout << "无环链: 判无环 ✓\n";
    destroy(a);

    Node* b = build({1, 2, 3, 4, 5});
    Node* e2 = b->next->next;            // 让 5 指回 3，形成环
    Node* tail = b; while (tail->next) tail = tail->next;
    tail->next = e2;
    Node* entry = nullptr;
    assert(detectCycle(b, &entry));
    assert(entry == e2 && entry->val == 3);
    std::cout << "有环链(5->3): 判有环 ✓，入口值 = " << entry->val << "\n";
    tail->next = nullptr;                // 先断环，才能安全 delete
    destroy(b);
}

// ===========================================================================
// 4. 倒数第 k 个：让 first 先走 k 步，再 first/second 同步走，
//    first 到空时 second 恰在倒数第 k 个。
// ===========================================================================
static Node* kthFromEnd(Node* head, int k) {
    Node* first = head;
    for (int i = 0; i < k; ++i) { if (!first) return nullptr; first = first->next; }
    Node* second = head;
    while (first) { first = first->next; second = second->next; }
    return second;
}

static void prob_kth_from_end() {
    std::cout << "\n=== 4. 倒数第 k 个 ===\n";
    Node* a = build({1, 2, 3, 4, 5});
    assert(kthFromEnd(a, 1)->val == 5);      // 倒数第 1 = 尾
    assert(kthFromEnd(a, 3)->val == 3);
    assert(kthFromEnd(a, 5)->val == 1);      // 倒数第 5 = 头
    assert(kthFromEnd(a, 6) == nullptr);     // 越界
    std::cout << "倒数 1/3/5 分别 = 5/3/1，越界返回 nullptr ✓\n";
    destroy(a);
}

// ===========================================================================
// 5. 合并两条有序链表：经典「哨兵 + 尾指针」写法，无需为首元素单开分支。
// ===========================================================================
static Node* mergeSorted(Node* a, Node* b) {
    Node dummy(0);
    Node* tail = &dummy;
    while (a && b) {
        if (a->val <= b->val) { tail->next = a; a = a->next; }
        else                  { tail->next = b; b = b->next; }
        tail = tail->next;
    }
    tail->next = a ? a : b;                 // 接上剩余部分
    return dummy.next;
}

static void prob_merge() {
    std::cout << "\n=== 5. 合并有序链表 ===\n";
    Node* a = build({1, 3, 5, 7});
    Node* b = build({2, 4, 6});
    Node* m = mergeSorted(a, b);
    assert((toVec(m) == std::vector<int>{1, 2, 3, 4, 5, 6, 7}));
    std::cout << "归并: "; for (int x : toVec(m)) std::cout << x << " "; std::cout << "\n";
    destroy(m);                              // a、b 的结点已被并入 m，只删 m 即可
}

// ===========================================================================
// 6. 删除倒数第 n 个：哑结点让「删头」不再特殊。
// ===========================================================================
static Node* removeNthFromEnd(Node* head, int n) {
    Node dummy(0, head);
    Node* fast = &dummy;
    for (int i = 0; i < n; ++i) fast = fast->next;   // 先走 n 步
    Node* slow = &dummy;
    while (fast->next) { fast = fast->next; slow = slow->next; }
    Node* doomed = slow->next;
    slow->next = doomed->next;               // 跨过被删结点
    delete doomed;
    return dummy.next;
}

static void prob_remove_nth() {
    std::cout << "\n=== 6. 删除倒数第 n 个 ===\n";
    Node* a = build({1, 2, 3, 4, 5});
    a = removeNthFromEnd(a, 2);
    assert((toVec(a) == std::vector<int>{1, 2, 3, 5}));
    std::cout << "删倒数第 2: "; for (int x : toVec(a)) std::cout << x << " "; std::cout << "\n";
    a = removeNthFromEnd(a, 4);              // 删头
    assert((toVec(a) == std::vector<int>{2, 3, 5}));
    std::cout << "再删倒数第 4(即头): "; for (int x : toVec(a)) std::cout << x << " "; std::cout << "\n";
    destroy(a);
}

// ===========================================================================
// 7. 回文判定：找中点 → 反转后半 → 逐一比较。O(n) 时间、O(1) 空间。
// ===========================================================================
static bool isPalindrome(Node* head) {
    if (!head || !head->next) return true;
    Node* slow = head;
    Node* fast = head;
    while (fast->next && fast->next->next) { slow = slow->next; fast = fast->next->next; }
    // 此时 slow 是前半的最后一个；反转 slow->next 起
    Node* second = reverseIter(slow->next);
    Node* p = head;
    Node* q = second;
    bool ok = true;
    while (q) { if (p->val != q->val) { ok = false; break; } p = p->next; q = q->next; }
    reverseIter(second);                     // 恢复链表（好习惯）
    return ok;
}

static void prob_palindrome() {
    std::cout << "\n=== 7. 回文判定 ===\n";
    Node* a = build({1, 2, 3, 2, 1});
    Node* b = build({1, 2, 3, 3, 2, 1});
    Node* c = build({1, 2, 3, 4});
    assert(isPalindrome(a) && isPalindrome(b) && !isPalindrome(c));
    std::cout << "1-2-3-2-1 ✓回文, 1-2-3-3-2-1 ✓回文, 1-2-3-4 ✗非回文\n";
    destroy(a); destroy(b); destroy(c);
}

// ===========================================================================
// 8. 相交链表：先各算长度，让长的先走长度差，再同速走，第一个相同结点即交点。
// ===========================================================================
static Node* intersection(Node* a, Node* b) {
    int la = 0, lb = 0;
    for (Node* p = a; p; p = p->next) ++la;
    for (Node* p = b; p; p = p->next) ++lb;
    while (la > lb) { a = a->next; --la; }
    while (lb > la) { b = b->next; --lb; }
    while (a != b) { a = a->next; b = b->next; }
    return a;                                // 可能为 nullptr（不相交）
}

// 相交演示：两条链在 shared 处汇合，必须「先摘断前缀与共享段的耦合」，
// 再分别释放前缀、最后只删共享段一次 —— 否则会重复释放 shared。
static void prob_intersection_safe() {
    std::cout << "\n=== 8. 相交链表（长度差对齐）===\n";
    Node* shared = build({7, 8, 9});
    Node* a = new Node(1, new Node(2, shared));
    Node* b = new Node(10, new Node(20, new Node(30, shared)));
    Node* x = intersection(a, b);
    assert(x == shared);
    const int xv = x->val;                         // 先取值，随后就要释放这些结点了
    // 摘断前缀与 shared 的耦合，再分别释放
    Node* a1 = a;            Node* a2 = a->next;   // 1,2
    Node* b1 = b;            Node* b2 = b->next;
    Node* b3 = b->next->next;                      // 30
    a2->next = nullptr;                            // 断开 2 -> shared
    b3->next = nullptr;                            // 断开 30 -> shared
    delete a1; delete a2;
    delete b1; delete b2; delete b3;
    destroy(shared);                               // 只删共享段一次
    std::cout << "交点值 = " << xv << "（应为共享尾巴首结点 7），"
                 "前缀与共享段均正确释放 ✓\n";
}

// ===========================================================================
// 9. LRU 缓存：哈希表(定位 O(1)) + 双向链表(维护访问顺序，淘汰 O(1))
//    * get(key)：命中则把该结点 splice 到最前（最近使用），返回值。
//    * put(key,value)：已存在则改值并提到最前；否则插到最前，超额则淘汰最尾。
//    std::list 的 splice 是 O(1)（只改指针、不搬元素）——正是双向链表的价值。
// ===========================================================================
class LRUCache {
public:
    explicit LRUCache(std::size_t capacity) : cap_(capacity) {}

    int get(int key) {
        auto it = pos_.find(key);
        if (it == pos_.end()) return -1;             // 未命中
        // 提到最前：把 it 指向的结点 splice 到 begin() 之前（即成为新 begin）
        items_.splice(items_.begin(), items_, it->second);
        return it->second->second;
    }

    void put(int key, int value) {
        auto it = pos_.find(key);
        if (it != pos_.end()) {                      // 已存在：更新值并提前
            it->second->second = value;
            items_.splice(items_.begin(), items_, it->second);
            return;
        }
        items_.emplace_front(key, value);            // 新键插到最前
        pos_[key] = items_.begin();
        if (items_.size() > cap_) {                  // 超额：淘汰最久未用（尾部）
            pos_.erase(items_.back().first);
            items_.pop_back();
        }
    }

    std::size_t size() const { return items_.size(); }

    void dump(std::ostream& os) const {
        os << "MRU [";
        bool first = true;
        for (const auto& kv : items_) {
            if (!first) os << ", ";
            os << kv.first << ":" << kv.second;
            first = false;
        }
        os << "] LRU";
    }

private:
    std::size_t cap_;
    std::list<std::pair<int, int>> items_;                              // 前=最近使用
    std::unordered_map<int, std::list<std::pair<int, int>>::iterator> pos_;
};

static void prob_lru() {
    std::cout << "\n=== 9. LRU 缓存（哈希 + 双向链表，get/put 均 O(1)）===\n";
    LRUCache c(2);

    c.put(1, 100);
    c.put(2, 200);
    std::cout << "put(1,100).put(2,200): "; c.dump(std::cout); std::cout << "\n";

    assert(c.get(1) == 100);                  // 命中 1 → 1 变最近
    std::cout << "get(1)=100 之后:        "; c.dump(std::cout); std::cout << "\n";

    c.put(3, 300);                            // 容量满 → 淘汰最久未用的 2
    std::cout << "put(3,300) 触发淘汰:    "; c.dump(std::cout); std::cout << "\n";
    assert(c.get(2) == -1);                   // 2 已被淘汰
    assert(c.get(1) == 100 && c.get(3) == 300);

    c.put(1, 111);                            // 更新已存在的键
    assert(c.get(1) == 111);
    std::cout << "put(1,111) 更新后:      "; c.dump(std::cout); std::cout << "\n";

    c.put(4, 400);                            // 淘汰最久未用的 3
    assert(c.get(3) == -1);
    std::cout << "put(4,400) 后 get(3)=-1: "; c.dump(std::cout); std::cout << "\n";
    std::cout << "LRU 行为全部正确 ✓（这正是 KV cache / 嵌入表缓存的淘汰策略）\n";
}

int main() {
    std::cout << "======== 03 链表经典问题 ========";
    prob_reverse();
    prob_middle();
    prob_cycle();
    prob_kth_from_end();
    prob_merge();
    prob_remove_nth();
    prob_palindrome();
    prob_intersection_safe();
    prob_lru();
    std::cout << "\n全部断言通过，程序正常结束。\n";
    return 0;
}
