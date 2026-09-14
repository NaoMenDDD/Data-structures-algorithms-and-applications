// 09_skiplist.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 09 章 搜索树》
// 主题：跳表（Skip List）—— 用「抛硬币」代替旋转的概率型有序结构
//
//   问题：链表插入/删除 O(1)（已知位置时），但查找 O(n)；BST/AVL 查找 O(log n)
//   但要平衡维护。跳表（Pugh, 1989）给出第三条路：
//
//     在有序链表上叠「快速通道」（express lane）：每个结点以概率 p（通常 1/2）
//     决定是否「长到上一层」。层数期望 O(log n)，于是查找像坐电梯——
//     先在高层的稀疏链表里大步跳，跳不动了再下一层。**期望** O(log n)。
//
//   为什么是概率的？因为「有没有上一层」是抛硬币定的，整棵树是**随机的**，
//   不需要任何旋转/再平衡。只要随机种子好，就极难退化（比确定性平衡树还好实现）。
//
//   关键操作：
//     * search：从最高层的 head 出发，能往右就往右，不能就下移，直到最底层；
//     * insert：先记下「一路上要接在哪」（update[]），逐层插入，直到硬币说停；
//     * delete：同样记 update[]，逐层把目标结点摘掉。
//
//   它是 Redis 的 zset、LevelDB/RocksDB 的 MemTable 的底层结构 —— 工程上比
//   红黑树更好写、并发更友好（AI 训练里的参数服务器、键值缓存常用）。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 09_skiplist.cpp -o 09_skiplist && ./09_skiplist
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 09_skiplist.cpp -o sl_san && ./sl_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cmath>          // std::log2
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <vector>

static constexpr int MAX_LEVEL = 32;          // 支持约 2^32 个元素
static constexpr double P = 0.5;              // 晋升概率

struct Node {
    int   key;
    // next[i] = 在第 i 层上，本结点的后继。用 vector 按需分配，避免 32 个指针浪费。
    std::vector<Node*> next;
    explicit Node(int k, int level) : key(k), next(static_cast<std::size_t>(level) + 1, nullptr) {}
};

class SkipList {
public:
    SkipList() : head_(new Node(-1, MAX_LEVEL - 1)), level_(0), size_(0) {}
    ~SkipList() {
        Node* cur = head_->next[0];
        while (cur) { Node* nxt = cur->next[0]; delete cur; cur = nxt; }
        delete head_;
    }

    bool search(int key) const {
        Node* x = head_;
        for (int i = level_; i >= 0; --i) {                 // 从当前最高层往下
            while (x->next[static_cast<std::size_t>(i)] &&
                   x->next[static_cast<std::size_t>(i)]->key < key)
                x = x->next[static_cast<std::size_t>(i)];   // 本层能往右就往右
        }
        x = x->next[0];                                     // 落到最底层再看一步
        return x && x->key == key;
    }

    void insert(int key) {
        std::vector<Node*> update(static_cast<std::size_t>(MAX_LEVEL), nullptr);
        Node* x = head_;
        for (int i = level_; i >= 0; --i) {
            while (x->next[static_cast<std::size_t>(i)] &&
                   x->next[static_cast<std::size_t>(i)]->key < key)
                x = x->next[static_cast<std::size_t>(i)];
            update[static_cast<std::size_t>(i)] = x;        // 记下每层的前驱
        }
        x = x->next[0];
        if (x && x->key == key) return;                     // 已存在，忽略（集合语义）

        const int lvl = randomLevel();
        if (lvl > level_) {                                 // 新层：更高层的前驱都是 head
            for (int i = level_ + 1; i <= lvl; ++i)
                update[static_cast<std::size_t>(i)] = head_;
            level_ = lvl;
        }
        Node* node = new Node(key, lvl);
        for (int i = 0; i <= lvl; ++i) {                    // 逐层穿针引线
            node->next[static_cast<std::size_t>(i)] = update[static_cast<std::size_t>(i)]->next[static_cast<std::size_t>(i)];
            update[static_cast<std::size_t>(i)]->next[static_cast<std::size_t>(i)] = node;
        }
        ++size_;
    }

    bool remove(int key) {
        std::vector<Node*> update(static_cast<std::size_t>(MAX_LEVEL), nullptr);
        Node* x = head_;
        for (int i = level_; i >= 0; --i) {
            while (x->next[static_cast<std::size_t>(i)] &&
                   x->next[static_cast<std::size_t>(i)]->key < key)
                x = x->next[static_cast<std::size_t>(i)];
            update[static_cast<std::size_t>(i)] = x;
        }
        x = x->next[0];
        if (!x || x->key != key) return false;

        for (int i = 0; i <= level_; ++i) {                 // 从每层把 x 摘掉
            if (update[static_cast<std::size_t>(i)]->next[static_cast<std::size_t>(i)] != x) break;
            update[static_cast<std::size_t>(i)]->next[static_cast<std::size_t>(i)] = x->next[static_cast<std::size_t>(i)];
        }
        delete x;
        while (level_ > 0 && !head_->next[static_cast<std::size_t>(level_)]) --level_;  // 收缩空层
        --size_;
        return true;
    }

    std::size_t size() const { return size_; }
    int level() const { return level_; }

    // 与 search 相同，但统计「从 head 到命中点」访问过的结点数（用于教学对比）。
    Node* search_probe(int key, int* steps) const {
        int s = 0;
        Node* x = head_;
        for (int i = level_; i >= 0; --i) {
            while (x->next[static_cast<std::size_t>(i)] &&
                   x->next[static_cast<std::size_t>(i)]->key < key) {
                x = x->next[static_cast<std::size_t>(i)];
                ++s;
            }
        }
        x = x->next[0];
        if (x) ++s;
        if (steps) *steps = s;
        return x;
    }

    std::vector<int> toVector() const {                  // 最底层即全体有序
        std::vector<int> out;
        for (Node* x = head_->next[0]; x; x = x->next[0]) out.push_back(x->key);
        return out;
    }

private:
    int randomLevel() const {
        int lvl = 0;
        while (lvl < MAX_LEVEL - 1 && rng_() % 2 == 0) ++lvl;   // 抛硬币：正面就升一层
        return lvl;
    }

    Node* head_;
    int   level_;                                        // 当前最高非空层
    std::size_t size_;
    mutable std::mt19937 rng_{20260913};
};

int main() {
    std::cout << "======== 09 跳表（Skip List）========\n\n";

    // ---------- 基本操作 ----------
    std::cout << "=== 1. 插入 / 查找 / 删除 ===\n";
    {
        SkipList sl;
        for (int k : {3, 6, 7, 9, 12, 19, 17, 26, 21, 25}) sl.insert(k);
        sl.insert(7);                                     // 重复应忽略
        std::cout << "  当前元素: ";
        for (int k : sl.toVector()) std::cout << k << " ";
        std::cout << "\n";
        assert(sl.size() == 10);
        assert(sl.search(19) && sl.search(3) && sl.search(25) && !sl.search(4));
        std::cout << "  查找 19 = 1，查找 4 = 0，size = " << sl.size() << " ✓\n";

        assert(sl.remove(19) && !sl.search(19));
        assert(!sl.remove(19));                           // 再删应失败
        std::cout << "  删除 19 后：存在? " << sl.search(19)
                  << "，再删一次返回 " << sl.remove(19) << " ✓\n";
    }

    // ---------- 与 std::set 随机对拍 ----------
    std::cout << "\n=== 2. 与 std::set 随机对拍 ===\n";
    {
        std::mt19937 rng(20260913);
        for (int trial = 0; trial < 100; ++trial) {
            SkipList sl;
            std::set<int> s;
            for (int op = 0; op < 400; ++op) {
                const int k = static_cast<int>(rng() % 300);
                if (rng() % 3 != 0) { sl.insert(k); s.insert(k); }
                else                { const bool a = sl.remove(k), b = (s.erase(k) > 0); assert(a == b); }
            }
            assert(sl.toVector() == std::vector<int>(s.begin(), s.end()));
            for (int k = 0; k < 300; ++k) assert(sl.search(k) == (s.count(k) > 0));
        }
        std::cout << "  100 轮 × 400 次随机增删：有序序列 == std::set ✓\n";
    }

    // ---------- 对比：跳表 vs 有序链表 的查找步数 ----------
    std::cout << "\n=== 3. ★ 有「快速通道」的跳表 vs 纯链表 ===\n";
    {
        const int N = 100000;
        SkipList sl;
        for (int i = 0; i < N; ++i) sl.insert(i);         // 有序插入（对链表最坏！）
        std::cout << "  插入 0.." << (N - 1) << " 后跳表层数 = " << sl.level()
                  << "（≈ log2(" << N << ") = " << std::log2(double(N)) << "）\n";

        // 统计跳表查找的平均「比较次数」：用 search 成功命中点计数
        // 这里用一个更直观的指标：查找 N-1（最坏点）时，从 head 到目标经过的结点数。
        // 纯链表要走 N 步；跳表大约 log n / p 步。
        // 我们用「到达目标前访问的结点数」衡量。
        std::mt19937 rng(7);
        long long skipSteps = 0, listSteps = 0;
        const int Q = 2000;
        for (int q = 0; q < Q; ++q) {
            const int key = static_cast<int>(rng() % N);
            // 跳表：模拟查找路径长度
            int steps = 0;
            Node* x = sl.search_probe(key, &steps);       // 见下方辅助方法
            (void)x;
            skipSteps += steps;
            listSteps += key + 1;                          // 纯链表：从 0 走到 key 需要 key+1 步
        }
        std::cout << "  随机查 " << Q << " 次（键域 0.." << (N - 1) << "）：\n";
        std::cout << "     纯有序链表 平均步数 ≈ " << (double(listSteps) / Q) << "\n";
        std::cout << "     跳表       平均步数 ≈ " << (double(skipSteps) / Q) << "\n";
        std::cout << "     加速 ≈ " << (double(listSteps) / double(skipSteps)) << "x\n";
    }

    std::cout << "\n=== 4. 为什么工程上爱用跳表 ===\n"
                 "  * 实现远比红黑树简单：没有旋转，只有「抛硬币 + 逐层穿针」；\n"
                 "  * 支持**范围查询**：找到 lo 后沿最底层顺序走即可得 [lo, hi]；\n"
                 "  * 并发友好：各层指针可做无锁 CAS（Redis、LevelDB 都是这么用的）；\n"
                 "  * 期望 O(log n)，但不保证最坏 —— 用足够好的随机数即可，工程上够用；\n"
                 "  * AI 落点：特征索引、近邻检索的候选集、训练用 KV 存储的 MemTable。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
