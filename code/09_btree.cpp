// 09_btree.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 09 章 搜索树》
// 主题：B 树 / B+ 树 —— 为「磁盘 / 大块存取」而生的多路搜索树
//
//   前面 BST / AVL / 跳表都是**二叉**的：每个结点存 1 个键、最多 2 个孩子。
//   但数据库 / 文件系统的数据在**磁盘块**上，一次 I/O 读一整块（如 4KB~16KB）。
//   二叉树每层只利用一个键，I/O 次数 = 树高 = O(log2 n)，常数大。
//
//   B 树（Bayer & McCreight, 1972）的思想：把结点做「胖」——
//     一个结点存**几百个键**、几百个孩子，让它恰好塞满一个磁盘块。
//   于是树高从 log2(n) 降到 log_{几百}(n)：10 亿条记录只需 3~4 层，
//   查一条记录只需 3~4 次磁盘 I/O。这就是 MySQL InnoDB、文件系统、
//   LevelDB SSTable 全都用 B 树 / B+ 树的原因。
//
//   本文件实现的是最小度数为 t 的 B 树（CLRS 记号）：
//     * 每个结点最多 2t-1 个键、2t 个孩子；非根结点至少 t-1 个键；
//     * 所有叶子在同一层（B 树是**完美平衡**的，靠「向上分裂」保证）。
//
//   插入用 CLRS 的「预分裂（preemptive split）」：一路向下时，
//   遇见满结点（2t-1 个键）就先把它裂成两半、把中间键顶进父结点，
//   于是真正落笔插入时，叶子一定不满，**永远不用回溯**。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 09_btree.cpp -o 09_btree && ./09_btree
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 09_btree.cpp -o bt_san && ./bt_san
// ---------------------------------------------------------------------------

#include <algorithm>      // std::max
#include <cassert>
#include <cmath>          // std::log
#include <cstddef>
#include <iostream>
#include <random>
#include <set>
#include <vector>

// ---------------------------------------------------------------------------
// B 树结点：n 个键 + n+1 个孩子（叶子时 children 为空）
// ---------------------------------------------------------------------------
struct BNode {
    bool        leaf;
    std::vector<int>     keys;
    std::vector<BNode*>  children;
    explicit BNode(bool isLeaf) : leaf(isLeaf) {}
};

class BTree {
public:
    explicit BTree(int t) : t_(t), root_(nullptr) {}
    ~BTree() { destroy(root_); }

    bool search(int key) const {
        BNode* x = root_;
        while (x) {
            std::size_t i = 0;
            while (i < x->keys.size() && key > x->keys[i]) ++i;   // 在结点内线性找
            if (i < x->keys.size() && key == x->keys[i]) return true;
            if (x->leaf) return false;
            x = x->children[i];                                    // 落到对应子树
        }
        return false;
    }

    void insert(int key) {
        if (!root_) { root_ = new BNode(true); root_->keys.push_back(key); return; }
        if (search(key)) return;                                  // 集合语义，忽略重复

        if (static_cast<int>(root_->keys.size()) == 2 * t_ - 1) {  // 根满了：先长高
            BNode* s = new BNode(false);
            s->children.push_back(root_);
            splitChild(s, 0);
            root_ = s;
        }
        insertNonFull(root_, key);
    }

    std::size_t size() const { return size_; }
    int height() const { return heightOf(root_); }

    void inorder(std::vector<int>& out) const { inorder(root_, out); }

    // 供演示：统计所有结点的键数分布，验证「非根结点 >= t-1 个键」不变式
    void collectKeyCounts(std::vector<std::size_t>& counts) const { collect(root_, counts); }

private:
    // 把 x 的第 i 个孩子（满结点）裂成两半，中间键升到 x 里
    void splitChild(BNode* x, std::size_t i) {
        BNode* y = x->children[i];                    // 待裂的满孩子
        BNode* z = new BNode(y->leaf);
        const int mid = t_ - 1;

        z->keys.assign(y->keys.begin() + mid + 1, y->keys.end());      // y 的右半给 z
        if (!y->leaf)
            z->children.assign(y->children.begin() + mid + 1, y->children.end());

        const int upKey = y->keys[mid];
        y->keys.resize(static_cast<std::size_t>(mid));                 // y 保留左半
        if (!y->leaf) y->children.resize(static_cast<std::size_t>(mid + 1));

        x->keys.insert(x->keys.begin() + static_cast<long>(i), upKey); // 中间键升到父
        x->children.insert(x->children.begin() + static_cast<long>(i) + 1, z);
    }

    // 保证调用时 x 不满（< 2t-1 个键）
    void insertNonFull(BNode* x, int key) {
        int i = static_cast<int>(x->keys.size()) - 1;
        if (x->leaf) {
            x->keys.push_back(0);
            while (i >= 0 && key < x->keys[static_cast<std::size_t>(i)]) {
                x->keys[static_cast<std::size_t>(i + 1)] = x->keys[static_cast<std::size_t>(i)];
                --i;
            }
            x->keys[static_cast<std::size_t>(i + 1)] = key;           // 插入有序位置
            ++size_;
        } else {
            while (i >= 0 && key < x->keys[static_cast<std::size_t>(i)]) --i;
            ++i;                                                      // 应下探的孩子下标
            if (static_cast<int>(x->children[static_cast<std::size_t>(i)]->keys.size()) == 2 * t_ - 1) {
                splitChild(x, static_cast<std::size_t>(i));           // 下探前先裂
                if (key > x->keys[static_cast<std::size_t>(i)]) ++i;  // 裂后可能要换边
            }
            insertNonFull(x->children[static_cast<std::size_t>(i)], key);
        }
    }

    void inorder(BNode* x, std::vector<int>& out) const {
        if (!x) return;
        for (std::size_t i = 0; i < x->keys.size(); ++i) {
            if (!x->leaf) inorder(x->children[i], out);
            out.push_back(x->keys[i]);
        }
        if (!x->leaf) inorder(x->children[x->keys.size()], out);
    }

    void collect(BNode* x, std::vector<std::size_t>& counts) const {
        if (!x) return;
        counts.push_back(x->keys.size());
        for (BNode* c : x->children) collect(c, counts);
    }

    int heightOf(BNode* x) const {                // 边数定义的树高（叶子 = 0）
        if (!x || x->leaf) return 0;
        return 1 + heightOf(x->children[0]);
    }

    static void destroy(BNode* x) {
        if (!x) return;
        for (BNode* c : x->children) destroy(c);
        delete x;
    }

    int    t_;
    BNode* root_ = nullptr;
    std::size_t size_ = 0;
};

int main() {
    std::cout << "======== 09 B 树（多路搜索树）========\n\n";

    const int t = 3;                              // 最小度数 t=3：每结点 2..5 个键

    // ---------- 基本操作 ----------
    std::cout << "=== 1. 插入 / 查找 / 中序（t = 3）===\n";
    {
        BTree bt(t);
        for (int k : {10, 20, 5, 6, 12, 30, 7, 17}) bt.insert(k);
        bt.insert(10);                            // 重复忽略
        std::vector<int> out;
        bt.inorder(out);
        std::cout << "  中序遍历（升序）: ";
        for (std::size_t i = 0; i < out.size(); ++i) std::cout << out[i] << (i + 1 < out.size() ? " " : "");
        std::cout << "\n";
        assert((out == std::vector<int>{5, 6, 7, 10, 12, 17, 20, 30}));
        assert(bt.search(17) && bt.search(5) && !bt.search(11));
        std::cout << "  查找 17 = 1，查找 11 = 0，size = " << bt.size()
                  << "，高度 = " << bt.height() << " ✓\n";
    }

    // ---------- 与 std::set 随机对拍 + 不变式校验 ----------
    std::cout << "\n=== 2. 与 std::set 随机对拍 + 校验 B 树不变式 ===\n";
    {
        std::mt19937 rng(20260913);
        for (int trial = 0; trial < 200; ++trial) {
            BTree bt(t);
            std::set<int> s;
            const int n = static_cast<int>(rng() % 2000);
            for (int i = 0; i < n; ++i) { const int k = static_cast<int>(rng() % 5000); bt.insert(k); s.insert(k); }

            std::vector<int> out; bt.inorder(out);
            assert(out == std::vector<int>(s.begin(), s.end()));       // 有序且与 set 一致
            for (int k : s) assert(bt.search(k));

            std::vector<std::size_t> counts; bt.collectKeyCounts(counts);
            for (std::size_t c : counts) assert(c <= static_cast<std::size_t>(2 * t - 1));  // 上界
            // 非根结点 >= t-1 个键：这里 counts[0] 是根，其余都该 >= t-1
            for (std::size_t i = 1; i < counts.size(); ++i) assert(counts[i] >= static_cast<std::size_t>(t - 1));
        }
        std::cout << "  200 组随机测试：中序 == std::set，且「2t-1 上界 / t-1 下界」全部满足 ✓\n";
    }

    // ---------- ★ B 树高度 vs 二叉树高度 ----------
    std::cout << "\n=== 3. ★ 为什么「胖结点」能大幅降低树高 ===\n";
    std::cout << "      N        B树(t=3)高   BST高(=N-1)   B树(t=8)高   理想二叉 log2(N)\n";
    for (int N : {1000, 100000, 1000000}) {
        BTree b3(3), b8(8);
        for (int i = 1; i <= N; ++i) { b3.insert(i); b8.insert(i); }   // 升序插入（BST 最坏）
        std::cout << "     " << N << "     " << b3.height() << "        " << (N - 1)
                  << "        " << b8.height() << "          " << std::log2(double(N)) << "\n";
        // B 树高度 ≈ log_t(N)：t=3 时约 log2(N)/log2(3)，t=8 时约 log2(N)/3
        assert(b3.height() <= static_cast<int>(std::log2(double(N)) / std::log2(3.0)) + 3);
        assert(b8.height() < b3.height());
    }
    std::cout << "  -> 升序插入对 BST 是灾难（高 N-1），对 B 树几乎无影响（永远完美平衡）。\n";
    std::cout << "     一个结点存 2t-1 个键，就把树高从 log2(n) 压到 log_{2t}(n)；\n";
    std::cout << "     t 取几百（刚好塞满一个磁盘块）时，10 亿条记录也只要 3~4 层。\n\n";

    // ---------- B 树 vs B+ 树 ----------
    std::cout << "=== 4. B 树 vs B+ 树（工程上的选择）===\n"
                 "  * B 树：数据键分散在所有结点，查找可能在内部结点命中，一次 I/O 搞定；\n"
                 "  * B+ 树：**只有叶子存数据**，内部结点只存「路由键」，叶子用链表串起来。\n"
                 "      - 内部结点更小 -> 同样一个磁盘块能装更多键 -> 树更矮 -> I/O 更少；\n"
                 "      - 叶子成链 -> **范围查询**只需顺序扫叶子，天然支持 `ORDER BY`；\n"
                 "      - 所以 MySQL InnoDB、PostgreSQL、Oracle 的索引都是 B+ 树。\n\n";
    std::cout << "=== 5. AI / 工程落点 ===\n"
                 "  * 向量数据库（Faiss / Milvus）的**元数据索引**、检索结果的键值存储；\n"
                 "  * 训练数据的分布式 KV（RocksDB + B+ 树 / LSM 树）做样本 shuffle、缓存；\n"
                 "  * 持久化的 checkpoint、experiment tracking 的底层存储；\n"
                 "  * 理解「局部性 / 块I/O」是工程系统性能的第一性原理 —— 也是 GPU 显存\n"
                 "    分层（寄存器/L2/HBM）优化的同一思路：让每次访问都「值回票价」。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
