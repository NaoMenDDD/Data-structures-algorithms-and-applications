// 09_bst.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 09 章 搜索树》
// 主题：二叉搜索树（BST）—— 折半查找的「动态」版本
//
//   有序数组能 O(log n) 查找，但插入/删除是 O(n)（要移位）；
//   哈希表能 O(1) 查找，但不支持「范围查询 / 按序取出」。
//   BST 想要两全：**查找 O(h)、插入 O(h)、删除 O(h)、且中序即有序**。
//   代价是 h 可能是 O(n)（树退化成链）—— 这引出后面的 AVL / 红黑树。
//
//   本文件演示：
//     1. insert / search / remove（删除的三种情形）；
//     2. 中序 = 升序；min / max / successor / predecessor；
//     3. 范围查询（返回 [lo, hi] 内的所有键）—— 哈希表做不到的事；
//     4. BST 合法性校验（用「上下界」递归，比只比父子更强）；
//     5. ★ 退化实验：按升序插入 1..N，树高 = N-1，查找退化 O(n)。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 09_bst.cpp -o 09_bst && ./09_bst
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 09_bst.cpp -o bst_san && ./bst_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

struct Node {
    int   key;
    Node *left = nullptr, *right = nullptr;
    explicit Node(int k) : key(k) {}
};

// ---------------------------------------------------------------------------
// 插入：小的往左、大的往右（重复键忽略）
// ---------------------------------------------------------------------------
Node* insert(Node* r, int k) {
    if (!r) return new Node(k);
    if (k < r->key) r->left = insert(r->left, k);
    else if (k > r->key) r->right = insert(r->right, k);
    return r;                                  // 相等：忽略（集合语义）
}

bool search(Node* r, int k) {
    while (r) {
        if (k == r->key) return true;
        r = (k < r->key) ? r->left : r->right;   // 迭代版，省栈
    }
    return false;
}

Node* findMin(Node* r) { while (r && r->left) r = r->left; return r; }
Node* findMax(Node* r) { while (r && r->right) r = r->right; return r; }

// ---------------------------------------------------------------------------
// 删除：三种情形
//   ① 叶子            -> 直接删
//   ② 只有一个孩子    -> 用孩子顶替
//   ③ 有两个孩子      -> 用「中序后继」（右子树最小）替换，再删后继
// ---------------------------------------------------------------------------
Node* remove(Node* r, int k) {
    if (!r) return nullptr;
    if (k < r->key)      r->left  = remove(r->left, k);
    else if (k > r->key) r->right = remove(r->right, k);
    else {                                       // 找到要删的结点
        if (!r->left)  { Node* t = r->right; delete r; return t; }   // 情形①②（无左）
        if (!r->right) { Node* t = r->left;  delete r; return t; }   // 情形②（无右）
        Node* succ = findMin(r->right);          // 情形③：中序后继
        r->key = succ->key;
        r->right = remove(r->right, succ->key);  // 递归删掉那个后继
    }
    return r;
}

// ---------------------------------------------------------------------------
// 中序 -> 升序向量；successor / predecessor
// ---------------------------------------------------------------------------
void inorder(Node* r, std::vector<int>& out) {
    if (!r) return;
    inorder(r->left, out);
    out.push_back(r->key);
    inorder(r->right, out);
}

// 中序后继：大于 k 的最小键
Node* successor(Node* root, int k) {
    Node* succ = nullptr;
    while (root) {
        if (k < root->key) { succ = root; root = root->left; }   // root 可能是答案，再试更小的
        else               { root = root->right; }               // k >= root->key，往右找
    }
    return succ;
}

// 中序前驱：小于 k 的最大键
Node* predecessor(Node* root, int k) {
    Node* pred = nullptr;
    while (root) {
        if (k > root->key) { pred = root; root = root->right; }
        else               { root = root->left; }
    }
    return pred;
}

// ---------------------------------------------------------------------------
// 范围查询：返回所有 lo <= key <= hi 的键（BST 的核心优势）
//   利用 BST 性质剪枝：若 root->key 已 < lo，左子树全部不用看；反之亦然。
// ---------------------------------------------------------------------------
void rangeQuery(Node* r, int lo, int hi, std::vector<int>& out) {
    if (!r) return;
    if (lo < r->key) rangeQuery(r->left, lo, hi, out);   // 左子树可能有 < root 的值
    if (lo <= r->key && r->key <= hi) out.push_back(r->key);
    if (r->key < hi) rangeQuery(r->right, lo, hi, out);  // 右子树可能有 > root 的值
}

int height(Node* r) {
    if (!r) return -1;
    const int l = height(r->left), rr = height(r->right);
    return 1 + (l > rr ? l : rr);
}

void destroy(Node* r) {
    if (!r) return;
    destroy(r->left);
    destroy(r->right);
    delete r;
}

// ---------------------------------------------------------------------------
// BST 合法性校验：用 (下界, 上界) 递归
//   只检查「父 > 左孩子 且 父 < 右孩子」是不够的！必须保证整棵左子树 < root。
// ---------------------------------------------------------------------------
bool isBST(Node* r, long lo, long hi) {
    if (!r) return true;
    if (r->key <= lo || r->key >= hi) return false;
    return isBST(r->left, lo, r->key) && isBST(r->right, r->key, hi);
}

int main() {
    std::cout << "======== 09 二叉搜索树（BST）========\n\n";

    // ---------- 基本操作 ----------
    std::cout << "=== 1. 插入 / 中序 / 查找 ===\n";
    Node* root = nullptr;
    const std::vector<int> keys = {50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45, 55, 65, 75, 85};
    for (int k : keys) root = insert(root, k);
    // 重复插入应被忽略
    root = insert(root, 50);
    root = insert(root, 30);

    std::vector<int> sorted;
    inorder(root, sorted);
    std::cout << "  中序遍历（即升序）: ";
    for (std::size_t i = 0; i < sorted.size(); ++i) std::cout << sorted[i] << (i + 1 < sorted.size() ? " " : "");
    std::cout << "\n";
    assert((sorted == std::vector<int>{10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85}));
    assert(isBST(root, -1000000, 1000000));
    assert(search(root, 45) && !search(root, 46));
    std::cout << "  查找 45 = " << search(root, 45) << "，查找 46 = " << search(root, 46) << " ✓\n";
    std::cout << "  min = " << findMin(root)->key << "，max = " << findMax(root)->key << "\n";
    std::cout << "  树高 = " << height(root) << "（15 个结点，理想约 log2(16)=4）\n";

    // ---------- 前驱 / 后继 ----------
    std::cout << "\n=== 2. 前驱 / 后继 ===\n";
    assert(successor(root, 45)->key == 50);
    assert(predecessor(root, 45)->key == 40);
    assert(successor(root, 85) == nullptr);       // 无后继
    assert(predecessor(root, 10) == nullptr);     // 无前驱
    std::cout << "  45 的后继 = " << successor(root, 45)->key
              << "，前驱 = " << predecessor(root, 45)->key << " ✓\n";

    // ---------- 范围查询 ----------
    std::cout << "\n=== 3. 范围查询 [40, 65]（哈希表做不到）===\n";
    std::vector<int> rng;
    rangeQuery(root, 40, 65, rng);
    std::cout << "  ";
    for (std::size_t i = 0; i < rng.size(); ++i) std::cout << rng[i] << (i + 1 < rng.size() ? " " : "");
    std::cout << "\n";
    assert((rng == std::vector<int>{40, 45, 50, 55, 60, 65}));

    // ---------- 删除 ----------
    std::cout << "\n=== 4. 删除（三种情形）===\n";
    // 删叶子 10
    root = remove(root, 10);
    assert(!search(root, 10) && isBST(root, -1000000, 1000000));
    std::cout << "  删叶子 10 ✓\n";
    // 删单孩子结点 25（只有... 其实 25 是叶子；改删 20，它有孩子 25）
    root = remove(root, 20);
    assert(!search(root, 20) && search(root, 25) && isBST(root, -1000000, 1000000));
    std::cout << "  删单孩子结点 20（25 顶上）✓\n";
    // 删双孩子结点 30（用中序后继 35 替换）
    root = remove(root, 30);
    assert(!search(root, 30) && isBST(root, -1000000, 1000000));
    std::cout << "  删双孩子结点 30（中序后继顶替）✓\n";
    sorted.clear(); inorder(root, sorted);
    std::cout << "  删除后中序: ";
    for (std::size_t i = 0; i < sorted.size(); ++i) std::cout << sorted[i] << (i + 1 < sorted.size() ? " " : "");
    std::cout << "\n";

    // ---------- 与 std::set 对拍 ----------
    std::cout << "\n=== 5. 与 std::set 随机对拍 ===\n";
    {
        std::mt19937 rng2(20260913);
        for (int trial = 0; trial < 100; ++trial) {
            Node* t = nullptr;
            std::set<int> s;
            for (int op = 0; op < 300; ++op) {
                const int k = static_cast<int>(rng2() % 200);
                if (rng2() % 2) { t = insert(t, k); s.insert(k); }
                else            { t = remove(t, k); s.erase(k); }
            }
            std::vector<int> got;
            inorder(t, got);
            assert(std::vector<int>(s.begin(), s.end()) == got);
            assert(isBST(t, -1, 100000));
            destroy(t);
        }
        std::cout << "  100 轮 × 300 次随机增删：BST 中序 == std::set ✓\n";
    }

    // ---------- ★ 退化 ----------
    std::cout << "\n=== 6. ★ 退化：按升序插入 1..N ===\n";
    for (int N : {8, 1000, 100000}) {
        Node* chain = nullptr;
        for (int i = 1; i <= N; ++i) chain = insert(chain, i);   // 升序插入
        std::cout << "  升序插入 1.." << N << " -> 树高 = " << height(chain)
                  << "（N-1 = " << N - 1 << "，退化成链表！）\n";
        assert(height(chain) == N - 1);
        destroy(chain);
    }
    std::cout << "  -> 升序插入时 BST 查找是 O(n)，和链表一样慢。\n"
                 "     这正是 AVL / 红黑树 / 跳表要解决的问题：**自动保持平衡**。\n\n";

    destroy(root);
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
