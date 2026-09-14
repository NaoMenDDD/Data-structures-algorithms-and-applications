// 09_avl.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 09 章 搜索树》
// 主题：AVL 树 —— 让 BST「永远不退化」的自平衡二叉搜索树
//
//   上一节我们看到：升序插入会让 BST 退化成链表，查找变成 O(n)。
//   AVL 树（Adelson-Velsky & Landis, 1962）用一条**不变式**根治了它：
//
//     对**每个结点**，其左右子树高度差（平衡因子 BF）∈ {-1, 0, +1}。
//
//   一旦插入/删除破坏不变式，就用**旋转**（rotation）在 O(1) 内局部修复，
//   且修复后整棵子树的高度恢复到插入前的值 —— 所以只需沿路径修一趟，O(log n)。
//
//   四种失衡 + 对应旋转：
//     LL（左左，新结点在左孩子的左边）-> 右旋
//     RR（右右）                      -> 左旋
//     LR（左右，新结点在左孩子的右边）-> 先左旋左孩子，再右旋
//     RL（右左）                      -> 先右旋右孩子，再左旋
//   口诀：看「从失衡点往下走的前两步方向」，LL/RR 单旋，LR/RL 双旋。
//
//   高度界：n 个结点的 AVL 树高度 h < 1.44 log2(n+2) - 0.328，故查找/插入/删除 O(log n)。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 09_avl.cpp -o 09_avl && ./09_avl
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 09_avl.cpp -o avl_san && ./avl_san
// ---------------------------------------------------------------------------

#include <algorithm>      // std::max / std::shuffle
#include <cassert>
#include <cmath>          // std::abs / std::log2
#include <cstddef>
#include <iostream>
#include <random>
#include <set>
#include <vector>

struct Node {
    int   key;
    int   height = 1;                 // 叶子高度 = 1（空树高度 = 0）
    Node *left = nullptr, *right = nullptr;
    explicit Node(int k) : key(k) {}
};

static int  H(Node* r) { return r ? r->height : 0; }
static int  BF(Node* r) { return r ? H(r->left) - H(r->right) : 0; }
static void pull(Node* r) { r->height = 1 + std::max(H(r->left), H(r->right)); }

// ---------------------------------------------------------------------------
// 旋转：只改指针，O(1)。旋转后必须重新 pull 受影响的两个结点。
// ---------------------------------------------------------------------------
static Node* rotateRight(Node* y) {           // 右旋：y 的左边过重
    Node* x = y->left;
    y->left = x->right;
    x->right = y;
    pull(y); pull(x);                         // 先 pull 较低的 y，再 pull 新的根 x
    return x;                                 // x 成为子树新根
}

static Node* rotateLeft(Node* x) {            // 左旋：x 的右边过重
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    pull(x); pull(y);
    return y;
}

// ---------------------------------------------------------------------------
// 插入：普通 BST 插入 + 回溯时修复平衡
// ---------------------------------------------------------------------------
static Node* insert(Node* r, int k) {
    if (!r) return new Node(k);
    if (k < r->key)      r->left  = insert(r->left, k);
    else if (k > r->key) r->right = insert(r->right, k);
    else                 return r;            // 重复键忽略

    pull(r);
    const int bf = BF(r);

    if (bf > 1  && BF(r->left)  >= 0) return rotateRight(r);              // LL
    if (bf > 1  && BF(r->left)  <  0) { r->left  = rotateLeft(r->left);  // LR
                                        return rotateRight(r); }
    if (bf < -1 && BF(r->right) <= 0) return rotateLeft(r);               // RR
    if (bf < -1 && BF(r->right) >  0) { r->right = rotateRight(r->right);// RL
                                        return rotateLeft(r); }
    return r;
}

static Node* findMin(Node* r) { while (r && r->left) r = r->left; return r; }

// ---------------------------------------------------------------------------
// 删除：BST 删除 + 回溯时修复（删除比插入更麻烦：可能一路修到根）
// ---------------------------------------------------------------------------
static Node* remove(Node* r, int k) {
    if (!r) return nullptr;
    if (k < r->key)      r->left  = remove(r->left, k);
    else if (k > r->key) r->right = remove(r->right, k);
    else {
        if (!r->left || !r->right) {          // 0 或 1 个孩子
            Node* child = r->left ? r->left : r->right;
            delete r;
            return child;
        }
        Node* succ = findMin(r->right);       // 2 个孩子：中序后继
        r->key = succ->key;
        r->right = remove(r->right, succ->key);
    }

    pull(r);
    const int bf = BF(r);

    if (bf > 1  && BF(r->left)  >= 0) return rotateRight(r);              // LL
    if (bf > 1  && BF(r->left)  <  0) { r->left  = rotateLeft(r->left);  // LR
                                        return rotateRight(r); }
    if (bf < -1 && BF(r->right) <= 0) return rotateLeft(r);               // RR
    if (bf < -1 && BF(r->right) >  0) { r->right = rotateRight(r->right);// RL
                                        return rotateLeft(r); }
    return r;
}

static bool search(Node* r, int k) {
    while (r) { if (k == r->key) return true; r = (k < r->key) ? r->left : r->right; }
    return false;
}

// ---------------------------------------------------------------------------
// 校验：中序有序 + 每个结点 BF ∈ {-1,0,1} + height 字段正确 + 返回树高
// ---------------------------------------------------------------------------
static int checkAvl(Node* r, std::vector<int>& out, bool* balanced) {
    if (!r) return 0;
    const int lh = checkAvl(r->left, out, balanced);
    out.push_back(r->key);
    const int rh = checkAvl(r->right, out, balanced);
    if (std::abs(lh - rh) > 1) *balanced = false;             // 不变式被破坏
    if (r->height != 1 + std::max(lh, rh)) *balanced = false; // height 字段算错
    return 1 + std::max(lh, rh);
}

static void destroy(Node* r) {
    if (!r) return;
    destroy(r->left); destroy(r->right); delete r;
}

int main() {
    std::cout << "======== 09 AVL 树：自平衡二叉搜索树 ========\n\n";

    // ---------- 演示：升序插入不再退化 ----------
    std::cout << "=== 1. ★ 升序插入 1..N（BST 会退化成链，AVL 不会）===\n";
    std::cout << "      N    BST高度(=N-1)   AVL高度   AVL理论上界 1.44*log2(N+2)\n";
    for (int N : {8, 1000, 100000}) {
        Node* avl = nullptr;
        for (int i = 1; i <= N; ++i) avl = insert(avl, i);
        std::vector<int> out; bool ok = true;
        const int h = checkAvl(avl, out, &ok);
        const double bound = 1.44 * std::log2(double(N) + 2.0);
        std::cout << "     " << N << "   " << (N - 1) << "   " << h << "   " << bound << "\n";
        assert(ok);
        assert(h <= static_cast<int>(bound) + 1);
        for (int i = 1; i <= N; ++i) assert(out[static_cast<std::size_t>(i - 1)] == i);
        assert(search(avl, 1) && search(avl, N) && !search(avl, N + 1));
        destroy(avl);
    }
    std::cout << "  -> 同样升序插入，AVL 高度被压到 O(log n)，查找不再退化。\n\n";

    // ---------- 四种旋转的定点测试 ----------
    std::cout << "=== 2. 四种失衡各自触发对应旋转 ===\n";
    const char* names[] = {"LL (右旋)", "RR (左旋)", "LR (先左旋再右旋)", "RL (先右旋再左旋)"};
    const std::vector<std::vector<int>> cases = {
        {30, 20, 10}, {10, 20, 30}, {30, 10, 20}, {10, 30, 20}
    };
    for (std::size_t i = 0; i < cases.size(); ++i) {
        Node* t = nullptr;
        for (int k : cases[i]) t = insert(t, k);
        std::vector<int> out; bool ok = true;
        checkAvl(t, out, &ok);
        // 三次插入后中序必须有序，且根应回到中间的键（20）
        assert(ok && (out == std::vector<int>{10, 20, 30}));
        assert(t->key == 20);
        std::cout << "   " << names[i] << "：插入后根 = " << t->key << "，中序 10 20 30 ✓\n";
        destroy(t);
    }

    // ---------- 与 std::set 随机对拍（增 + 删）----------
    std::cout << "\n=== 3. 与 std::set 随机对拍（每插入后必查平衡）===\n";
    {
        std::mt19937 rng(20260913);
        long long maxH = 0;                       // 记录最大树高，验证始终 O(log n)
        for (int trial = 0; trial < 100; ++trial) {
            Node* t = nullptr;
            std::set<int> s;
            for (int op = 0; op < 400; ++op) {
                const int k = static_cast<int>(rng() % 300);
                if (rng() % 3 != 0) { t = insert(t, k); s.insert(k); }   // 2/3 插入
                else                { t = remove(t, k); s.erase(k); }    // 1/3 删除
                std::vector<int> got; bool ok = true;
                const int h = checkAvl(t, got, &ok);
                assert(ok);                                           // 任意时刻都平衡
                assert(std::vector<int>(s.begin(), s.end()) == got);  // 与 set 一致
                maxH = std::max<long long>(maxH, h);
            }
            destroy(t);
        }
        std::cout << "  100 轮 × 400 次随机增删：全程平衡因子合法 + 中序 == std::set ✓\n";
        std::cout << "  过程中出现的最大树高 = " << maxH << "（键域 0..299，log2(300) ≈ "
                  << std::log2(300.0) << "，始终 O(log n)）\n";
    }

    // ---------- 删除的定向测试 ----------
    std::cout << "\n=== 4. 删除后仍保持平衡（删除触发连锁旋转）===\n";
    {
        Node* t = nullptr;
        for (int i = 1; i <= 20; ++i) t = insert(t, i);   // 20 个结点
        std::vector<int> before; bool ok = true; checkAvl(t, before, &ok);
        assert(ok);
        // 删一半，且不是简单的从一端删（乱序删，最容易触发双旋）
        for (int k : {8, 3, 15, 1, 20, 11, 6}) t = remove(t, k);
        std::vector<int> after; ok = true; const int h = checkAvl(t, after, &ok);
        assert(ok);
        assert(static_cast<int>(after.size()) == 20 - 7);
        for (std::size_t i = 1; i < after.size(); ++i) assert(after[i - 1] < after[i]);
        std::cout << "  删 7 个结点后：剩余 " << after.size() << " 个，高度 " << h << "，仍平衡 ✓\n";
        destroy(t);
    }

    // ---------- 自平衡 vs 朴素 BST 的高度对比 ----------
    std::cout << "\n=== 5. 同一个随机序列：AVL 高度 vs 最坏 BST 高度 ===\n";
    {
        std::mt19937 rng(42);
        std::vector<int> seq;
        for (int i = 0; i < 10000; ++i) seq.push_back(i);
        std::shuffle(seq.begin(), seq.end(), rng);

        Node* avl = nullptr;
        for (int k : seq) avl = insert(avl, k);
        std::vector<int> o; bool ok = true;
        const int ah = checkAvl(avl, o, &ok);
        assert(ok);
        std::cout << "  随机插入 10000 个键：AVL 高度 = " << ah
                  << "（理想 log2(10000) ≈ " << std::log2(10000.0) << "）\n";
        destroy(avl);
    }

    std::cout << "\n=== 6. 复杂度小结 ===\n"
                 "  * 查找 / 插入 / 删除：均 O(log n)（高度被不变式锁死）；\n"
                 "  * 每次操作最多 O(log n) 次旋转，每次旋转 O(1)；\n"
                 "  * 代价：每个结点多存一个 height（或 BF），插入/删除要回溯修复；\n"
                 "  * 对比：红黑树旋转更少、增删更快，是 STL map/set 的实现；\n"
                 "          AVL 更严格平衡，查询更快，适合「读多写少」。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
