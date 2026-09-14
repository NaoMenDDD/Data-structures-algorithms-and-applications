// =============================================================================
//  07_tree_properties.cpp
//  树的性质与经典问题：
//    - 高度、直径（diameter）
//    - 平衡判断、对称判断
//    - 结点/叶子/度为 2 的结点计数，验证 n0 = n2 + 1
//    - n 个结点的最小高度 = ceil(log2(n+1)) - 1
//    - Morris 中序遍历（O(1) 额外空间）
//    - 线索二叉树（threaded binary tree）的构建与无栈中序遍历
//
//  编译： g++ -std=c++17 -O2 -o 07_tree_properties 07_tree_properties.cpp
//  运行： ./07_tree_properties
// =============================================================================
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stack>
#include <vector>

using std::cout;
using std::endl;
using std::max;
using std::stack;
using std::vector;

struct Node {
    int   val;
    Node* left;
    Node* right;
    explicit Node(int v) : val(v), left(nullptr), right(nullptr) {}
};

// -----------------------------------------------------------------------------
// 高度：约定「空树高度 = -1，只含叶子的树高度 = 0」（以边数计）
// -----------------------------------------------------------------------------
int height(Node* r) {
    if (!r) return -1;
    return 1 + max(height(r->left), height(r->right));
}

// -----------------------------------------------------------------------------
// 直径：任意两结点间最长路径的边数 = 某个结点左右子树高度之和的最大值 + 2
// 用后序一次遍历完成，O(n)。
// -----------------------------------------------------------------------------
int diameterHelper(Node* r, int& best) {
    if (!r) return -1;
    int lh = diameterHelper(r->left,  best);
    int rh = diameterHelper(r->right, best);
    best = max(best, lh + rh + 2);          // 穿过 r 的最长路径
    return 1 + max(lh, rh);
}
int diameter(Node* r) {
    int best = 0;
    diameterHelper(r, best);
    return best;
}

// -----------------------------------------------------------------------------
// 平衡判断：任一结点左右子树高度差不超过 1。
// 用返回值 -1 表示「已不平衡」做剪枝，仍是 O(n)。
// -----------------------------------------------------------------------------
int balancedHeight(Node* r) {
    if (!r) return 0;                        // 这里用「结点数」计高度，方便
    int lh = balancedHeight(r->left);
    if (lh == -1) return -1;
    int rh = balancedHeight(r->right);
    if (rh == -1) return -1;
    if (std::abs(lh - rh) > 1) return -1;
    return 1 + max(lh, rh);
}
bool isBalanced(Node* r) { return balancedHeight(r) != -1; }

// -----------------------------------------------------------------------------
// 对称判断：镜像比较
// -----------------------------------------------------------------------------
bool mirror(Node* a, Node* b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a->val == b->val && mirror(a->left, b->right) &&
           mirror(a->right, b->left);
}
bool isSymmetric(Node* r) { return !r || mirror(r->left, r->right); }

// -----------------------------------------------------------------------------
// 计数：总结点、叶子、度为 2 的结点
// -----------------------------------------------------------------------------
void countInfo(Node* r, int& total, int& leaves, int& deg2) {
    if (!r) return;
    ++total;
    int d = (r->left ? 1 : 0) + (r->right ? 1 : 0);
    if (d == 0) ++leaves;
    if (d == 2) ++deg2;
    countInfo(r->left,  total, leaves, deg2);
    countInfo(r->right, total, leaves, deg2);
}

// -----------------------------------------------------------------------------
// Morris 中序遍历：借用叶子结点的空右指针暂存「中序后继」，遍历完再复原。
// 时间 O(n)，额外空间 O(1)（不含输出数组）。
// -----------------------------------------------------------------------------
vector<int> morrisInorder(Node* root) {
    vector<int> out;
    Node* cur = root;
    while (cur) {
        if (!cur->left) {
            out.push_back(cur->val);         // 无左子树，直接访问并向右
            cur = cur->right;
        } else {
            Node* pred = cur->left;          // 找中序前驱（左子树最右结点）
            while (pred->right && pred->right != cur) pred = pred->right;
            if (!pred->right) {              // 第一次到达：建立线索
                pred->right = cur;
                cur = cur->left;
            } else {                         // 第二次到达：拆除线索并访问
                pred->right = nullptr;       // 复原树结构
                out.push_back(cur->val);
                cur = cur->right;
            }
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// 线索二叉树：把每个「右指针为空」的结点右指针指向其中序后继。
// -----------------------------------------------------------------------------
struct TNode {
    int    val;
    TNode* left;
    TNode* right;
    bool   rightThread;                     // true 表示 right 是线索而非孩子
    explicit TNode(int v)
        : val(v), left(nullptr), right(nullptr), rightThread(false) {}
};

// 用一次中序遍历建立右线索
void buildThreads(TNode* root) {
    stack<TNode*> st;
    TNode* cur  = root;
    TNode* prev = nullptr;
    while (cur || !st.empty()) {
        while (cur) {
            st.push(cur);
            cur = cur->left;
        }
        cur = st.top();
        st.pop();
        if (prev && !prev->right) {         // 前驱的右指针空闲 => 指向后继
            prev->right = cur;
            prev->rightThread = true;
        }
        prev = cur;
        cur = cur->right;                   // 此刻 cur->right 仍是真实孩子
    }
}

// 借助线索做中序（O(1) 额外空间，且不需要栈）
vector<int> threadedInorder(TNode* root) {
    vector<int> out;
    if (!root) return out;
    TNode* cur = root;
    while (cur->left) cur = cur->left;      // 最左结点
    while (cur) {
        out.push_back(cur->val);
        if (cur->rightThread) {
            cur = cur->right;               // 直接跳到中序后继
        } else {
            cur = cur->right;
            if (!cur) break;
            while (cur->left) cur = cur->left;
        }
    }
    return out;
}

// 由「层序数组」构造完全二叉树（下标 1 起）：孩子为 2i、2i+1
Node* buildCompleteFromLevel(const vector<int>& a, int i, int n) {
    if (i >= n) return nullptr;
    Node* r = new Node(a[i]);
    r->left  = buildCompleteFromLevel(a, 2 * i + 1, n);
    r->right = buildCompleteFromLevel(a, 2 * i + 2, n);
    return r;
}

void destroy(Node* r) {
    if (!r) return;
    destroy(r->left);
    destroy(r->right);
    delete r;
}
void destroyT(TNode* r) {
    if (!r) return;
    if (!r->rightThread) destroyT(r->right);   // 线索不能当孩子递归
    destroyT(r->left);
    delete r;
}

// 向二叉搜索树插入（用来生成结构各异的测试树）
Node* bstInsert(Node* r, int v) {
    if (!r) return new Node(v);
    if (v < r->val) r->left = bstInsert(r->left, v);
    else if (v > r->val) r->right = bstInsert(r->right, v);
    return r;
}

int main() {
    // ---------- 基础性质 ----------
    vector<int> lv = {1, 2, 3, 4, 5, 6, 7};
    Node* full = buildCompleteFromLevel(lv, 0, lv.size());
    assert(height(full) == 2);
    assert(diameter(full) == 4);            // 4-2-1-3-7 或 4-2-1-3-6 等
    assert(isBalanced(full));
    // 注意：full 只是「形状满」，值并不镜像对称（左孩子 2 vs 右孩子 3），
    // 所以 isSymmetric 应为 false。真正的值对称树要让孩子成镜像。
    assert(!isSymmetric(full));

    // 值对称的完美二叉树：层序 {1,2,2,3,4,4,3}
    //           1
    //         ╱   ╲
    //        2     2
    //       ╱ ╲   ╱ ╲
    //      3   4 4   3
    vector<int> symLv = {1, 2, 2, 3, 4, 4, 3};
    Node* sym = buildCompleteFromLevel(symLv, 0, symLv.size());
    assert(isSymmetric(sym));
    destroy(sym);

    int total = 0, leaves = 0, deg2 = 0;
    countInfo(full, total, leaves, deg2);
    assert(total == 7 && leaves == 4 && deg2 == 3);
    assert(leaves == deg2 + 1);             // n0 = n2 + 1

    // ---------- n0 = n2 + 1 的随机验证 ----------
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 200; ++trial) {
        vector<int> perm(20);
        std::iota(perm.begin(), perm.end(), 1);
        std::shuffle(perm.begin(), perm.end(), rng);
        Node* t = nullptr;
        for (int x : perm) t = bstInsert(t, x);
        int tt = 0, lf = 0, d2 = 0;
        countInfo(t, tt, lf, d2);
        assert(lf == d2 + 1);
        destroy(t);
    }
    cout << "随机 BT 上 n0 = n2 + 1 验证通过（200 次）\n";

    // ---------- n 个结点的最小高度 = ceil(log2(n+1)) - 1 ----------
    for (int n = 1; n <= 64; ++n) {
        vector<int> a(n);
        std::iota(a.begin(), a.end(), 0);
        Node* t = buildCompleteFromLevel(a, 0, n);
        int expected = static_cast<int>(std::ceil(std::log2(n + 1.0))) - 1;
        assert(height(t) == expected);
        destroy(t);
    }
    cout << "n=1..64 完全二叉树高度 = ceil(log2(n+1))-1 验证通过\n";

    // ---------- Morris 遍历 = 递归中序 ----------
    Node* t = nullptr;
    vector<int> keys = {50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45};
    for (int x : keys) t = bstInsert(t, x);
    vector<int> sorted = keys;
    std::sort(sorted.begin(), sorted.end());
    assert(morrisInorder(t) == sorted);
    // Morris 不改变树结构，再跑一次仍成立
    assert(morrisInorder(t) == sorted);
    cout << "Morris 中序遍历 == 有序序列，且不改动树结构\n";

    // ---------- 线索二叉树 ----------
    TNode* tr = nullptr;
    // 手工构造：      4
    //              ╱   ╲
    //             2     6
    //            ╱ ╲   ╱ ╲
    //           1   3 5   7
    tr = new TNode(4);
    tr->left = new TNode(2);  tr->right = new TNode(6);
    tr->left->left = new TNode(1); tr->left->right = new TNode(3);
    tr->right->left = new TNode(5); tr->right->right = new TNode(7);
    buildThreads(tr);
    vector<int> tin = threadedInorder(tr);
    assert((tin == vector<int>{1, 2, 3, 4, 5, 6, 7}));
    cout << "线索二叉树无栈中序遍历: ";
    for (int x : tin) cout << x << ' ';
    cout << endl;

    destroy(full);
    destroy(t);
    destroyT(tr);

    cout << "\n[07_tree_properties] 全部自测通过 ✔\n";
    return 0;
}
