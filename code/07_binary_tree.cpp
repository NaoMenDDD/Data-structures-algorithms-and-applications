// =============================================================================
//  07_binary_tree.cpp
//  链式二叉树：四种遍历（前序/中序/后序/层序）的递归与非递归实现、
//              由「前序 + 中序」重建二叉树、最近公共祖先（LCA）。
//
//  编译： g++ -std=c++17 -O2 -Wall -Wextra -o 07_binary_tree 07_binary_tree.cpp
//  运行： ./07_binary_tree
// =============================================================================
#include <algorithm>          // std::reverse
#include <cassert>
#include <iostream>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

using std::cout;
using std::endl;
using std::queue;
using std::stack;
using std::string;
using std::unordered_map;
using std::vector;

// -----------------------------------------------------------------------------
// 结点定义
// -----------------------------------------------------------------------------
struct Node {
    int   val;
    Node* left;
    Node* right;
    explicit Node(int v) : val(v), left(nullptr), right(nullptr) {}
};

// -----------------------------------------------------------------------------
// 1. 递归遍历
// -----------------------------------------------------------------------------
void preorderRec(Node* r, vector<int>& out) {
    if (!r) return;
    out.push_back(r->val);          // 根
    preorderRec(r->left, out);      // 左
    preorderRec(r->right, out);     // 右
}

void inorderRec(Node* r, vector<int>& out) {
    if (!r) return;
    inorderRec(r->left, out);       // 左
    out.push_back(r->val);          // 根
    inorderRec(r->right, out);      // 右
}

void postorderRec(Node* r, vector<int>& out) {
    if (!r) return;
    postorderRec(r->left, out);     // 左
    postorderRec(r->right, out);    // 右
    out.push_back(r->val);          // 根
}

// -----------------------------------------------------------------------------
// 2. 非递归遍历（显式栈）
// -----------------------------------------------------------------------------

// 前序：压栈顺序「右、左」，出栈即为「根、左、右」
vector<int> preorderIter(Node* root) {
    vector<int> out;
    if (!root) return out;
    stack<Node*> st;
    st.push(root);
    while (!st.empty()) {
        Node* cur = st.top();
        st.pop();
        out.push_back(cur->val);
        if (cur->right) st.push(cur->right);   // 先压右
        if (cur->left)  st.push(cur->left);    // 后压左 => 左子树先访问
    }
    return out;
}

// 中序：一路向左入栈，弹出访问后转向右子树
vector<int> inorderIter(Node* root) {
    vector<int> out;
    stack<Node*> st;
    Node* cur = root;
    while (cur || !st.empty()) {
        while (cur) {                 // 走到最左
            st.push(cur);
            cur = cur->left;
        }
        cur = st.top();
        st.pop();
        out.push_back(cur->val);      // 访问根
        cur = cur->right;             // 转向右子树
    }
    return out;
}

// 后序写法 A：按「根-右-左」做镜像前序，最后整体反转 => 「左-右-根」
vector<int> postorderIter(Node* root) {
    vector<int> out;
    if (!root) return out;
    stack<Node*> st;
    st.push(root);
    while (!st.empty()) {
        Node* cur = st.top();
        st.pop();
        out.push_back(cur->val);
        if (cur->left)  st.push(cur->left);    // 先压左
        if (cur->right) st.push(cur->right);   // 后压右 => 右先出
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// 后序写法 B：单栈 + 记录上一次访问的结点（模拟递归的「回溯」）
vector<int> postorderOneStack(Node* root) {
    vector<int> out;
    stack<Node*> st;
    Node* cur  = root;
    Node* last = nullptr;              // 上一次被访问并输出的结点
    while (cur || !st.empty()) {
        while (cur) {                  // 一路向左
            st.push(cur);
            cur = cur->left;
        }
        Node* top = st.top();
        if (top->right && top->right != last) {
            cur = top->right;          // 右子树还没处理完，转进去
        } else {
            out.push_back(top->val);   // 右子树已处理（或为空），可以输出根
            last = top;
            st.pop();
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// 3. 层序遍历（队列），返回「逐层分组」的结果
// -----------------------------------------------------------------------------
vector<vector<int>> levelOrder(Node* root) {
    vector<vector<int>> res;
    if (!root) return res;
    queue<Node*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = static_cast<int>(q.size());
        vector<int> level;
        level.reserve(sz);
        for (int i = 0; i < sz; ++i) {
            Node* cur = q.front();
            q.pop();
            level.push_back(cur->val);
            if (cur->left)  q.push(cur->left);
            if (cur->right) q.push(cur->right);
        }
        res.push_back(std::move(level));
    }
    return res;
}

// -----------------------------------------------------------------------------
// 4. 由「前序 + 中序」重建二叉树
//    前序第一个元素是根；在中序里找到根的位置，左右即两棵子树的中序区间。
// -----------------------------------------------------------------------------
Node* buildPreIn(const vector<int>& pre, int pl, int pr,
                 const vector<int>& in,  int il, int ir,
                 const unordered_map<int, int>& inPos) {
    if (pl > pr) return nullptr;                 // 空子树
    int rootVal = pre[pl];
    int k = inPos.at(rootVal);                   // 根在中序中的下标
    int leftLen = k - il;                         // 左子树结点个数
    Node* root = new Node(rootVal);
    root->left  = buildPreIn(pre, pl + 1, pl + leftLen, in, il, k - 1, inPos);
    root->right = buildPreIn(pre, pl + leftLen + 1, pr, in, k + 1, ir, inPos);
    return root;
}

Node* buildFromPreIn(const vector<int>& pre, const vector<int>& in) {
    assert(pre.size() == in.size());
    unordered_map<int, int> inPos;
    inPos.reserve(in.size() * 2);
    for (int i = 0; i < static_cast<int>(in.size()); ++i) inPos[in[i]] = i;
    return buildPreIn(pre, 0, static_cast<int>(pre.size()) - 1,
                      in, 0, static_cast<int>(in.size()) - 1, inPos);
}

// -----------------------------------------------------------------------------
// 5. 最近公共祖先（LCA）—— 普通二叉树的朴素递归
//    若 p、q 分别落在某结点左右两侧（或其一就是该结点），该结点即为 LCA。
// -----------------------------------------------------------------------------
Node* lca(Node* root, Node* p, Node* q) {
    if (!root || root == p || root == q) return root;
    Node* L = lca(root->left,  p, q);
    Node* R = lca(root->right, p, q);
    if (L && R) return root;          // 一左一右 => root 是分叉点
    return L ? L : R;
}

Node* findNode(Node* root, int v) {
    if (!root) return nullptr;
    if (root->val == v) return root;
    Node* l = findNode(root->left, v);
    return l ? l : findNode(root->right, v);
}

// -----------------------------------------------------------------------------
// 辅助函数
// -----------------------------------------------------------------------------
bool equalTree(Node* a, Node* b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a->val == b->val && equalTree(a->left, b->left) &&
           equalTree(a->right, b->right);
}

void destroy(Node* root) {
    if (!root) return;
    destroy(root->left);
    destroy(root->right);
    delete root;
}

template <class T>
void printVec(const string& name, const vector<T>& v) {
    cout << name << ": [ ";
    for (const auto& x : v) cout << x << ' ';
    cout << "]\n";
}

// -----------------------------------------------------------------------------
// 自测
// -----------------------------------------------------------------------------
int main() {
    //              1
    //            ╱   ╲
    //           2     3
    //          ╱ ╲   ╱ ╲
    //         4   5 6   7
    vector<int> pre = {1, 2, 4, 5, 3, 6, 7};
    vector<int> in  = {4, 2, 5, 1, 6, 3, 7};

    Node* root = buildFromPreIn(pre, in);

    vector<int> a, b;
    preorderRec(root, a);
    b = preorderIter(root);
    printVec("前序(递归)", a);
    printVec("前序(迭代)", b);
    assert(a == pre);
    assert(b == pre);

    a.clear();
    inorderRec(root, a);
    b = inorderIter(root);
    printVec("中序(递归)", a);
    printVec("中序(迭代)", b);
    assert(a == in);
    assert(b == in);

    a.clear();
    postorderRec(root, a);
    vector<int> c = postorderIter(root);
    vector<int> d = postorderOneStack(root);
    printVec("后序(递归)", a);
    printVec("后序(迭代-反转)", c);
    printVec("后序(迭代-单栈)", d);
    assert(a == c && c == d);
    assert((a == vector<int>{4, 5, 2, 6, 7, 3, 1}));

    vector<vector<int>> lv = levelOrder(root);
    cout << "层序: ";
    for (const auto& level : lv) {
        cout << '[';
        for (size_t i = 0; i < level.size(); ++i)
            cout << level[i] << (i + 1 < level.size() ? "," : "");
        cout << "] ";
    }
    cout << endl;
    assert((lv == vector<vector<int>>{{1}, {2, 3}, {4, 5, 6, 7}}));

    // 重建后的树应与原树结构一致（再序列化比较）
    Node* rebuilt = buildFromPreIn(pre, in);
    assert(equalTree(root, rebuilt));

    // LCA
    Node* n4 = findNode(root, 4);
    Node* n5 = findNode(root, 5);
    Node* n6 = findNode(root, 6);
    Node* n7 = findNode(root, 7);
    assert(lca(root, n4, n5)->val == 2);
    assert(lca(root, n4, n6)->val == 1);
    assert(lca(root, n6, n7)->val == 3);
    assert(lca(root, n4, n4)->val == 4);
    cout << "LCA(4,5)=" << lca(root, n4, n5)->val
         << "  LCA(4,6)=" << lca(root, n4, n6)->val
         << "  LCA(6,7)=" << lca(root, n6, n7)->val << endl;

    // 边界：空树、单结点、退化成链（同时验证非递归遍历不爆栈的直觉）
    assert(preorderIter(nullptr).empty());
    Node* chain = new Node(1);
    chain->right = new Node(2);
    chain->right->right = new Node(3);
    assert((postorderIter(chain) == vector<int>{3, 2, 1}));
    assert((inorderIter(chain) == vector<int>{1, 2, 3}));

    destroy(root);
    destroy(rebuilt);
    destroy(chain);

    cout << "\n[07_binary_tree] 全部自测通过 ✔\n";
    return 0;
}
