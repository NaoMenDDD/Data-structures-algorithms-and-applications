// 05_call_stack_autodiff.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 05 章 栈》
// 主题：调用栈、递归↔显式栈、以及「自动微分磁带」—— 栈在深度学习里的真实落点
//
//   第 1 部分：递归与调用栈
//     * 递归函数每次调用会在**调用栈（call stack）**上压入一个栈帧（frame）。
//     * 任何递归都能用「显式栈 + 循环」改写（DFS、回溯、编译器、解释器都这么做）。
//     * 这里把阶乘递归改写成显式栈版本，直观看到「进程的栈」和「你手里的栈」是
//       同一个东西。
//
//   第 2 部分：反向模式自动微分（reverse-mode autodiff）的「磁带（tape）」
//     * 这是 PyTorch/TensorFlow 求梯度的核心机制：
//         前向：每做一次运算，就把「一个结点（值 + 局部偏导 + 父结点）」压进一条
//               **磁带（本质是一个栈/数组）**，记录计算图。
//         反向：沿磁带**逆序**（即反向拓扑序）回传梯度。
//     * 所以「栈」不只是括号匹配那种教学例子 —— 它是「自动求导」这件深度学习
//       最核心的事的底层数据结构。
//
//   第 3 部分：反向传播也可以写成「显式栈上的 DFS」—— 呼应第 1 部分。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 05_call_stack_autodiff.cpp -o autodiff && ./autodiff
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 05_call_stack_autodiff.cpp -o autodiff_san && ./autodiff_san
// ---------------------------------------------------------------------------

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stack>
#include <string>
#include <utility>   // std::pair
#include <vector>

// ===========================================================================
// 1. 递归 ↔ 显式栈
// ===========================================================================
//
// 递归写法：调用栈由编译器/运行时替你管理，每层一个栈帧。
static long long factorialRec(int n) {
    if (n <= 1) return 1;
    return n * factorialRec(n - 1);
}

// 显式栈写法：自己维护「待处理」的工作栈。
// 注意：阶乘是「线性递归」，用显式栈反而更啰嗦；但一旦是「树形递归」（如斐波那契、
// 汉诺塔、DFS），显式栈就能把递归深度变成你自己控制的堆内存，从而**避免栈溢出**。
static long long factorialIterative(int n) {
    std::stack<int> work;                 // 模拟「待算的 n」
    for (int i = n; i >= 2; --i) work.push(i);   // 压入 n, n-1, ..., 2
    long long acc = 1;                    // 对应 base case fact(1)=1
    while (!work.empty()) {               // 逐步「返回」
        acc *= work.top();
        work.pop();
    }
    return acc;
}

static void part1_recursion() {
    std::cout << "=== 1. 递归 ↔ 显式栈（阶乘）===\n";
    for (int n : {0, 1, 5, 10, 20}) {
        const long long r = factorialRec(n);
        const long long i = factorialIterative(n);
        std::cout << "  " << n << "! = " << r << "（递归） == " << i << "（显式栈）\n";
        assert(r == i);
    }

    // 「栈帧」长什么样？一个递归函数的「帧」大致包含：参数、局部变量、返回地址。
    // 下面用一个结构体把它显式化，打印出来看。
    struct Frame { int n; long long partial; };
    auto factorialFrames = [](int n) {
        std::vector<Frame> frames;            // 用 vector 当栈，方便打印
        for (int i = n; i >= 1; --i) frames.push_back(Frame{i, 1});
        long long acc = 1;
        std::string trace;
        while (!frames.empty()) {
            Frame f = frames.back();
            frames.pop_back();
            acc *= f.n;
            trace += std::to_string(f.n) + (frames.empty() ? "" : " * ");
        }
        return std::pair<long long, std::string>(acc, trace);
    };
    auto [val, trace] = factorialFrames(6);
    std::cout << "  6! 的显式栈展开: " << trace << " = " << val << "\n";
    std::cout << "  -> 递归深度 = n = 6（每层一个帧），显式栈把「进程栈」换成了「堆上的栈」\n\n";
}

// ===========================================================================
// 2. 反向模式自动微分磁带
// ===========================================================================
//
// 每个结点记录：值 val、梯度 grad、两个父结点下标、以及「对本结点求偏导」的
// 局部导数 (localA = ∂node/∂parentA，localB = ∂node/∂parentB)。
//
// 前向：append 结点（这就是「压栈」）。
// 反向：从根出发 grad=1，**逆序**遍历所有结点，把 grad 沿局部导数传给父结点。
//      逆序遍历之所以正确：父结点一定比子结点**先**被创建，所以下标更小，
//      逆序恰好是「拓扑序的反向」—— 保证结点在被传播梯度前，它自己的 grad 已收齐。
class Tape {
public:
    struct Node {
        double val = 0.0;
        double grad = 0.0;
        int    parentA = -1, parentB = -1;
        double localA  = 0.0, localB = 0.0;   // ∂node/∂parentA, ∂node/∂parentB
    };

    int constant(double v) {
        nodes_.push_back(Node{v});
        return static_cast<int>(nodes_.size()) - 1;
    }

    // 二元算子：给出值、两个父结点、两个局部导数
    int add(int a, int b) { return make(nodes_[a].val + nodes_[b].val, a, b, 1.0, 1.0); }
    int sub(int a, int b) { return make(nodes_[a].val - nodes_[b].val, a, b, 1.0, -1.0); }
    int mul(int a, int b) {
        return make(nodes_[a].val * nodes_[b].val, a, b, nodes_[b].val, nodes_[a].val);
    }
    // 一元算子：只用一个父结点
    int sinOf(int a) { return make(std::sin(nodes_[a].val), a, -1, std::cos(nodes_[a].val), 0.0); }
    int neg(int a)   { return make(-nodes_[a].val, a, -1, -1.0, 0.0); }

    double value(int i) const { return nodes_[static_cast<std::size_t>(i)].val; }
    double grad(int i)  const { return nodes_[static_cast<std::size_t>(i)].grad; }
    std::size_t tapeSize() const { return nodes_.size(); }

    // 反向传播：从 root 出发，逆序把梯度送回去
    void backward(int root) {
        for (Node& n : nodes_) n.grad = 0.0;             // 清零
        nodes_[static_cast<std::size_t>(root)].grad = 1.0;  // ∂root/∂root = 1
        for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
            const Node& n = nodes_[static_cast<std::size_t>(i)];
            if (n.parentA >= 0)
                nodes_[static_cast<std::size_t>(n.parentA)].grad += n.grad * n.localA;
            if (n.parentB >= 0)
                nodes_[static_cast<std::size_t>(n.parentB)].grad += n.grad * n.localB;
        }
    }

private:
    int make(double v, int a, int b, double la, double lb) {
        nodes_.push_back(Node{v, 0.0, a, b, la, lb});
        return static_cast<int>(nodes_.size()) - 1;
    }
    std::vector<Node> nodes_;
};

static void part2_autodiff() {
    std::cout << "=== 2. 反向模式自动微分的「磁带」（栈）===\n";

    // 目标函数：f(x, y) = (x*y + x) + sin(x)
    // 手算：∂f/∂x = y + 1 + cos(x)，∂f/∂y = x
    const double x = 2.0, y = 3.0;

    Tape tape;
    const int ix  = tape.constant(x);
    const int iy  = tape.constant(y);
    const int xy  = tape.mul(ix, iy);        // x*y            = 6
    const int t1  = tape.add(xy, ix);        // x*y + x        = 8
    const int sx  = tape.sinOf(ix);          // sin(x)         = 0.9093
    const int f   = tape.add(t1, sx);        // (x*y+x)+sin(x) = 8.9093

    tape.backward(f);

    const double dfdx = tape.grad(ix);
    const double dfdy = tape.grad(iy);
    const double expectDx = y + 1.0 + std::cos(x);
    const double expectDy = x;

    std::cout << "  f = (x*y + x) + sin(x)，取 x=2, y=3\n";
    std::cout << "  f        = " << tape.value(f) << "  （手算 8 + sin(2) = 8.9093）\n";
    std::cout << "  ∂f/∂x    = " << dfdx << "  （解析 y+1+cos(x) = " << expectDx << "）\n";
    std::cout << "  ∂f/∂y    = " << dfdy << "  （解析 x = " << expectDy << "）\n";
    std::cout << "  磁带长度 = " << tape.tapeSize() << " 个结点（前向每步压一个）\n";
    assert(std::fabs(dfdx - expectDx) < 1e-12);
    assert(std::fabs(dfdy - expectDy) < 1e-12);

    // 用「数值梯度」独立验证一遍（有限差分），双重保险
    auto fval = [](double x, double y) { return (x * y + x) + std::sin(x); };
    const double h = 1e-6;
    const double numDx = (fval(x + h, y) - fval(x - h, y)) / (2 * h);
    const double numDy = (fval(x, y + h) - fval(x, y - h)) / (2 * h);
    std::cout << "  数值梯度 ∂f/∂x ≈ " << numDx << ", ∂f/∂y ≈ " << numDy
              << "（与磁带结果一致）\n";
    assert(std::fabs(dfdx - numDx) < 1e-6);
    assert(std::fabs(dfdy - numDy) < 1e-6);
    std::cout << "  -> 这就是 PyTorch autograd 的骨架：前向建图（压栈），反向回传（逆序出栈）\n\n";
}

// ===========================================================================
// 3. 反向传播 = 显式栈上的 DFS（呼应第 1 部分）
// ===========================================================================
//
// 上面用「逆序数组」代替了显式栈，因为它们等价。这里用真正的 DFS 显式栈再实现
// 一遍，强调「反向传播本质上是对计算图做一次 DFS/拓扑遍历」。
struct DNode {
    char   op;                 // '+', '*', 's'(sin), 'c'(const)
    double val  = 0.0;
    double grad = 0.0;
    DNode* a = nullptr;
    DNode* b = nullptr;
};

// 从 root 出发，用显式栈做 DFS，先得到**后序**（每个结点只记一次），再逆序回传。
// 必须去重：同一个子表达式（如本例的 x）会被多个父结点共享，若重复记录，
// 它的梯度被回传多次就会算错。这正是「计算图是 DAG 而非树」带来的关键细节。
static void backwardDFS(DNode* root) {
    std::vector<DNode*> topo;
    std::vector<DNode*> done;                          // 访问过的结点
    std::vector<std::pair<DNode*, bool>> stk{{root, false}};

    auto visited = [&](DNode* n) {
        for (DNode* d : done) if (d == n) return true;
        return false;
    };

    while (!stk.empty()) {
        auto [n, expanded] = stk.back();
        stk.pop_back();
        if (expanded) { topo.push_back(n); continue; }  // 子都处理完，才记入后序
        if (visited(n)) continue;
        done.push_back(n);
        stk.push_back({n, true});                       // 稍后再记 n（后序）
        if (n->a) stk.push_back({n->a, false});
        if (n->b) stk.push_back({n->b, false});
    }

    // 清零
    for (DNode* n : topo) n->grad = 0.0;
    root->grad = 1.0;
    // 逆后序 = 反向拓扑序：保证处理某结点时，消费它的结点都已回传完毕
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        DNode* n = *it;
        switch (n->op) {
            case '+':
                if (n->a) n->a->grad += n->grad * 1.0;
                if (n->b) n->b->grad += n->grad * 1.0;
                break;
            case '*':
                if (n->a) n->a->grad += n->grad * n->b->val;
                if (n->b) n->b->grad += n->grad * n->a->val;
                break;
            case 's':
                if (n->a) n->a->grad += n->grad * std::cos(n->a->val);
                break;
            default: break;   // 'c' 常数：无父结点
        }
    }
}

static void part3_dfs_backward() {
    std::cout << "=== 3. 反向传播 = 显式栈上的 DFS ===\n";
    // 复用同一个函数 f(x,y) = (x*y + x) + sin(x)
    DNode cx{'c'}, cy{'c'}, m{'*'}, p{'+'}, s{'s'}, f{'+'};
    cx.val = 2.0; cy.val = 3.0;
    m.a = &cx; m.b = &cy; m.val = cx.val * cy.val;      // x*y
    p.a = &m;  p.b = &cx; p.val = m.val + cx.val;       // x*y + x
    s.a = &cx;            s.val = std::sin(cx.val);     // sin(x)
    f.a = &p;  f.b = &s;  f.val = p.val + s.val;        // 总和

    backwardDFS(&f);
    std::cout << "  用显式栈 DFS 回传：∂f/∂x = " << cx.grad
              << ", ∂f/∂y = " << cy.grad << "\n";
    assert(std::fabs(cx.grad - (3.0 + 1.0 + std::cos(2.0))) < 1e-12);
    assert(std::fabs(cy.grad - 2.0) < 1e-12);
    std::cout << "  与第 2 部分的磁带结果一致 ✓\n";
    std::cout << "  -> 深度学习框架的 backward 就是「对计算图做一次拓扑/DFS 遍历」，\n"
                 "     而遍历的载体正是「栈」。\n\n";
}

int main() {
    std::cout << "======== 05 调用栈 / 递归 / 自动微分磁带 ========\n\n";
    part1_recursion();
    part2_autodiff();
    part3_dfs_backward();
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
