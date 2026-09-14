// ===========================================================================
// 14_nqueens_backtracking.cpp
// 回溯（Backtracking）：在解空间树上「深度优先 + 及时剪枝」
//   1) N 皇后：不合法就回退，统计解的个数（n=1..12），并打印 n=8 的一个解
//   2) 数独求解：在「行 / 列 / 宫」约束下逐格试探 + 回退
//   3) 子集 / 全排列枚举：验证回溯能恰好生成 2^n 个子集、n! 个排列
//
// 回溯 = 递归枚举 + 剪枝。「剪枝」是它与「暴力枚举」的唯一区别：
//   一旦当前部分解已经不可能扩展成完整解，立刻回退，不再往下试。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 14_nqueens_backtracking.cpp -o 14_nqueens_backtracking && ./14_nqueens_backtracking
// ===========================================================================
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace bt {

// ===========================================================================
// 1. N 皇后：用「位掩码」记录列 / 两条对角线的占用，冲突检测 O(1)
//    第 row 行可放的位置 = ~(cols | diag1 | diag2)，取低 n 位
// ===========================================================================
long long g_nodes = 0;                          // 访问过的解空间结点数（尝试放子的次数）

void nqueensCount(int n, int row, int cols, int d1, int d2, long long& count) {
    if (row == n) { ++count; return; }
    int avail = (~(cols | d1 | d2)) & ((1 << n) - 1);
    while (avail) {
        const int bit = avail & (-avail);       // 取最低位的 1
        avail ^= bit;
        ++g_nodes;
        nqueensCount(n, row + 1, cols | bit,
                     ((d1 | bit) << 1) & ((1 << n) - 1),   // 主对角线整体左移
                     (d2 | bit) >> 1,                      // 副对角线整体右移
                     count);
    }
}

// 找出 n 皇后的「第一个」解，并记下每行皇后所在列
bool nqueensFirst(int n, int row, int cols, int d1, int d2, std::vector<int>& pos) {
    if (row == n) return true;
    int avail = (~(cols | d1 | d2)) & ((1 << n) - 1);
    while (avail) {
        const int bit = avail & (-avail);
        avail ^= bit;
        const int col = __builtin_ctz(static_cast<unsigned>(bit));
        pos[static_cast<std::size_t>(row)] = col;
        if (nqueensFirst(n, row + 1, cols | bit,
                         ((d1 | bit) << 1) & ((1 << n) - 1), (d2 | bit) >> 1, pos))
            return true;
    }
    return false;
}

std::vector<std::string> nqueensBoard(int n) {
    std::vector<int> pos(static_cast<std::size_t>(n), 0);
    const bool ok = nqueensFirst(n, 0, 0, 0, 0, pos);
    assert(ok);
    std::vector<std::string> board(static_cast<std::size_t>(n), std::string(static_cast<std::size_t>(n), '.'));
    for (int r = 0; r < n; ++r) board[static_cast<std::size_t>(r)][static_cast<std::size_t>(pos[static_cast<std::size_t>(r)])] = 'Q';
    return board;
}

// ===========================================================================
// 2. 数独：逐个空格试探 1..9，用行 / 列 / 宫三个约束剪枝
// ===========================================================================
bool sudokuSolve(std::vector<std::vector<int>>& g, long long& nodes) {
    int r = -1, c = -1;
    for (int i = 0; i < 9 && r < 0; ++i)          // 找第一个空格
        for (int j = 0; j < 9; ++j)
            if (g[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] == 0) { r = i; c = j; break; }
    if (r < 0) return true;                       // 没有空格 -> 解完
    for (int v = 1; v <= 9; ++v) {
        ++nodes;
        bool ok = true;
        for (int k = 0; k < 9 && ok; ++k)         // 行列宫三查
            if (g[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)] == v ||
                g[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)] == v ||
                g[static_cast<std::size_t>((r / 3) * 3 + k / 3)][static_cast<std::size_t>((c / 3) * 3 + k % 3)] == v)
                ok = false;
        if (!ok) continue;
        g[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = v;
        if (sudokuSolve(g, nodes)) return true;
        g[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = 0;   // 回退
    }
    return false;
}

// ===========================================================================
// 3. 子集 / 全排列枚举
// ===========================================================================
void genSubsets(int n, int i, std::vector<int>& cur, long long& cnt) {
    if (i > n) { ++cnt; return; }                 // 每个元素「选/不选」都走到叶子
    genSubsets(n, i + 1, cur, cnt);               // 不选 i
    cur.push_back(i);
    genSubsets(n, i + 1, cur, cnt);               // 选 i
    cur.pop_back();                               // 回退
}

void genPermutations(int n, std::vector<bool>& used, std::vector<int>& cur, long long& cnt) {
    if (static_cast<int>(cur.size()) == n) { ++cnt; return; }
    for (int v = 1; v <= n; ++v) {
        if (used[static_cast<std::size_t>(v)]) continue;   // 剪枝：已用过
        used[static_cast<std::size_t>(v)] = true;
        cur.push_back(v);
        genPermutations(n, used, cur, cnt);
        cur.pop_back();
        used[static_cast<std::size_t>(v)] = false;         // 回退
    }
}

} // namespace bt

static void testNQueens() {
    std::cout << "===== 1) N 皇后：解空间剪枝 =====\n";
    std::cout << "  n\t解的个数\t访问结点数\n";
    for (int n = 4; n <= 12; ++n) {
        long long count = 0;
        bt::g_nodes = 0;
        bt::nqueensCount(n, 0, 0, 0, 0, count);
        std::cout << "  " << n << "\t" << count << "\t\t" << bt::g_nodes << "\n";
    }
    // 已知解数校验：n=8 有 92 个解，n=4 有 2 个
    long long c8 = 0; bt::nqueensCount(8, 0, 0, 0, 0, c8);
    long long c4 = 0; bt::nqueensCount(4, 0, 0, 0, 0, c4);
    assert(c8 == 92 && c4 == 2);
    std::cout << "  n=8 的一个解（. 为空，Q 为皇后）：\n";
    for (const auto& row : bt::nqueensBoard(8)) std::cout << "    " << row << "\n";
    std::cout << "  [断言通过] n=4 得 2 解、n=8 得 92 解（与已知结果一致）\n\n";
}

static void testSudoku() {
    std::cout << "===== 2) 数独求解（回溯 + 行/列/宫剪枝） =====\n";
    // 0 表示空格
    std::vector<std::vector<int>> g = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9},
    };
    long long nodes = 0;
    const bool ok = bt::sudokuSolve(g, nodes);
    assert(ok);
    std::cout << "  求解成功，试探次数 = " << nodes << "，解为：\n";
    for (const auto& row : g) {
        std::cout << "    ";
        for (int v : row) std::cout << v << " ";
        std::cout << "\n";
    }
    // 校验是合法数独
    for (int i = 0; i < 9; ++i) {
        std::vector<bool> r(10, false), c(10, false), b(10, false);
        for (int k = 0; k < 9; ++k) {
            r[static_cast<std::size_t>(g[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)])] = true;
            c[static_cast<std::size_t>(g[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)])] = true;
            b[static_cast<std::size_t>(g[static_cast<std::size_t>((i / 3) * 3 + k / 3)][static_cast<std::size_t>((i % 3) * 3 + k % 3)])] = true;
        }
        for (int v = 1; v <= 9; ++v) assert(r[static_cast<std::size_t>(v)] && c[static_cast<std::size_t>(v)] && b[static_cast<std::size_t>(v)]);
    }
    std::cout << "  [断言通过] 每行/列/宫都恰含 1..9 各一次\n\n";
}

static void testEnumeration() {
    std::cout << "===== 3) 子集 / 全排列：回溯枚举的规模 =====\n";
    for (int n = 3; n <= 10; ++n) {
        long long sc = 0, pc = 0;
        std::vector<int> cur;
        bt::genSubsets(n, 1, cur, sc);
        std::vector<bool> used(static_cast<std::size_t>(n) + 1, false);
        std::vector<int> perm;
        bt::genPermutations(n, used, perm, pc);
        long long twoN = 1; for (int i = 0; i < n; ++i) twoN *= 2;
        long long fact = 1; for (int i = 2; i <= n; ++i) fact *= i;
        std::cout << "  n=" << n << "\t子集数=" << sc << " (2^n=" << twoN << ")"
                  << "\t排列数=" << pc << " (n!=" << fact << ")\n";
        assert(sc == twoN && pc == fact);
    }
    std::cout << "  [断言通过] 子集数恰为 2^n、排列数恰为 n!（回溯不漏不重）\n\n";
}

int main() {
    std::cout << "########## 回溯：N 皇后 / 数独 / 枚举 ##########\n\n";
    testNQueens();
    testSudoku();
    testEnumeration();
    std::cout << "全部测试通过。\n";
    return 0;
}
