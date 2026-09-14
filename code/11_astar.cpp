// 11_astar.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 11 章 图算法》
// 主题：A* 搜索 —— 「有方向感的 Dijkstra」，世界模型做规划的核心
//
//   Dijkstra 从起点向四面八方「均匀」扩散，直到碰到终点。它「没长眼睛」，
//   明明终点在东边，西边的路也要探一遍。
//
//   A* 给每个结点加一个**启发式估计 h(n)**（「离终点还有多远」的猜测），
//   按 **f(n) = g(n) + h(n)** 出队：
//     * g(n)：从起点到 n 的**已知**代价；
//     * h(n)：从 n 到终点的**估计**代价（启发式）。
//   于是搜索被「拽向」终点方向，探的结点大幅减少。
//
//   两条关键性质：
//     ① **可采纳（admissible）**：h(n) <= 真实剩余代价 —— 保证 A* 找到**最优解**。
//        直觉：h 永远不「高估」，就不会因为乐观而放弃最优路径。
//     ② **一致（consistent / monotone）**：h(n) <= w(n,n') + h(n')，则每个点只需
//        扩展一次（像 Dijkstra 一样）。网格上的曼哈顿距离同时满足①②。
//
//   若 h = 0，A* 退化成 Dijkstra；h 越准，探的结点越少，直到 h 完全精确时
//   只沿最短路走。**h 高估（不可采纳）会更快，但可能拿到次优解**（加权 A*）。
//
//   这是**世界模型/强化学习做规划**（在想象的未来里找路）的主力算法，
//   也是游戏寻路、机器人导航、拼图求解（15-puzzle）的标准工具。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 11_astar.cpp -o 11_astar && ./11_astar
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 11_astar.cpp -o astar_san && ./astar_san
// ---------------------------------------------------------------------------

#include <algorithm>      // std::reverse
#include <cassert>
#include <cmath>          // std::abs
#include <cstddef>
#include <functional>     // std::greater
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <vector>

static constexpr int INF = std::numeric_limits<int>::max() / 4;

// 网格地图：'#' 障碍、'.' 空地、'S' 起点、'E' 终点
struct Grid {
    int rows = 0, cols = 0;
    std::vector<std::string> cells;
    int start = -1, goal = -1;                       // 线性编号 r*cols+c

    void parse(const std::vector<std::string>& lines) {
        cells = lines;
        rows = static_cast<int>(lines.size());
        cols = rows ? static_cast<int>(lines[0].size()) : 0;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                const char ch = cells[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                if (ch == 'S') start = r * cols + c;
                if (ch == 'E') goal  = r * cols + c;
            }
    }
    bool passable(int r, int c) const {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return false;
        return cells[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] != '#';
    }
};

// h 类型：0 = Dijkstra（无启发）；1 = 曼哈顿（可采纳，网格 4 邻接）；2 = 加权曼哈顿（不可采纳）
static int heuristic(const Grid& g, int node, int type, int weight) {
    if (type == 0) return 0;
    const int r = node / g.cols, c = node % g.cols;
    const int gr = g.goal / g.cols, gc = g.goal % g.cols;
    const int manhattan = std::abs(r - gr) + std::abs(c - gc);
    return (type == 1) ? manhattan : weight * manhattan;      // 加权 -> 可能高估
}

// A* / Dijkstra 统一实现：hType=0 即 Dijkstra。返回最优代价、路径、扩展结点数。
static int search(const Grid& g, int hType, int hWeight,
                  std::vector<int>& path, long long& expanded) {
    const int n = g.rows * g.cols;
    std::vector<int> gScore(static_cast<std::size_t>(n), INF);
    std::vector<int> parent(static_cast<std::size_t>(n), -1);
    std::vector<char> closed(static_cast<std::size_t>(n), 0);
    using P = std::pair<int,int>;                             // (f, node)
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;

    gScore[static_cast<std::size_t>(g.start)] = 0;
    pq.push({heuristic(g, g.start, hType, hWeight), g.start});
    expanded = 0;

    const int dr[] = {1,-1,0,0}, dc[] = {0,0,1,-1};
    while (!pq.empty()) {
        const auto [f, u] = pq.top(); pq.pop();
        (void)f;
        if (closed[static_cast<std::size_t>(u)]) continue;    // 已扩展过，跳过
        closed[static_cast<std::size_t>(u)] = 1;
        ++expanded;
        if (u == g.goal) break;

        const int ur = u / g.cols, uc = u % g.cols;
        for (int d = 0; d < 4; ++d) {
            const int nr = ur + dr[d], nc = uc + dc[d];
            if (!g.passable(nr, nc)) continue;
            const int v = nr * g.cols + nc;
            const int tentative = gScore[static_cast<std::size_t>(u)] + 1;   // 每条边权 = 1
            if (tentative < gScore[static_cast<std::size_t>(v)]) {
                gScore[static_cast<std::size_t>(v)] = tentative;
                parent[static_cast<std::size_t>(v)] = u;
                pq.push({tentative + heuristic(g, v, hType, hWeight), v});
            }
        }
    }
    path.clear();
    if (gScore[static_cast<std::size_t>(g.goal)] == INF) return INF;
    for (int v = g.goal; v != -1; v = parent[static_cast<std::size_t>(v)]) path.push_back(v);
    std::reverse(path.begin(), path.end());
    return gScore[static_cast<std::size_t>(g.goal)];
}

int main() {
    std::cout << "======== 11 A* 搜索（启发式寻路）========\n\n";

    // ---------- 1. 一张带障碍的地图 ----------
    std::cout << "=== 1. 地图与三种搜索的对比 ===\n";
    Grid g;
    g.parse({
        "S...#.....",
        ".##.#.###.",
        ".#......#.",
        ".#.####.#.",
        "...#..E...",
    });
    assert(g.start >= 0 && g.goal >= 0);

    std::vector<int> pA, pD, pW;
    long long eD = 0, eA = 0, eW = 0;
    const int cD = search(g, 0, 0, pD, eD);       // Dijkstra
    const int cA = search(g, 1, 1, pA, eA);       // A*（曼哈顿）
    const int cW = search(g, 2, 3, pW, eW);       // 加权 A*（w=3，可能次优）

    std::cout << "    算法                路径长   扩展结点数\n";
    std::cout << "    Dijkstra (h=0)       " << cD << "       " << eD << "\n";
    std::cout << "    A* (曼哈顿)          " << cA << "       " << eA << "\n";
    std::cout << "    加权 A* (w=3)        " << cW << "       " << eW << "\n";
    assert(cA == cD);                              // 可采纳 -> 与最优一致
    assert(eA <= eD);                              // 启发式减少扩展
    std::cout << "  -> A* 与 Dijkstra 路径长相同（都最优），但 A* 少扩展 "
              << (eD - eA) << " 个结点（少了 " << (100.0 * (eD - eA) / eD) << "%）✓\n";
    std::cout << "     加权 A* 更少扩展，但可能牺牲最优性（此例 cW=" << cW << "）。\n";

    // 把最优路径画出来
    {
        std::vector<std::string> art = g.cells;
        for (int v : pA) if (v != g.start && v != g.goal)
            art[static_cast<std::size_t>(v / g.cols)][static_cast<std::size_t>(v % g.cols)] = '*';
        std::cout << "\n  A* 找到的最优路径（* 标出）：\n";
        for (const auto& line : art) std::cout << "     " << line << "\n";
    }

    // ---------- 2. ★ 大网格上对拍 + 效率 ----------
    std::cout << "\n=== 2. ★ 随机大网格：A* 与 Dijkstra 结果一致、扩展更少 ===\n";
    {
        std::mt19937 rng(20260913);
        int count = 0;
        double totalRatio = 0;
        for (int trial = 0; trial < 30; ++trial) {
            const int R = 40, C = 40;
            std::vector<std::string> lines(static_cast<std::size_t>(R), std::string(static_cast<std::size_t>(C), '.'));
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c)
                    if (rng() % 100 < 25) lines[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = '#';  // 25% 障碍
            lines[0][0] = 'S';
            lines[static_cast<std::size_t>(R - 1)][static_cast<std::size_t>(C - 1)] = 'E';
            Grid gg; gg.parse(lines);

            std::vector<int> pd, pa;
            long long ed = 0, ea = 0;
            const int cd = search(gg, 0, 0, pd, ed);
            const int ca = search(gg, 1, 1, pa, ea);
            assert(cd == ca);                                    // 最优性一致
            if (cd != INF && ed > 0) { totalRatio += static_cast<double>(ea) / ed; ++count; }
        }
        std::cout << "  30 个 40x40 随机迷宫：A* 与 Dijkstra 路径长完全一致 ✓\n";
        std::cout << "  A* 平均扩展结点数 / Dijkstra = " << (totalRatio / count)
                  << "（越接近 0 说明启发式越省）\n";
    }

    // ---------- 3. ★ h = 0 时 A* 退化成 Dijkstra；开阔图上启发式帮不上忙 ----------
    std::cout << "\n=== 3. ★ 开阔地图：A* 省不了；有障碍才省 ===\n";
    {
        // (a) 无障碍：每个格子都在某条最短路上，f = g + h 恒为常数 -> A* 无法剪枝
        Grid open; open.parse({"S.....", "......", ".....E"});
        std::vector<int> p0, p1;
        long long e0 = 0, e1 = 0;
        search(open, 0, 0, p0, e0);
        search(open, 1, 1, p1, e1);
        std::cout << "  无障碍 3x6：Dijkstra 扩展 " << e0 << " 个，A* 扩展 " << e1
                  << " 个（相同）\n";
        std::cout << "     -> 开阔图上每格都在某条最短路上，f=g+h 恒等，启发式无从剪枝。\n";

        // (b) 有障碍：终点在「右上角」，起点左下，中间一堵近乎封死的墙，
        //     唯一缺口在左边 —— 于是「贴着墙往右」的那一大片格子其实都在绕远路。
        //     Dijkstra 会按 g 大小把它们统统铺开；A* 用 h 看出它们「离终点很远」，
        //     直接推开不扩展。这就是启发式省结点的典型场景。
        Grid wall; wall.parse({
            ".....E...",
            ".........",
            ".........",
            ".#######.",
            ".........",
            "S........",
        });
        std::vector<int> q0, q1;
        long long f0 = 0, f1 = 0;
        const int a0 = search(wall, 0, 0, q0, f0);
        const int a1 = search(wall, 1, 1, q1, f1);
        assert(a0 == a1);
        std::cout << "  有障碍 6x9：Dijkstra 扩展 " << f0 << " 个，A* 扩展 " << f1
                  << " 个（路径长都 = " << a0 << "，A* 只探了 " << (100.0 * f1 / f0) << "%）\n";
        std::cout << "     -> 墙右侧那片格子虽离起点近，却「离终点远」；A* 靠 h 把它们挡住，\n";
        std::cout << "        Dijkstra 没有方向感，只能按 g 把它们一层层铺开。\n";
        assert(f1 < f0);
        std::cout << "  -> 结论：启发式的价值来自「让搜索少走弯路」；开阔直线场景本就无弯路可省。\n";
    }

    std::cout << "\n=== 4. 复杂度和要点 ===\n"
                 "  * A* 复杂度取决于 h：h 越准扩展越少；h=0 时退化为 Dijkstra O(E log V)；\n"
                 "  * **可采纳**（h <= 真实剩余）=> 保证最优；**一致**（h <= w + h'）=> 每点只扩展一次；\n"
                 "  * 曼哈顿距离对 4 邻接网格「可采纳且一致」，是标准选择；\n"
                 "  * 加权 A*（乘系数 > 1）不可采纳 -> 更快但可能次优，是「速度 vs 最优」的旋钮；\n"
                 "  * 与 Dijkstra 共用「优先队列 + 松弛」骨架，只多了一个启发式项；\n"
                 "  * AI 落点：规划（MPC / 树搜索）、游戏寻路、拼图求解、机器人导航。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
