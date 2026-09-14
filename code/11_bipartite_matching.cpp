// 11_bipartite_matching.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 11 章 图算法》
// 主题：二分图匹配 —— 最大卡氏匹配（匈牙利/增广路）、König 定理、指派问题
//
//   二分图（bipartite graph）：顶点能分成两半 L、R，所有边都「跨两半」。
//   匹配（matching）：一组互不共端点的边。**最大匹配** = 边数最多的匹配。
//   现实原型：任务分给工人、论文分给审稿人、广告位对应用户、半监督标注里
//             「无标签样本 ↔ 聚类中心」的指派……都是同一件事。
//
//   ★ 增广路（augmenting path）：从「未匹配」的左点出发，交替走
//     「非匹配边 / 匹配边 / 非匹配边 …」，终于一个「未匹配」的右点。
//     把这条路上匹配/非匹配**翻转**，匹配数 +1。
//     **Berge 定理**：匹配最大 ⟺ 不存在增广路。于是不断找增广路即可。
//     这就是 Kuhn（匈牙利增广路）算法，复杂度 O(V·E)。
//
//   ★ König 定理：二分图里 **最大匹配 = 最小点覆盖**。
//     （点覆盖：一组顶点，使得每条边至少有一端被选中。）
//     配套地，最大匹配 = 最大独立集 的补，于是 最大独立集 = V − 最小点覆盖。
//     「匹配↔覆盖」这层对偶是组合优化里最漂亮的结果之一，也是 min-cut 的影子。
//
//   ★ 指派问题（assignment）：n 个工人干 n 项活，代价 c[i][j]，
//     求「一人一活」的**最小总代价完美匹配**。用匈牙利算法（带势的增广路，
//     即 Kuhn–Munkres）在 O(n^3) 内解决。它其实是**最小费用流**的特例，
//     而最小费用流又是「最短路的反复增广」—— 与 Dijkstra 一脉相承。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 11_bipartite_matching.cpp -o 11_bipartite_matching && ./11_bipartite_matching
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 11_bipartite_matching.cpp -o match_san && ./match_san
// ---------------------------------------------------------------------------

#include <algorithm>      // std::next_permutation, std::min
#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>        // std::iota
#include <random>
#include <string>
#include <vector>

static constexpr int INF = std::numeric_limits<int>::max() / 4;
static constexpr long long LLINF = std::numeric_limits<long long>::max() / 4;

// ===========================================================================
// 第一部分：二分图最大匹配（Kuhn 增广路算法）
//   left 点 0..L-1，right 点 0..R-1；adj[u] 给出左点 u 的所有右邻居。
// ===========================================================================
class KuhnMatching {
public:
    KuhnMatching(int L, int R) : L_(L), R_(R), adj_(static_cast<std::size_t>(L)) {}

    void addEdge(int u, int v) { adj_[static_cast<std::size_t>(u)].push_back(v); }

    // 返回最大匹配大小，并填充 matchL_ / matchR_（-1 表示未匹配）。
    int solve() {
        matchL_.assign(static_cast<std::size_t>(L_), -1);
        matchR_.assign(static_cast<std::size_t>(R_), -1);
        int result = 0;
        for (int u = 0; u < L_; ++u) {
            used_.assign(static_cast<std::size_t>(R_), 0);   // 每轮重置「右点是否访问过」
            if (tryAugment(u)) ++result;
        }
        return result;
    }

    const std::vector<int>& matchL() const { return matchL_; }
    const std::vector<int>& matchR() const { return matchR_; }

    // ★ König：返回最小点覆盖的大小（= 最大匹配），并标出被选中的顶点。
    //   做法：从「未匹配的左点」出发做交替可达性搜索得集合 Z；
    //         最小点覆盖 = (L \ Z) ∪ (R ∩ Z)。
    int minVertexCover(std::vector<char>& coverL, std::vector<char>& coverR) {
        // 要求已经调用过 solve()
        std::vector<char> visL(static_cast<std::size_t>(L_), 0);
        std::vector<char> visR(static_cast<std::size_t>(R_), 0);
        std::vector<int> stack;
        for (int u = 0; u < L_; ++u)
            if (matchL_[static_cast<std::size_t>(u)] == -1) { visL[static_cast<std::size_t>(u)] = 1; stack.push_back(u); }
        while (!stack.empty()) {
            const int u = stack.back(); stack.pop_back();
            for (int v : adj_[static_cast<std::size_t>(u)]) {
                if (visR[static_cast<std::size_t>(v)]) continue;
                visR[static_cast<std::size_t>(v)] = 1;                       // 非匹配边：左 -> 右
                const int w = matchR_[static_cast<std::size_t>(v)];
                if (w != -1 && !visL[static_cast<std::size_t>(w)]) {         // 匹配边：右 -> 左
                    visL[static_cast<std::size_t>(w)] = 1;
                    stack.push_back(w);
                }
            }
        }
        coverL.assign(static_cast<std::size_t>(L_), 0);
        coverR.assign(static_cast<std::size_t>(R_), 0);
        int cnt = 0;
        for (int u = 0; u < L_; ++u) if (!visL[static_cast<std::size_t>(u)]) { coverL[static_cast<std::size_t>(u)] = 1; ++cnt; }
        for (int v = 0; v < R_; ++v) if ( visR[static_cast<std::size_t>(v)]) { coverR[static_cast<std::size_t>(v)] = 1; ++cnt; }
        return cnt;
    }

private:
    // 试着从左点 u 找一条增广路；used_ 防止在同一轮里重复访问同一右点。
    bool tryAugment(int u) {
        for (int v : adj_[static_cast<std::size_t>(u)]) {
            if (used_[static_cast<std::size_t>(v)]) continue;
            used_[static_cast<std::size_t>(v)] = 1;
            if (matchR_[static_cast<std::size_t>(v)] == -1 || tryAugment(matchR_[static_cast<std::size_t>(v)])) {
                matchL_[static_cast<std::size_t>(u)] = v;                    // 翻转：u 占住 v
                matchR_[static_cast<std::size_t>(v)] = u;
                return true;
            }
        }
        return false;
    }

    int L_, R_;
    std::vector<std::vector<int>> adj_;
    std::vector<int> matchL_, matchR_;
    std::vector<char> used_;
};

// 参考实现：用「枚举每个左点匹配谁」的 DP 求最大匹配（只用于小图对拍）。
static int bruteForceMatching(int L, int R, const std::vector<std::vector<int>>& adj) {
    std::vector<std::vector<int>> memo(static_cast<std::size_t>(L) + 1,
                                       std::vector<int>(static_cast<std::size_t>(1) << R, -1));
    // f(i, mask)：考虑左点 i..L-1，右点已用 mask 时能得到的最大匹配数
    struct Rec {
        int L, R;
        const std::vector<std::vector<int>>& adj;
        std::vector<std::vector<int>>& memo;
        int f(int i, int mask) {
            if (i == L) return 0;
            int& m = memo[static_cast<std::size_t>(i)][static_cast<std::size_t>(mask)];
            if (m != -1) return m;
            m = f(i + 1, mask);                                          // 左点 i 不匹配
            for (int v : adj[static_cast<std::size_t>(i)])
                if (!(mask & (1 << v))) m = std::max(m, 1 + f(i + 1, mask | (1 << v)));
            return m;
        }
    } rec{L, R, adj, memo};
    return rec.f(0, 0);
}

// ===========================================================================
// 第二部分：指派问题（最小代价完美匹配）—— 匈牙利 / Kuhn–Munkres，O(n^3)
//   a 为 n×n 代价矩阵；返回最小总代价。要求 n>=1。
//   核心思想：维护每行/每列的「势」u[i]、v[j]，让「约化代价」 c[i][j]-u[i]-v[j] >= 0，
//   再找「零代价」的增广路；不成立时用最小差值 delta 调整势 —— 这就是对偶。
// ===========================================================================
static long long hungarian(const std::vector<std::vector<long long>>& a) {
    const int n = static_cast<int>(a.size());
    std::vector<long long> u(static_cast<std::size_t>(n) + 1, 0), v(static_cast<std::size_t>(n) + 1, 0);
    std::vector<int> p(static_cast<std::size_t>(n) + 1, 0), way(static_cast<std::size_t>(n) + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<long long> minv(static_cast<std::size_t>(n) + 1, LLINF);
        std::vector<char> used(static_cast<std::size_t>(n) + 1, 0);
        do {
            used[static_cast<std::size_t>(j0)] = 1;
            const int i0 = p[static_cast<std::size_t>(j0)];
            long long delta = LLINF;
            int j1 = -1;
            for (int j = 1; j <= n; ++j) {
                if (used[static_cast<std::size_t>(j)]) continue;
                const long long cur = a[static_cast<std::size_t>(i0 - 1)][static_cast<std::size_t>(j - 1)]
                                      - u[static_cast<std::size_t>(i0)] - v[static_cast<std::size_t>(j)];
                if (cur < minv[static_cast<std::size_t>(j)]) {
                    minv[static_cast<std::size_t>(j)] = cur;
                    way[static_cast<std::size_t>(j)] = j0;
                }
                if (minv[static_cast<std::size_t>(j)] < delta) { delta = minv[static_cast<std::size_t>(j)]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[static_cast<std::size_t>(j)]) {
                    u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
                    v[static_cast<std::size_t>(j)] -= delta;
                } else {
                    minv[static_cast<std::size_t>(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[static_cast<std::size_t>(j0)] != 0);
        do {                                                              // 沿 way 回溯，翻转匹配
            const int j1 = way[static_cast<std::size_t>(j0)];
            p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }
    long long cost = 0;
    for (int j = 1; j <= n; ++j)
        if (p[static_cast<std::size_t>(j)] > 0) cost += a[static_cast<std::size_t>(p[static_cast<std::size_t>(j)] - 1)][static_cast<std::size_t>(j - 1)];
    return cost;
}

// 参考实现：枚举所有排列求最小代价（只用于小 n 对拍）。
static long long bruteForceAssignment(const std::vector<std::vector<long long>>& a) {
    const int n = static_cast<int>(a.size());
    std::vector<int> perm(static_cast<std::size_t>(n));
    std::iota(perm.begin(), perm.end(), 0);
    long long best = LLINF;
    do {
        long long s = 0;
        for (int i = 0; i < n; ++i) s += a[static_cast<std::size_t>(i)][static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])];
        best = std::min(best, s);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

int main() {
    std::cout << "======== 11 二分图匹配与指派问题 ========\n\n";

    // ---------- 1. 最大匹配：把 4 个任务分给工人 ----------
    std::cout << "=== 1. 最大匹配：任务分给工人 ===\n";
    {
        // 左 = 4 个任务 T0..T3，右 = 4 个工人 W0..W3；边 = 「谁能干」
        KuhnMatching m(4, 4);
        // T0: W0,W1   T1: W1   T2: W1,W2   T3: W2,W3
        m.addEdge(0, 0); m.addEdge(0, 1);
        m.addEdge(1, 1);
        m.addEdge(2, 1); m.addEdge(2, 2);
        m.addEdge(3, 2); m.addEdge(3, 3);
        const int sz = m.solve();
        std::cout << "  任务 T0{W0,W1}  T1{W1}  T2{W1,W2}  T3{W2,W3}\n";
        std::cout << "  最大匹配（能同时派出的活）= " << sz << "（此图 4 个任务全能派出去）\n";
        std::cout << "  一个最优指派：";
        for (int u = 0; u < 4; ++u) std::cout << "T" << u << "->W" << m.matchL()[static_cast<std::size_t>(u)] << " ";
        std::cout << "\n";
        assert(sz == 4);

        // 一个更「缺人」的例子：3 个任务抢同一个工人
        KuhnMatching m2(3, 3);
        m2.addEdge(0, 0); m2.addEdge(1, 0); m2.addEdge(2, 0);   // 三个任务都只能找 W0
        m2.addEdge(2, 1);                                       // 只有 T2 还能找 W1
        const int sz2 = m2.solve();
        std::cout << "  另一例：T0,T1,T2 都只能找 W0，仅 T2 还能找 W1 -> 最大匹配 = " << sz2 << "\n";
        assert(sz2 == 2);                                       // 最多派出 T2->W1, 某个->W0
    }

    // ---------- 2. ★ 对拍：Kuhn == 暴力（随机小图）----------
    std::cout << "\n=== 2. ★ 随机小二分图：Kuhn 增广路 == 暴力 DP ===\n";
    {
        std::mt19937 rng(20260913);
        int tested = 0;
        for (int trial = 0; trial < 3000; ++trial) {
            const int L = 1 + static_cast<int>(rng() % 6);
            const int R = 1 + static_cast<int>(rng() % 6);
            KuhnMatching m(L, R);
            std::vector<std::vector<int>> adj(static_cast<std::size_t>(L));
            for (int u = 0; u < L; ++u)
                for (int v = 0; v < R; ++v)
                    if (rng() % 100 < 45) { m.addEdge(u, v); adj[static_cast<std::size_t>(u)].push_back(v); }
            const int got = m.solve();
            const int want = bruteForceMatching(L, R, adj);
            assert(got == want);
            ++tested;
        }
        std::cout << "  " << tested << " 个随机二分图：最大匹配与暴力 DP 完全一致 ✓\n";
    }

    // ---------- 3. ★ König 定理：最大匹配 == 最小点覆盖 ----------
    std::cout << "\n=== 3. ★ König 定理：最大匹配 = 最小点覆盖 ===\n";
    {
        std::mt19937 rng(20260914);
        int tested = 0;
        for (int trial = 0; trial < 2000; ++trial) {
            const int L = 1 + static_cast<int>(rng() % 7);
            const int R = 1 + static_cast<int>(rng() % 7);
            KuhnMatching m(L, R);
            std::vector<std::vector<int>> adj(static_cast<std::size_t>(L));
            for (int u = 0; u < L; ++u)
                for (int v = 0; v < R; ++v)
                    if (rng() % 100 < 40) { m.addEdge(u, v); adj[static_cast<std::size_t>(u)].push_back(v); }
            const int match = m.solve();
            std::vector<char> cL, cR;
            const int cover = m.minVertexCover(cL, cR);
            assert(cover == match);                                  // König：两者相等
            // 校验确实「覆盖」了每条边
            for (int u = 0; u < L; ++u)
                for (int v : adj[static_cast<std::size_t>(u)])
                    assert(cL[static_cast<std::size_t>(u)] || cR[static_cast<std::size_t>(v)]);
            ++tested;
        }
        std::cout << "  " << tested << " 个随机图：最小点覆盖大小 == 最大匹配，且确实覆盖每条边 ✓\n";
        std::cout << "  -> 推论：最大独立集 = V − 最小点覆盖 = V − 最大匹配（König）。\n";
    }

    // ---------- 4. ★ 指派问题：匈牙利算法 vs 暴力排列 ----------
    std::cout << "\n=== 4. ★ 指派问题（最小代价完美匹配）===\n";
    {
        // 一个直观例子：3 个工人 × 3 项活，代价矩阵
        std::vector<std::vector<long long>> cost = {
            {4, 1, 3},
            {2, 0, 5},
            {3, 2, 2},
        };
        const long long best = hungarian(cost);
        std::cout << "  代价矩阵：\n";
        for (const auto& row : cost) {
            std::cout << "     ";
            for (long long x : row) std::cout << x << " ";
            std::cout << "\n";
        }
        std::cout << "  最小总代价 = " << best << "（= 1 + 2 + 2：0 号工人干 1 号活、1 号干 0 号、2 号干 2 号）\n";
        assert(best == bruteForceAssignment(cost));

        // 对拍：随机 n×n（n<=7）代价矩阵
        std::mt19937 rng(20260915);
        int tested = 0;
        for (int trial = 0; trial < 400; ++trial) {
            const int n = 1 + static_cast<int>(rng() % 7);
            std::vector<std::vector<long long>> a(static_cast<std::size_t>(n), std::vector<long long>(static_cast<std::size_t>(n)));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = static_cast<long long>(rng() % 50);
            assert(hungarian(a) == bruteForceAssignment(a));
            ++tested;
        }
        std::cout << "  " << tested << " 个随机代价矩阵（n<=7）：匈牙利算法 == 枚举所有排列的最优 ✓\n";
    }

    std::cout << "\n=== 5. 复杂度与要点 ===\n"
                 "  * 二分图最大匹配：Kuhn 增广路 O(V·E)；稠密图可用 Hopcroft–Karp 优化到 O(E·√V)；\n"
                 "  * **Berge 定理**：无增广路 ⟺ 匹配最大（算法的正确性基石）；\n"
                 "  * **König 定理**：二分图里 最大匹配 = 最小点覆盖 = V − 最大独立集；\n"
                 "  * 指派问题（最小代价完美匹配）：匈牙利/Kuhn–Munkres O(n^3)，本质是最小费用流；\n"
                 "  * 最小费用流 = 「每次沿最短路增广」，与第 11 章 Dijkstra/Bellman-Ford 一脉相承；\n"
                 "  * AI 落点：任务分配、数据关联（多目标跟踪）、半监督/自监督里的样本-原型指派、\n"
                 "    最优传输（optimal transport，Wasserstein 距离的离散版）也建立在这套匹配之上。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
