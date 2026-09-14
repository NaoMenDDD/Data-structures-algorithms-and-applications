// 11_union_find.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 11 章 图算法》
// 主题：并查集（Union-Find / Disjoint Set Union, DSU）—— 「近 O(1)」的动态连通性
//
//   问题：维护一堆「等价类」。两个操作：
//     * find(x)：x 属于哪个集合（返回代表元）；
//     * unite(x, y)：把 x、y 所在的集合合并。
//   应用：判「两点是否连通」（Kruskal 建 MST）、数连通分量、判环、
//         社交网络「朋友圈」、图像连通区域、等价类合并……
//
//   两种优化，一次拥有：
//     ① 按秩合并（union by rank）：合并时把「矮树」挂到「高树」上，
//        避免树越来越高；
//     ② 路径压缩（path compression）：find 时把一路上所有结点直接挂到根，
//        让树越用越扁。
//
//   两个优化叠加后，m 次操作的**总**代价是 O(m·α(n))，其中 α 是
//   **反阿克曼函数** —— 它增长极慢，对任何能想象的 n（< 宇宙原子数），
//   α(n) ≤ 4。所以说「近 O(1)」。
//
//   本文件实测「不用优化 / 只用按秩 / 只用压缩 / 两者都用」的差别。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 11_union_find.cpp -o 11_union_find && ./11_union_find
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 11_union_find.cpp -o uf_san && ./uf_san
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 可配置的并查集：两个开关分别控制「按秩合并」「路径压缩」，
// 方便对比四种组合的代价。hops_ 统计所有 find 累积走过的父指针跳数。
// ---------------------------------------------------------------------------
class DSU {
public:
    DSU(int n, bool useRank, bool useCompress)
        : parent_(static_cast<std::size_t>(n)), rank_(static_cast<std::size_t>(n), 0),
          useRank_(useRank), useCompress_(useCompress) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int x) {
        if (useCompress_) {
            int root = x;
            while (parent_[static_cast<std::size_t>(root)] != root) {          // 先走到根
                root = parent_[static_cast<std::size_t>(root)];
                ++hops_;
            }
            while (parent_[static_cast<std::size_t>(x)] != root) {             // 再把整条路径直接挂到根
                const int nxt = parent_[static_cast<std::size_t>(x)];
                parent_[static_cast<std::size_t>(x)] = root;
                x = nxt;
                ++hops_;
            }
            return root;
        }
        while (parent_[static_cast<std::size_t>(x)] != x) {                    // 不压缩：老实往上爬
            x = parent_[static_cast<std::size_t>(x)];
            ++hops_;
        }
        return x;
    }

    bool unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;
        if (useRank_) {
            if (rank_[static_cast<std::size_t>(ra)] < rank_[static_cast<std::size_t>(rb)]) std::swap(ra, rb);
            parent_[static_cast<std::size_t>(rb)] = ra;                        // 矮树挂到高树
            if (rank_[static_cast<std::size_t>(ra)] == rank_[static_cast<std::size_t>(rb)]) ++rank_[static_cast<std::size_t>(ra)];
        } else {
            parent_[static_cast<std::size_t>(rb)] = ra;                        // 不按秩：可能长出长链
        }
        return true;
    }

    bool connected(int a, int b) { return find(a) == find(b); }

    int countRoots() {
        int c = 0;
        for (int i = 0; i < static_cast<int>(parent_.size()); ++i) if (find(i) == i) ++c;
        return c;
    }

    long long hops() const { return hops_; }
    void resetHops() { hops_ = 0; }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
    bool useRank_;
    bool useCompress_;
    long long hops_ = 0;
};

// 度量：先做 m 次随机合并，再统计 m 次随机查询累积的父指针跳数。
static long long totalFindSteps(int n, int m, bool useRank, bool useCompress, unsigned seed) {
    DSU dsu(n, useRank, useCompress);
    std::mt19937 rng(seed);
    for (int i = 0; i < m; ++i)
        dsu.unite(static_cast<int>(rng() % static_cast<unsigned>(n)),
                  static_cast<int>(rng() % static_cast<unsigned>(n)));
    dsu.resetHops();
    for (int i = 0; i < m; ++i)
        dsu.find(static_cast<int>(rng() % static_cast<unsigned>(n)));
    return dsu.hops();
}

int main() {
    std::cout << "======== 11 并查集（Union-Find）========\n\n";

    // ---------- 1. 基本功能 ----------
    std::cout << "=== 1. 合并 / 查询 / 连通分量计数 ===\n";
    {
        DSU dsu(10, true, true);
        dsu.unite(0, 1); dsu.unite(1, 2);          // {0,1,2}
        dsu.unite(3, 4);                           // {3,4}
        dsu.unite(5, 6); dsu.unite(6, 7);         // {5,6,7}
        dsu.unite(7, 8);                          // {5,6,7,8}
        std::cout << "  合并后：\n";
        std::cout << "     0 与 2 连通? " << dsu.connected(0, 2) << "（应 1）\n";
        std::cout << "     0 与 3 连通? " << dsu.connected(0, 3) << "（应 0）\n";
        std::cout << "     5 与 8 连通? " << dsu.connected(5, 8) << "（应 1）\n";
        std::cout << "     连通分量数 = " << dsu.countRoots() << "（应 4：{0,1,2}{3,4}{5,6,7,8}{9}）\n";
        assert(dsu.connected(0, 2) && !dsu.connected(0, 3) && dsu.connected(5, 8));
        assert(dsu.countRoots() == 4);
        assert(!dsu.unite(0, 1));                  // 已连通 -> 返回 false（正是 Kruskal 判环的依据）
        std::cout << "  unite(0,1) 返回 " << dsu.unite(0, 1) << "（已连通 -> false，Kruskal 靠它判环）✓\n";
    }

    // ---------- 2. 与朴素连通性对拍 ----------
    std::cout << "\n=== 2. 与 BFS 连通性对拍（随机图）===\n";
    {
        std::mt19937 rng(20260913);
        int tested = 0;
        for (int trial = 0; trial < 200; ++trial) {
            const int n = 1 + static_cast<int>(rng() % 60);
            DSU dsu(n, true, true);
            // 邻接表，用于 BFS 参考实现
            std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                const int a = static_cast<int>(rng() % static_cast<unsigned>(n));
                const int b = static_cast<int>(rng() % static_cast<unsigned>(n));
                if (a != b) { dsu.unite(a, b); adj[static_cast<std::size_t>(a)].push_back(b); adj[static_cast<std::size_t>(b)].push_back(a); }
            }
            // BFS 数分量
            std::vector<char> vis(static_cast<std::size_t>(n), 0);
            int comps = 0;
            for (int s = 0; s < n; ++s) {
                if (vis[static_cast<std::size_t>(s)]) continue;
                ++comps;
                std::vector<int> stk{s}; vis[static_cast<std::size_t>(s)] = 1;
                while (!stk.empty()) {
                    const int u = stk.back(); stk.pop_back();
                    for (int w : adj[static_cast<std::size_t>(u)]) if (!vis[static_cast<std::size_t>(w)]) { vis[static_cast<std::size_t>(w)] = 1; stk.push_back(w); }
                }
            }
            assert(dsu.countRoots() == comps);     // 并查集分量数 == BFS 分量数
            // 随机抽查若干对的可达性
            for (int q = 0; q < 20; ++q) {
                const int a = static_cast<int>(rng() % static_cast<unsigned>(n));
                const int b = static_cast<int>(rng() % static_cast<unsigned>(n));
                // 参考：BFS 可达性
                std::vector<char> v2(static_cast<std::size_t>(n), 0);
                std::vector<int> st{a}; v2[static_cast<std::size_t>(a)] = 1;
                while (!st.empty()) { const int u = st.back(); st.pop_back(); for (int w : adj[static_cast<std::size_t>(u)]) if (!v2[static_cast<std::size_t>(w)]) { v2[static_cast<std::size_t>(w)] = 1; st.push_back(w); } }
                assert(dsu.connected(a, b) == (v2[static_cast<std::size_t>(b)] != 0));
            }
            ++tested;
        }
        std::cout << "  " << tested << " 个随机图：并查集分量数 == BFS，连通性抽查全对 ✓\n";
    }

    // ---------- 3. ★ 两种优化的威力 ----------
    std::cout << "\n=== 3. ★ 四种组合的总 find 步数（n = 100,000，m = 200,000）===\n";
    {
        const int n = 100000, m = 200000;
        std::cout << "     配置                       总 find 步数\n";
        std::cout << "     无优化（裸挂）              " << totalFindSteps(n, m, false, false, 1) << "\n";
        std::cout << "     只用按秩合并                " << totalFindSteps(n, m, true,  false, 1) << "\n";
        std::cout << "     只用路径压缩                " << totalFindSteps(n, m, false, true,  1) << "\n";
        std::cout << "     两者都用（推荐）            " << totalFindSteps(n, m, true,  true,  1) << "\n";
        std::cout << "  -> 两种优化把「可能 O(n) 的长链查找」压成「近 O(1)」；\n";
        std::cout << "     只用其一时，压缩或按秩都能大幅改善；两者叠加最稳。\n";
    }

    std::cout << "\n=== 4. 复杂度与要点 ===\n"
                 "  * find / unite 均摊代价 O(α(n))，α = 反阿克曼，α(n) <= 4（实际常数级）；\n"
                 "  * 关键：**按秩合并 + 路径压缩** 缺一不可，尤其对最坏输入；\n"
                 "  * Kruskal 用它判环（unite 返回 false 即有环）；\n"
                 "  * O(α(n)) 的紧界由 Tarjan (1975) 证明；α 增长极慢，实际就是常数；\n"
                 "  * 不能「分裂 / 撤销」—— 并查集只支持合并，不支持删除（要支持得用别的结构）。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
