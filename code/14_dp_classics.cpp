// ===========================================================================
// 14_dp_classics.cpp
// 动态规划（Dynamic Programming）四经典：
//   1) 0/1 背包：朴素递归（指数）→ 记忆化 → 递推，统计「子问题重叠」
//   2) 最长公共子序列 LCS：DP 表 + 回溯还原序列
//   3) 编辑距离（Levenshtein）：DP + 回溯还原「插入/删除/替换」操作
//   4) 最长递增子序列 LIS：O(n^2) DP vs O(n log n) 二分（patience sorting）
//
// 动态规划的三要素（贯穿全文件）：
//   (a) 最优子结构：大问题的最优解由子问题的最优解拼成；
//   (b) 重叠子问题：不同路径反复遇到同一子问题 —— 这正是「记忆化」的用武之地；
//   (c) 状态 + 转移方程 + 边界。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 14_dp_classics.cpp -o 14_dp_classics && ./14_dp_classics
// ===========================================================================
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ===========================================================================
// 1. 0/1 背包：每件物品选或不选，容量 cap，最大化价值
//    递归版会重复求解同一 (i, cap) —— 用「调用计数」亲眼看到指数爆炸
// ===========================================================================
namespace dp {

int knapRec(const std::vector<int>& w, const std::vector<int>& v,
            int i, int cap, long long& calls) {
    ++calls;                                    // 每进入一次递归就记一笔
    if (i < 0 || cap == 0) return 0;
    if (w[static_cast<std::size_t>(i)] > cap)                 // 装不下，只能不选
        return knapRec(w, v, i - 1, cap, calls);
    const int skip = knapRec(w, v, i - 1, cap, calls);
    const int take = v[static_cast<std::size_t>(i)] +
                     knapRec(w, v, i - 1, cap - w[static_cast<std::size_t>(i)], calls);
    return std::max(skip, take);
}

int knapMemo(const std::vector<int>& w, const std::vector<int>& v, int i, int cap,
             std::vector<std::vector<int>>& memo) {
    if (i < 0 || cap == 0) return 0;
    int& m = memo[static_cast<std::size_t>(i)][static_cast<std::size_t>(cap)];
    if (m >= 0) return m;                       // -1 表示未算过
    int best;
    if (w[static_cast<std::size_t>(i)] > cap) {
        best = knapMemo(w, v, i - 1, cap, memo);
    } else {
        const int skip = knapMemo(w, v, i - 1, cap, memo);
        const int take = v[static_cast<std::size_t>(i)] +
                         knapMemo(w, v, i - 1, cap - w[static_cast<std::size_t>(i)], memo);
        best = std::max(skip, take);
    }
    m = best;
    return best;
}

int knapDP(const std::vector<int>& w, const std::vector<int>& v, int cap) {  // 自底向上
    const int n = static_cast<int>(w.size());
    std::vector<int> f(static_cast<std::size_t>(cap) + 1, 0);   // f[c] = 容量 c 的最优价值
    for (int i = 0; i < n; ++i)
        for (int c = cap; c >= w[static_cast<std::size_t>(i)]; --c)   // 倒序 -> 每件只用一次
            f[static_cast<std::size_t>(c)] = std::max(
                f[static_cast<std::size_t>(c)],
                f[static_cast<std::size_t>(c - w[static_cast<std::size_t>(i)])] +
                    v[static_cast<std::size_t>(i)]);
    return f[static_cast<std::size_t>(cap)];
}

int knapBrute(const std::vector<int>& w, const std::vector<int>& v, int cap) {  // 枚举子集
    const int n = static_cast<int>(w.size());
    int best = 0;
    for (int mask = 0; mask < (1 << n); ++mask) {
        int tw = 0, tv = 0;
        for (int i = 0; i < n; ++i)
            if (mask & (1 << i)) { tw += w[static_cast<std::size_t>(i)]; tv += v[static_cast<std::size_t>(i)]; }
        if (tw <= cap) best = std::max(best, tv);
    }
    return best;
}

// ===========================================================================
// 2. 最长公共子序列 LCS
//    f[i][j] = a 前 i 个、b 前 j 个的 LCS 长度
//    a[i-1]==b[j-1]: f[i][j]=f[i-1][j-1]+1；否则 max(f[i-1][j], f[i][j-1])
// ===========================================================================
int lcsLen(const std::string& a, const std::string& b) {
    const std::size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> f(n + 1, std::vector<int>(m + 1, 0));
    for (std::size_t i = 1; i <= n; ++i)
        for (std::size_t j = 1; j <= m; ++j)
            f[i][j] = (a[i - 1] == b[j - 1])
                          ? f[i - 1][j - 1] + 1
                          : std::max(f[i - 1][j], f[i][j - 1]);
    return f[n][m];
}

std::string lcsString(const std::string& a, const std::string& b) {
    const std::size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> f(n + 1, std::vector<int>(m + 1, 0));
    for (std::size_t i = 1; i <= n; ++i)
        for (std::size_t j = 1; j <= m; ++j)
            f[i][j] = (a[i - 1] == b[j - 1]) ? f[i - 1][j - 1] + 1
                                             : std::max(f[i - 1][j], f[i][j - 1]);
    std::string s;                              // 从右下角回溯
    std::size_t i = n, j = m;
    while (i > 0 && j > 0) {
        if (a[i - 1] == b[j - 1]) { s.push_back(a[i - 1]); --i; --j; }
        else if (f[i - 1][j] >= f[i][j - 1]) --i;
        else --j;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

int lcsBrute(const std::string& a, const std::string& b, std::size_t i, std::size_t j) {
    if (i == a.size() || j == b.size()) return 0;
    if (a[i] == b[j]) return 1 + lcsBrute(a, b, i + 1, j + 1);
    return std::max(lcsBrute(a, b, i + 1, j), lcsBrute(a, b, i, j + 1));
}

// ===========================================================================
// 3. 编辑距离：把 a 改成 b 的最少操作数（插入 / 删除 / 替换各代价 1）
//    f[i][j] = a 前 i 个 -> b 前 j 个的最少操作
// ===========================================================================
int editDistance(const std::string& a, const std::string& b) {
    const std::size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> f(n + 1, std::vector<int>(m + 1, 0));
    for (std::size_t i = 0; i <= n; ++i) f[i][0] = static_cast<int>(i);   // 全删
    for (std::size_t j = 0; j <= m; ++j) f[0][j] = static_cast<int>(j);   // 全插
    for (std::size_t i = 1; i <= n; ++i)
        for (std::size_t j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) f[i][j] = f[i - 1][j - 1];          // 匹配，不花代价
            else f[i][j] = 1 + std::min({f[i - 1][j - 1],                // 替换
                                         f[i - 1][j],                    // 删 a[i-1]
                                         f[i][j - 1]});                  // 插 b[j-1]
        }
    return f[n][m];
}

std::string editScript(const std::string& a, const std::string& b) {
    const std::size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> f(n + 1, std::vector<int>(m + 1, 0));
    for (std::size_t i = 0; i <= n; ++i) f[i][0] = static_cast<int>(i);
    for (std::size_t j = 0; j <= m; ++j) f[0][j] = static_cast<int>(j);
    for (std::size_t i = 1; i <= n; ++i)
        for (std::size_t j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) f[i][j] = f[i - 1][j - 1];
            else f[i][j] = 1 + std::min({f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]});
        }
    std::string ops;
    std::size_t i = n, j = m;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && a[i - 1] == b[j - 1]) { --i; --j; }             // 保留
        else if (i > 0 && j > 0 && f[i][j] == f[i - 1][j - 1] + 1) {          // 替换
            ops += std::string("替换 ") + a[i - 1] + "->" + b[j - 1] + "\n";
            --i; --j;
        } else if (i > 0 && f[i][j] == f[i - 1][j] + 1) {                     // 删除
            ops += std::string("删除 ") + a[i - 1] + "\n";
            --i;
        } else {                                                             // 插入
            ops += std::string("插入 ") + b[j - 1] + "\n";
            --j;
        }
    }
    return ops;
}

int editBrute(const std::string& a, const std::string& b, std::size_t i, std::size_t j) {
    if (i == a.size()) return static_cast<int>(b.size() - j);   // 剩下全插
    if (j == b.size()) return static_cast<int>(a.size() - i);   // 剩下全删
    if (a[i] == b[j]) return editBrute(a, b, i + 1, j + 1);
    return 1 + std::min({editBrute(a, b, i + 1, j + 1),          // 替换
                         editBrute(a, b, i + 1, j),              // 删除
                         editBrute(a, b, i, j + 1)});            // 插入
}

// ===========================================================================
// 4. 最长递增子序列 LIS
// ===========================================================================
int lisN2(const std::vector<int>& a) {                          // O(n^2)
    const std::size_t n = a.size();
    if (n == 0) return 0;
    std::vector<int> f(n, 1);
    int best = 1;
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j)
            if (a[j] < a[i]) f[i] = std::max(f[i], f[j] + 1);
        best = std::max(best, f[i]);
    }
    return best;
}

// O(n log n)：维护 tails[k] = 「长度为 k+1 的递增子序列的最小可能结尾」
int lisNLogN(const std::vector<int>& a, std::vector<int>* out = nullptr) {
    const std::size_t n = a.size();
    if (n == 0) return 0;
    std::vector<int> tails;        // tails 单调递增
    std::vector<int> tailIdx;      // tails[k] 对应原数组下标
    std::vector<int> parent(n, -1);
    for (std::size_t i = 0; i < n; ++i) {
        auto it = std::lower_bound(tails.begin(), tails.end(), a[i]);
        const std::size_t pos = static_cast<std::size_t>(it - tails.begin());
        if (pos > 0) parent[i] = tailIdx[pos - 1];
        if (pos == tails.size()) { tails.push_back(a[i]); tailIdx.push_back(static_cast<int>(i)); }
        else { tails[pos] = a[i]; tailIdx[pos] = static_cast<int>(i); }
    }
    if (out) {                                       // 回溯还原一条 LIS
        out->clear();
        for (int k = tailIdx.back(); k != -1; k = parent[static_cast<std::size_t>(k)])
            out->push_back(a[static_cast<std::size_t>(k)]);
        std::reverse(out->begin(), out->end());
    }
    return static_cast<int>(tails.size());
}

} // namespace dp

// ===========================================================================
// 测试与打印
// ===========================================================================
static void testKnapsack() {
    std::cout << "===== 1) 0/1 背包：子问题重叠 = 指数 → 多项式 =====\n";
    // 先看「调用次数」：同一组数据，递归 vs 记忆化
    {
        std::vector<int> w = {5, 4, 6, 3, 7, 8, 2, 9, 5, 6, 4, 3, 8, 7, 2};
        std::vector<int> v = {8, 6, 9, 5, 11, 12, 3, 14, 7, 9, 6, 5, 12, 10, 4};
        const int cap = 30;
        long long calls = 0;
        const int r1 = dp::knapRec(w, v, static_cast<int>(w.size()) - 1, cap, calls);
        std::vector<std::vector<int>> memo(w.size(), std::vector<int>(cap + 1, -1));
        const int r2 = dp::knapMemo(w, v, static_cast<int>(w.size()) - 1, cap, memo);
        const int r3 = dp::knapDP(w, v, cap);
        assert(r1 == r2 && r2 == r3);
        std::cout << "  物品数 n=" << w.size() << " 容量 cap=" << cap << " 最优价值=" << r1 << "\n";
        std::cout << "  朴素递归调用次数 = " << calls
                  << "\t记忆化 / 递推的子问题数 = " << w.size() * (cap + 1) << "\n";
        std::cout << "  -> 朴素递归指数爆炸（同一 (i,cap) 反复重算）；记忆化每个子问题只算一次，\n";
        std::cout << "     状态数 n*cap 就是 O(n*cap)。\n";
    }
    // 对拍：随机小规模 vs 暴力枚举
    std::mt19937 rng(20260914);
    for (int t = 0; t < 3000; ++t) {
        const int n = 1 + static_cast<int>(rng() % 16);
        std::vector<int> w(n), v(n);
        for (int i = 0; i < n; ++i) { w[static_cast<std::size_t>(i)] = 1 + static_cast<int>(rng() % 12);
                                      v[static_cast<std::size_t>(i)] = static_cast<int>(rng() % 20); }
        const int cap = 1 + static_cast<int>(rng() % 40);
        const int a = dp::knapDP(w, v, cap);
        const int b = dp::knapBrute(w, v, cap);
        assert(a == b);
    }
    std::cout << "  [断言通过] 3000 组随机：DP == 暴力枚举\n\n";
}

static void testLCS() {
    std::cout << "===== 2) 最长公共子序列 LCS =====\n";
    const std::string a = "ABCBDAB", b = "BDCABA";
    std::cout << "  A = " << a << "\n  B = " << b << "\n";
    std::cout << "  LCS 长度 = " << dp::lcsLen(a, b) << "\t一个解 = \"" << dp::lcsString(a, b) << "\"\n";
    assert(dp::lcsLen(a, b) == dp::lcsBrute(a, b, 0, 0));
    std::mt19937 rng(7);
    auto randStr = [&](int n, int alpha) {
        std::string s;
        for (int i = 0; i < n; ++i) s.push_back(static_cast<char>('a' + rng() % static_cast<unsigned>(alpha)));
        return s;
    };
    for (int t = 0; t < 2000; ++t) {
        const std::string x = randStr(1 + static_cast<int>(rng() % 10), 1 + static_cast<int>(rng() % 3));
        const std::string y = randStr(1 + static_cast<int>(rng() % 10), 1 + static_cast<int>(rng() % 3));
        assert(dp::lcsLen(x, y) == dp::lcsBrute(x, y, 0, 0));
        assert(static_cast<int>(dp::lcsString(x, y).size()) == dp::lcsLen(x, y));   // 还原长度必须一致
    }
    std::cout << "  -> DP 表 O(n*m)；回调时「相等走对角、否则走较大者」即可还原一个 LCS。\n";
    std::cout << "  [断言通过] 2000 组随机：DP 长度 == 暴力；还原序列长度 == DP 长度\n\n";
}

static void testEdit() {
    std::cout << "===== 3) 编辑距离（Levenshtein） =====\n";
    const std::string a = "kitten", b = "sitting";
    std::cout << "  \"" << a << "\" -> \"" << b << "\" 距离 = " << dp::editDistance(a, b) << "\n";
    std::cout << "  编辑脚本：\n" << dp::editScript(a, b);
    assert(dp::editDistance(a, b) == 3);
    std::mt19937 rng(11);
    auto randStr = [&](int n, int alpha) {
        std::string s;
        for (int i = 0; i < n; ++i) s.push_back(static_cast<char>('a' + rng() % static_cast<unsigned>(alpha)));
        return s;
    };
    for (int t = 0; t < 1500; ++t) {
        const std::string x = randStr(static_cast<int>(rng() % 8), 1 + static_cast<int>(rng() % 3));
        const std::string y = randStr(static_cast<int>(rng() % 8), 1 + static_cast<int>(rng() % 3));
        assert(dp::editDistance(x, y) == dp::editBrute(x, y, 0, 0));
    }
    std::cout << "  -> 转移：相等取对角；否则 1 + min(替换, 删除, 插入)。\n";
    std::cout << "  [断言通过] 1500 组随机：DP == 暴力递归\n\n";
}

static void testLIS() {
    std::cout << "===== 4) 最长递增子序列 LIS：O(n^2) vs O(n log n) =====\n";
    {
        const std::vector<int> a = {10, 9, 2, 5, 3, 7, 101, 18};
        std::vector<int> one;
        const int len = dp::lisNLogN(a, &one);
        std::cout << "  数组 = 10 9 2 5 3 7 101 18\n";
        std::cout << "  LIS 长度 = " << len << "（O(n^2) 也给 " << dp::lisN2(a) << "），一条 LIS = ";
        for (std::size_t i = 0; i < one.size(); ++i) std::cout << one[i] << (i + 1 < one.size() ? " " : "");
        std::cout << "\n";
        assert(len == dp::lisN2(a));
    }
    std::mt19937 rng(20260913);
    for (int t = 0; t < 4000; ++t) {
        const int n = static_cast<int>(rng() % 40);
        std::vector<int> a(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = static_cast<int>(rng() % 50);
        assert(dp::lisNLogN(a) == dp::lisN2(a));
    }
    std::cout << "  -> tails 单调递增，用二分找「第一个 >= a[i]」的位置替换，O(n log n)。\n";
    std::cout << "  [断言通过] 4000 组随机：O(n log n) == O(n^2)\n\n";
}

int main() {
    std::cout << "########## 动态规划四经典 ##########\n\n";
    testKnapsack();
    testLCS();
    testEdit();
    testLIS();
    std::cout << "全部测试通过。\n";
    return 0;
}
