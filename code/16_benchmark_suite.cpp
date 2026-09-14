// ===========================================================================
// 16_benchmark_suite.cpp
// 面向世界模型研究的「微基准套件」：把全课的数据结构 / 算法接到 ML 系统上
//   1) Top-k：大小为 k 的堆 O(n log k) vs 全排序 O(n log n)
//   2) 查找：哈希表 O(1) vs 有序数组二分 O(log n)（缓存友好度之争）
//   3) kNN：KD 树剪枝 vs 暴力（空间检索 / 向量库的地基）
//   4) 矩阵乘法：朴素 ijk vs 循环换序 ikj vs 分块（缓存 / 布局决定性能）
//   5) 数值稳定：朴素 softmax 溢出 -> 减最大值的稳定版（logsumexp）
//   6) 序列解码：贪心 vs 束搜索（beam search）vs 穷举（生成 / 规划的核心）
//   7) 经验回放：环形缓冲区（固定容量 + 随机采样）的写入 / 采样吞吐
//
// 目的：让你亲眼看到「基础课的知识」如何直接决定训练 / 推理系统的性能与正确性。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 16_benchmark_suite.cpp -o 16_benchmark_suite && ./16_benchmark_suite
// ===========================================================================
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

static double msSince(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ===========================================================================
// 1. Top-k：堆 vs 全排序
// ===========================================================================
static void benchTopK() {
    std::cout << "===== 1) Top-k：大小为 k 的堆 vs 全排序 =====\n";
    const int n = 2000000, k = 100;
    std::mt19937 rng(20260913);
    std::vector<int> a(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = static_cast<int>(rng());

    std::vector<int> v1 = a;
    auto t0 = std::chrono::steady_clock::now();
    std::sort(v1.begin(), v1.end(), std::greater<int>());
    v1.resize(static_cast<std::size_t>(k));
    const double ts = msSince(t0);

    std::vector<int> v2;
    t0 = std::chrono::steady_clock::now();
    {
        std::priority_queue<int, std::vector<int>, std::greater<int>> h;   // 最小堆装 top-k
        for (int x : a) {
            if (static_cast<int>(h.size()) < k) h.push(x);
            else if (x > h.top()) { h.pop(); h.push(x); }
        }
        while (!h.empty()) { v2.push_back(h.top()); h.pop(); }
        std::reverse(v2.begin(), v2.end());
    }
    const double th = msSince(t0);
    assert(v1 == v2);
    std::cout << "  n=" << n << "，k=" << k << "\n";
    std::cout << "    全排序 O(n log n)  " << ts << " ms\n";
    std::cout << "    堆     O(n log k)  " << th << " ms   加速 " << (ts / th) << "x\n";
    std::cout << "  -> k << n 时堆完胜；流式/内存受限场景只能上堆（无法全排序）。\n\n";
}

// ===========================================================================
// 2. 查找：哈希 vs 有序数组二分
// ===========================================================================
static void benchLookup() {
    std::cout << "===== 2) 查找：unordered_map vs 有序数组二分 =====\n";
    const int n = 1000000, q = 1000000;
    std::mt19937 rng(20260914);
    std::vector<int> keys(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) keys[static_cast<std::size_t>(i)] = static_cast<int>(rng());
    std::vector<int> sorted(keys);
    std::sort(sorted.begin(), sorted.end());

    std::unordered_map<int, int> mp;
    mp.reserve(static_cast<std::size_t>(n) * 2);
    for (int i = 0; i < n; ++i) mp.emplace(keys[static_cast<std::size_t>(i)], i);

    const std::size_t hashMB = (mp.bucket_count() * sizeof(void*) + mp.size() * (sizeof(int) * 2 + sizeof(void*))) / 1048576;
    const std::size_t arrMB = sorted.size() * sizeof(int) / 1048576;
    std::cout << "  n=" << n << " 键（哈希表内存约 " << hashMB << " MB，有序数组约 " << arrMB << " MB），"
              << q << " 次查找\n";
    long long hitsM = 0, hitsB = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int x : keys) { if (mp.find(x) != mp.end()) ++hitsM; }    // 全命中：查询 = 键本身
    const double tm = msSince(t0);
    t0 = std::chrono::steady_clock::now();
    for (int x : keys) { if (std::binary_search(sorted.begin(), sorted.end(), x)) ++hitsB; }
    const double tb = msSince(t0);
    assert(hitsM == hitsB && hitsM == n);                          // 两种实现必须命中同样多的键
    std::cout << "    unordered_map（哈希）  " << tm << " ms   命中 " << hitsM
              << "   平均 " << (tm * 1e6 / q) << " ns/次\n";
    std::cout << "    有序数组二分            " << tb << " ms   命中 " << hitsB
              << "   平均 " << (tb * 1e6 / q) << " ns/次（≈ log2(" << n << ")≈20 次比较）\n";
    std::cout << "  -> 点查询哈希更快（O(1) vs O(log n)），代价是内存约 " << (hashMB / std::max<std::size_t>(1, arrMB))
              << " 倍 + 跳指针缓存不友好。\n";
    std::cout << "     有序数组的赢面在别处：零额外内存、数据本就有序、需要**范围查询**、\n";
    std::cout << "     或顺序扫描 —— 缓存友好度把「每步 20 次比较」的劣势部分抵消回来。\n\n";
}

// ===========================================================================
// 3. kNN：KD 树 vs 暴力
// ===========================================================================
namespace kd {
struct Node { std::array<double, 2> p; int left = -1, right = -1, axis = 0; };

struct Tree {
    std::vector<Node> nodes;
    int build(std::vector<std::array<double, 2>>& pts, int lo, int hi, int depth) {
        if (lo >= hi) return -1;
        const int axis = depth & 1;
        const int mid = lo + (hi - lo) / 2;
        std::nth_element(pts.begin() + lo, pts.begin() + mid, pts.begin() + hi,
                         [axis](const std::array<double, 2>& a, const std::array<double, 2>& b) { return a[static_cast<std::size_t>(axis)] < b[static_cast<std::size_t>(axis)]; });
        const int id = static_cast<int>(nodes.size());
        nodes.push_back(Node{});
        nodes[static_cast<std::size_t>(id)].p = pts[static_cast<std::size_t>(mid)];
        nodes[static_cast<std::size_t>(id)].axis = axis;
        const int l = build(pts, lo, mid, depth + 1);
        const int r = build(pts, mid + 1, hi, depth + 1);
        nodes[static_cast<std::size_t>(id)].left = l;
        nodes[static_cast<std::size_t>(id)].right = r;
        return id;
    }
    void build(std::vector<std::array<double, 2>> pts) { nodes.reserve(pts.size()); build(pts, 0, static_cast<int>(pts.size()), 0); }

    void query(int id, const std::array<double, 2>& q, int k,
               std::priority_queue<std::pair<double, int>>& heap, long long& visited) const {
        if (id < 0) return;
        ++visited;
        const Node& nd = nodes[static_cast<std::size_t>(id)];
        const int ax = nd.axis;
        const double d = std::hypot(nd.p[0] - q[0], nd.p[1] - q[1]);
        if (static_cast<int>(heap.size()) < k) heap.push({d, id});
        else if (d < heap.top().first) { heap.pop(); heap.push({d, id}); }
        const double diff = q[static_cast<std::size_t>(ax)] - nd.p[static_cast<std::size_t>(ax)];
        const int near = diff < 0 ? nd.left : nd.right;
        const int far = diff < 0 ? nd.right : nd.left;
        query(near, q, k, heap, visited);
        // 只有「切分面距离 < 当前第 k 近」时才需要搜另一侧
        if (heap.size() < static_cast<std::size_t>(k) || std::abs(diff) < heap.top().first)
            query(far, q, k, heap, visited);
    }
};
} // namespace kd

static void benchKNN() {
    std::cout << "===== 3) kNN：KD 树剪枝 vs 暴力（并验证结果一致） =====\n";
    const int n = 200000, k = 10, qn = 5000;
    std::mt19937 rng(20260915);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::array<double, 2>> pts(static_cast<std::size_t>(n));
    for (auto& p : pts) p = {u(rng), u(rng)};
    std::vector<std::array<double, 2>> qs(static_cast<std::size_t>(qn));
    for (auto& p : qs) p = {u(rng), u(rng)};

    kd::Tree tree;
    tree.build(pts);
    const auto t0 = std::chrono::steady_clock::now();
    long long visited = 0;
    double acc = 0;
    for (const auto& q : qs) {
        std::priority_queue<std::pair<double, int>> heap;
        tree.query(0, q, k, heap, visited);
        while (!heap.empty()) acc += heap.top().first, heap.pop();
    }
    const double tkd = msSince(t0);

    const auto t1 = std::chrono::steady_clock::now();
    double acc2 = 0;
    for (const auto& q : qs) {
        std::vector<double> dists;
        dists.reserve(static_cast<std::size_t>(n));
        for (const auto& p : pts) dists.push_back(std::hypot(p[0] - q[0], p[1] - q[1]));
        std::nth_element(dists.begin(), dists.begin() + k, dists.end());
        for (int i = 0; i < k; ++i) acc2 += dists[static_cast<std::size_t>(i)];
    }
    const double tbrute = msSince(t1);
    const double denom = std::min(tkd, tbrute);
    std::cout << "  n=" << n << " 点（2D），" << qn << " 次查询，k=" << k << "\n";
    std::cout << "    KD 树     " << tkd << " ms   （平均访问结点 " << (visited / qn) << " / " << n << "）\n";
    std::cout << "    暴力       " << tbrute << " ms\n";
    std::cout << "    两法前 k 距离和之比 ≈ " << (acc / acc2) << "（应为 1，验证结果一致）\n";
    assert(std::abs(acc - acc2) < 1e-6 * (1.0 + std::abs(acc2)) + 1e-9);
    (void)denom;
    std::cout << "  -> KD 树只访问约 " << (100.0 * (visited / qn) / n) << "% 的点，检索快一个量级；\n";
    std::cout << "     维度升高会「维度灾难」，于是有了 LSH / PQ / HNSW（向量库）。\n\n";
}

// ===========================================================================
// 4. 矩阵乘法：循环换序 / 分块
// ===========================================================================
static void benchMatmul() {
    std::cout << "===== 4) 矩阵乘法：内存布局决定性能 =====\n";
    const int n = 512;
    std::mt19937 rng(20260916);
    std::vector<double> A(static_cast<std::size_t>(n) * n), B(static_cast<std::size_t>(n) * n),
        C1(static_cast<std::size_t>(n) * n, 0), C2(static_cast<std::size_t>(n) * n, 0), C3(static_cast<std::size_t>(n) * n, 0);
    for (auto& x : A) x = static_cast<double>(rng() % 1000) / 1000.0;
    for (auto& x : B) x = static_cast<double>(rng() % 1000) / 1000.0;
    const std::size_t N = static_cast<std::size_t>(n);
    auto idx = [N](int i, int j) { return static_cast<std::size_t>(i) * N + static_cast<std::size_t>(j); };

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i)                    // naive ijk：B 的访问跨行，缓存差
        for (int j = 0; j < n; ++j) {
            double s = 0;
            for (int k = 0; k < n; ++k) s += A[idx(i, k)] * B[idx(k, j)];
            C1[idx(i, j)] = s;
        }
    const double t1 = msSince(t0);

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i)                    // ikj：内层连续扫 B 的一行，缓存友好
        for (int k = 0; k < n; ++k) {
            const double a = A[idx(i, k)];
            for (int j = 0; j < n; ++j) C2[idx(i, j)] += a * B[idx(k, j)];
        }
    const double t2 = msSince(t0);

    t0 = std::chrono::steady_clock::now();
    const int BLK = 64;                            // 分块：把工作集压进 cache
    for (int ii = 0; ii < n; ii += BLK)
        for (int kk = 0; kk < n; kk += BLK)
            for (int jj = 0; jj < n; jj += BLK)
                for (int i = ii; i < std::min(ii + BLK, n); ++i)
                    for (int k = kk; k < std::min(kk + BLK, n); ++k) {
                        const double a = A[idx(i, k)];
                        for (int j = jj; j < std::min(jj + BLK, n); ++j) C3[idx(i, j)] += a * B[idx(k, j)];
                    }
    const double t3 = msSince(t0);

    for (std::size_t t = 0; t < C1.size(); ++t) { assert(std::abs(C1[t] - C2[t]) < 1e-6); assert(std::abs(C1[t] - C3[t]) < 1e-6); }
    const double flop = 2.0 * static_cast<double>(n) * n * n;
    std::cout << "  n=" << n << "，浮点乘加量 ≈ " << (flop / 1e6) << " MFLOP\n";
    std::cout << "    朴素 ijk     " << t1 << " ms   " << (flop / (t1 * 1e6)) << " GFLOP/s\n";
    std::cout << "    换序 ikj     " << t2 << " ms   " << (flop / (t2 * 1e6)) << " GFLOP/s\n";
    std::cout << "    分块(BLK=64) " << t3 << " ms   " << (flop / (t3 * 1e6)) << " GFLOP/s\n";
    std::cout << "  -> 三种写法**算法完全一样**（都是 O(n^3)），性能却差数倍，全在缓存命中率。\n";
    std::cout << "     深度学习框架的矩阵乘（cuBLAS）、FlashAttention 都在做这件事。\n\n";
}

// ===========================================================================
// 5. 数值稳定：朴素 softmax vs 减最大值的稳定版
// ===========================================================================
static std::vector<double> softmaxNaive(const std::vector<double>& z) {
    std::vector<double> e(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) e[i] = std::exp(z[i]);      // 大 logits 直接溢出
    double s = 0;
    for (double x : e) s += x;
    for (double& x : e) x /= s;                                            // inf/inf -> nan
    return e;
}
static std::vector<double> softmaxStable(const std::vector<double>& z) {
    const double m = *std::max_element(z.begin(), z.end());
    std::vector<double> e(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) e[i] = std::exp(z[i] - m);  // 平移不变，先减最大值
    double s = 0;
    for (double x : e) s += x;
    for (double& x : e) x /= s;
    return e;
}

static void benchSoftmax() {
    std::cout << "===== 5) 数值稳定：softmax 的溢出与平移不变性 =====\n";
    std::vector<double> z = {1000.0, 1001.0, 999.0, 1002.0, 998.0};
    // 参考值：用均匀平移 -1000（结果应与 stable 完全一致）
    std::vector<double> zc(z);
    for (double& x : zc) x -= 1000.0;
    const std::vector<double> ref = softmaxNaive(zc);          // 平移后不溢出，作为参考
    const std::vector<double> st = softmaxStable(z);
    const std::vector<double> nv = softmaxNaive(z);            // 未平移 -> 溢出

    std::cout << "  logits = 1000 1001 999 1002 998\n";
    std::cout << "    朴素 softmax  : ";
    for (double x : nv) { if (std::isnan(x)) std::cout << "nan "; else std::cout << x << " "; }
    std::cout << "  <- exp(1000)=inf -> inf/inf=nan\n";
    std::cout << "    稳定 softmax  : ";
    for (double x : st) std::cout << x << " ";
    std::cout << "\n";
    std::cout << "    参考(平移后)  : ";
    for (double x : ref) std::cout << x << " ";
    std::cout << "\n";
    for (std::size_t i = 0; i < z.size(); ++i) assert(std::abs(st[i] - ref[i]) < 1e-12);
    double sum = 0; for (double x : st) sum += x;
    assert(std::abs(sum - 1.0) < 1e-12);
    bool overflowed = false; for (double x : nv) if (std::isnan(x) || std::isinf(x)) overflowed = true;
    assert(overflowed);
    // logsumexp 恒等式
    double lse = 1002.0 + std::log(std::exp(-2.0) + std::exp(-1.0) + std::exp(-3.0) + std::exp(0.0) + std::exp(-4.0));
    std::cout << "    logsumexp(logits) = " << lse << "（= max + log Σexp(z-max)）\n";
    std::cout << "  -> softmax 平移不变，但**必须在数值上减去最大值**，否则 exp 溢出成 nan。\n";
    std::cout << "     这就是注意力 / 交叉熵实现里处处 logsumexp 的原因。\n\n";
}

// ===========================================================================
// 6. 序列解码：贪心 vs 束搜索 vs 穷举
// ===========================================================================
namespace decode {
// 逐步打分：路径得分 = Σ_t ( trans[prev][tok] + base[t][tok] )
struct Model {
    int K, T;
    std::vector<std::vector<double>> base;   // base[t][tok]
    std::vector<std::vector<double>> trans;  // trans[prevTok][tok]
};

double scoreOf(const Model& m, const std::vector<int>& seq) {
    double s = 0;
    int prev = -1;
    for (int t = 0; t < m.T; ++t) {
        const int tok = seq[static_cast<std::size_t>(t)];
        s += m.base[static_cast<std::size_t>(t)][static_cast<std::size_t>(tok)];
        if (prev >= 0) s += m.trans[static_cast<std::size_t>(prev)][static_cast<std::size_t>(tok)];
        prev = tok;
    }
    return s;
}

std::vector<int> greedy(const Model& m) {
    std::vector<int> seq;
    int prev = -1;
    for (int t = 0; t < m.T; ++t) {
        int best = 0; double bs = -1e18;
        for (int tok = 0; tok < m.K; ++tok) {
            double s = m.base[static_cast<std::size_t>(t)][static_cast<std::size_t>(tok)];
            if (prev >= 0) s += m.trans[static_cast<std::size_t>(prev)][static_cast<std::size_t>(tok)];
            if (s > bs) { bs = s; best = tok; }
        }
        seq.push_back(best);
        prev = best;
    }
    return seq;
}

// 束搜索：每步保留得分最高的 width 条「部分路径」
std::vector<int> beam(const Model& m, int width) {
    std::vector<std::vector<int>> beams;
    std::vector<double> scores;
    beams.push_back({});
    scores.push_back(0.0);
    for (int t = 0; t < m.T; ++t) {
        std::vector<std::pair<double, std::vector<int>>> cand;
        for (std::size_t bi = 0; bi < beams.size(); ++bi)
            for (int tok = 0; tok < m.K; ++tok) {
                double s = scores[bi] + m.base[static_cast<std::size_t>(t)][static_cast<std::size_t>(tok)];
                if (!beams[bi].empty()) s += m.trans[static_cast<std::size_t>(beams[bi].back())][static_cast<std::size_t>(tok)];
                std::vector<int> ns = beams[bi];
                ns.push_back(tok);
                cand.push_back({s, std::move(ns)});
            }
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        if (static_cast<int>(cand.size()) > width) cand.resize(static_cast<std::size_t>(width));
        beams.clear(); scores.clear();
        for (auto& c : cand) { beams.push_back(c.second); scores.push_back(c.first); }
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < scores.size(); ++i) if (scores[i] > scores[best]) best = i;
    return beams[best];
}

double exhaustive(const Model& m) {
    double best = -1e18;
    std::vector<int> seq(static_cast<std::size_t>(m.T));
    long long total = 1;
    for (int t = 0; t < m.T; ++t) total *= m.K;
    for (long long code = 0; code < total; ++code) {
        long long c = code;
        for (int t = 0; t < m.T; ++t) { seq[static_cast<std::size_t>(t)] = static_cast<int>(c % m.K); c /= m.K; }
        best = std::max(best, scoreOf(m, seq));
    }
    return best;
}
} // namespace decode

static void benchDecode() {
    std::cout << "===== 6) 序列解码：贪心 vs 束搜索 vs 穷举 =====\n";
    std::mt19937 rng(20260917);
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    // 找一个「贪心不是最优」的实例，最能说明问题
    decode::Model found;
    bool ok = false;
    for (int seed = 0; seed < 100000 && !ok; ++seed) {
        std::mt19937 r(static_cast<unsigned>(seed));
        decode::Model m;
        m.K = 4; m.T = 6;
        m.base.assign(static_cast<std::size_t>(m.T), std::vector<double>(static_cast<std::size_t>(m.K)));
        m.trans.assign(static_cast<std::size_t>(m.K), std::vector<double>(static_cast<std::size_t>(m.K)));
        for (auto& row : m.base) for (auto& x : row) x = u(r);
        for (auto& row : m.trans) for (auto& x : row) x = u(r);
        const double g = decode::scoreOf(m, decode::greedy(m));
        const double ex = decode::exhaustive(m);
        if (g < ex - 1e-9) { found = m; ok = true; }
    }
    assert(ok);
    const double g = decode::scoreOf(found, decode::greedy(found));
    const double b1 = decode::scoreOf(found, decode::beam(found, 1));
    const double b2 = decode::scoreOf(found, decode::beam(found, 2));
    const double bK = decode::scoreOf(found, decode::beam(found, found.K));
    const double ex = decode::exhaustive(found);
    assert(std::abs(b1 - g) < 1e-9);              // 宽 1 的束搜索 == 贪心
    assert(b2 >= g - 1e-9 && bK >= b2 - 1e-9);    // 束越宽解越好（单调）
    assert(bK <= ex + 1e-9);                      // 不可能超过穷举
    std::cout << "  K=4 个候选 token，解码长度 T=6（穷举 4^6=4096 条路径）\n";
    std::cout << "    贪心（=束宽1）  得分 " << g << "\n";
    std::cout << "    束搜索 宽2      得分 " << b2 << "\n";
    std::cout << "    束搜索 宽4      得分 " << bK << "\n";
    std::cout << "    穷举（精确最优） 得分 " << ex << "\n";
    std::cout << "  -> 贪心每步只看眼前，会错过全局最优；束搜索保留 B 条部分路径，用「B 倍开销」换质量；\n";
    std::cout << "     B 大到 K^T 就退化成穷举。这是生成式模型 / 世界模型规划的标准解码策略。\n\n";
}

// ===========================================================================
// 7. 经验回放：环形缓冲区
// ===========================================================================
namespace rb {
struct Replay {
    std::vector<std::array<float, 4>> buf;   // 每条经验 = 4 个 float
    std::size_t cap, head = 0, size = 0;
    explicit Replay(std::size_t c) : buf(c), cap(c) {}
    void push(const std::array<float, 4>& e) {
        buf[head] = e;
        head = (head + 1) % cap;
        if (size < cap) ++size;
    }
    const std::array<float, 4>& at(std::size_t i) const { return buf[(head + cap - size + i) % cap]; }
};
} // namespace rb

static void benchReplay() {
    std::cout << "===== 7) 经验回放：环形缓冲区 =====\n";
    const std::size_t cap = 1u << 20;        // 容量 100 万条
    rb::Replay rp(cap);
    const int pushes = 3000000;
    std::mt19937 rng(20260918);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < pushes; ++i)
        rp.push({static_cast<float>(i), static_cast<float>(rng()), 1.0f, 0.0f});
    const double tp = msSince(t0);

    t0 = std::chrono::steady_clock::now();
    double acc = 0;
    const int batches = 100000;
    for (int b = 0; b < batches; ++b)
        for (int j = 0; j < 32; ++j) acc += rp.at(rng() % rp.size)[0];
    const double ts = msSince(t0);

    assert(rp.size == cap);                                   // 写满后保持容量
    assert(rp.at(rp.size - 1)[0] == static_cast<float>(pushes - 1));   // 最新一条
    std::cout << "  容量 = " << cap << "，写入 " << pushes << " 条\n";
    std::cout << "    写入        " << tp << " ms   （" << (pushes / (tp / 1000.0) / 1e6) << " M 条/s）\n";
    std::cout << "    随机采样     " << ts << " ms   （" << batches << " 批 × 32 条，acc=" << acc << "）\n";
    std::cout << "  -> 环形缓冲区写入 O(1)、内存有界（旧经验被覆盖），是 RL / 世界模型数据管线的标配。\n";
    std::cout << "     底层就是第 06 章的「循环队列」。\n\n";
}

int main() {
    std::cout << "########## 面向世界模型研究的微基准套件 ##########\n\n";
    benchTopK();
    benchLookup();
    benchKNN();
    benchMatmul();
    benchSoftmax();
    benchDecode();
    benchReplay();
    std::cout << "全部基准完成，所有正确性断言通过。\n";
    return 0;
}
