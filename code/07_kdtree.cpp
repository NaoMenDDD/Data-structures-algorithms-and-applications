// 07_kdtree.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 07 章 树与二叉树》
// 主题：KD-树（k-dimensional tree）—— 二叉树在「空间检索」里的明星应用
//
//   普通二叉树按「值的比较」分叉（左小右大）；KD-树按「某一维坐标」分叉：
//   每一层轮流选一个维度作为切分轴，把空间切成两半，递归下去。
//   于是树就变成了一棵「空间划分树」，能在高维空间里做**最近邻检索**。
//
//   本文件演示：
//     1. 构造：按当前层维度取中位数切分（nth_element），保证树平衡，O(n log n)；
//     2. 1-最近邻（1-NN）：先走"近"的那一侧，再用「到切分面的距离」剪枝；
//     3. k-最近邻（k-NN）：用一个大小为 k 的最大堆维护当前候选；
//     4. 正确性：与暴力 O(N) 逐一比对（随机对拍）；
//     5. 剪枝的效率：统计访问结点数，KD-树远小于 N；
//     6. ★ 维度灾难（curse of dimensionality）：维度一高，剪枝迅速失效，
//        KD-树退化成暴力搜索 —— 这正是「近似最近邻（ANN）/向量数据库」存在的理由。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 07_kdtree.cpp -o 07_kdtree && ./07_kdtree
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 07_kdtree.cpp -o kd_san && ./kd_san
// ---------------------------------------------------------------------------

#include <algorithm>    // std::nth_element, std::sort
#include <cassert>
#include <cmath>        // std::sqrt
#include <cstddef>
#include <cstdint>      // std::uint64_t
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

using Point = std::vector<double>;   // 维度运行时决定

// 确定性伪随机（LCG + 混合），保证结果可复现
class Rng {
public:
    explicit Rng(std::uint64_t seed) : s_(seed) {}
    std::uint64_t next() {
        s_ = s_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return s_;
    }
    double unit() { return double((next() >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53); }
private:
    std::uint64_t s_;
};

static double dist2(const Point& a, const Point& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i]; s += d * d; }
    return s;
}

// ===========================================================================
// KD-树结点
// ===========================================================================
struct KdNode {
    Point  pt;
    int    axis;                  // 本结点按哪一维切分
    KdNode *left = nullptr, *right = nullptr;
    explicit KdNode(Point p, int a) : pt(std::move(p)), axis(a) {}
};

// 按 depth 轮换维度，取当前区间的**中位数**作切分点（nth_element，均摊 O(n)）。
// 取中位数保证树高 ~log2(n)，从而每次查询期望 O(log n)。
static KdNode* build(std::vector<Point>& pts, int lo, int hi, int depth, int dim) {
    if (lo >= hi) return nullptr;
    const int axis = depth % dim;
    const int mid  = lo + (hi - lo) / 2;
    std::nth_element(pts.begin() + lo, pts.begin() + mid, pts.begin() + hi,
                     [axis](const Point& a, const Point& b) { return a[axis] < b[axis]; });
    KdNode* node = new KdNode(pts[mid], axis);
    node->left  = build(pts, lo, mid, depth + 1, dim);
    node->right = build(pts, mid + 1, hi, depth + 1, dim);
    return node;
}

static void destroy(KdNode* r) {
    if (!r) return;
    destroy(r->left);
    destroy(r->right);
    delete r;
}

// ===========================================================================
// 1-最近邻：best 里维护当前最小距离与对应点
// ===========================================================================
struct NnResult {
    double dist2 = std::numeric_limits<double>::infinity();
    const Point* pt = nullptr;
    long long visited = 0;        // 访问过的结点数（衡量剪枝效果）
};

static void nnSearch(KdNode* node, const Point& target, NnResult& best) {
    if (!node) return;
    ++best.visited;

    const double d = dist2(node->pt, target);
    if (d < best.dist2) { best.dist2 = d; best.pt = &node->pt; }

    const int    axis = node->axis;
    const double diff = target[axis] - node->pt[axis];
    KdNode* nearChild = diff < 0 ? node->left  : node->right;
    KdNode* farChild  = diff < 0 ? node->right : node->left;

    nnSearch(nearChild, target, best);              // 先走"近"侧
    // 剪枝：若目标到切分面的距离已达不到当前最优，则"远"侧不可能更近
    if (diff * diff < best.dist2) nnSearch(farChild, target, best);
}

static NnResult nearest(KdNode* root, const Point& target) {
    NnResult best;
    nnSearch(root, target, best);
    return best;
}

// ===========================================================================
// k-最近邻：用大小为 k 的**最大堆**维护候选
//   * 堆未满 k：直接塞；
//   * 堆已满：若新点比堆顶（当前第 k 近）更近，替换堆顶。
//   堆顶距离就是当前"第 k 近距离"，作为剪枝半径。
// ===========================================================================
struct Cand {
    double dist2;
    const Point* pt;
    bool operator<(const Cand& o) const { return dist2 < o.dist2; }   // 堆按此比较 -> 最大堆
};

static void knnSearch(KdNode* node, const Point& target, std::size_t k,
                      std::vector<Cand>& heap, long long& visited) {
    if (!node) return;
    ++visited;

    const double d = dist2(node->pt, target);
    if (heap.size() < k) {
        heap.push_back({d, &node->pt});
        std::push_heap(heap.begin(), heap.end());
    } else if (d < heap.front().dist2) {                 // 比当前最差候选更近
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = {d, &node->pt};
        std::push_heap(heap.begin(), heap.end());
    }

    const int    axis = node->axis;
    const double diff = target[axis] - node->pt[axis];
    KdNode* nearChild = diff < 0 ? node->left  : node->right;
    KdNode* farChild  = diff < 0 ? node->right : node->left;

    knnSearch(nearChild, target, k, heap, visited);

    // 剪枝半径 = 第 k 近距离；堆未满则必须搜索远侧
    const double radius = (heap.size() < k) ? std::numeric_limits<double>::infinity()
                                            : heap.front().dist2;
    if (diff * diff < radius) knnSearch(farChild, target, k, heap, visited);
}

static std::vector<Cand> kNearest(KdNode* root, const Point& target, std::size_t k, long long& visited) {
    std::vector<Cand> heap;
    heap.reserve(k);
    visited = 0;
    knnSearch(root, target, k, heap, visited);
    std::sort(heap.begin(), heap.end());   // 按距离升序输出
    return heap;
}

// ===========================================================================
// 暴力基准
// ===========================================================================
static double bruteForceNn(const std::vector<Point>& pts, const Point& q) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& p : pts) best = std::min(best, dist2(p, q));
    return best;
}

static std::vector<double> bruteForceKnn(const std::vector<Point>& pts, const Point& q, std::size_t k) {
    std::vector<double> ds;
    ds.reserve(pts.size());
    for (const auto& p : pts) ds.push_back(dist2(p, q));
    std::sort(ds.begin(), ds.end());
    ds.resize(std::min(k, ds.size()));
    return ds;
}

// 生成 n 个 dim 维随机点（每个坐标 [0,1)）
static std::vector<Point> genPoints(std::size_t n, int dim, std::uint64_t seed) {
    Rng rng(seed);
    std::vector<Point> pts(n, Point(dim));
    for (auto& p : pts) for (double& x : p) x = rng.unit();
    return pts;
}

// ===========================================================================
// 演示 1：2 维 KD-树的 1-NN 与 k-NN 正确性
// ===========================================================================
static void demo_correctness() {
    std::cout << "=== 1. 2 维 KD-树：最近邻正确性（对拍暴力） ===\n";
    const int dim = 2;
    const std::size_t n = 5000;
    std::vector<Point> pts = genPoints(n, dim, 20260913);

    std::vector<Point> work = pts;                       // build 会重排，故用副本
    KdNode* root = build(work, 0, static_cast<int>(work.size()), 0, dim);

    Rng rng(999);
    // 1-NN 对拍
    for (int t = 0; t < 200; ++t) {
        Point q(dim);
        for (double& x : q) x = rng.unit() * 1.2 - 0.1;  // 查询稍微出界，测试剪枝边界
        const NnResult r = nearest(root, q);
        const double bf = bruteForceNn(pts, q);
        assert(std::fabs(r.dist2 - bf) < 1e-12);
    }
    std::cout << "  1-NN：200 次随机查询与暴力结果完全一致 ✓\n";

    // k-NN 对拍
    const std::size_t k = 7;
    for (int t = 0; t < 100; ++t) {
        Point q(dim);
        for (double& x : q) x = rng.unit();
        long long visited = 0;
        const std::vector<Cand> got = kNearest(root, q, k, visited);
        const std::vector<double> exp = bruteForceKnn(pts, q, k);
        assert(got.size() == exp.size());
        for (std::size_t i = 0; i < exp.size(); ++i)
            assert(std::fabs(got[i].dist2 - exp[i]) < 1e-12);
    }
    std::cout << "  " << k << "-NN：100 次随机查询与暴力结果完全一致 ✓\n";

    // 打印一个查询的 5 近邻
    Point q = {0.5, 0.5};
    long long visited = 0;
    const std::vector<Cand> five = kNearest(root, q, 5, visited);
    std::cout << "  查询 (0.5, 0.5) 的 5 个近邻（距离升序）:\n";
    for (const auto& c : five)
        std::cout << "    (" << std::fixed << std::setprecision(3)
                  << c.pt->at(0) << ", " << c.pt->at(1) << ")  d = "
                  << std::sqrt(c.dist2) << "\n";

    destroy(root);
    std::cout << "\n";
}

// ===========================================================================
// 演示 2：剪枝的效率（访问结点数 & 时间）
// ===========================================================================
static void demo_pruning() {
    std::cout << "=== 2. 剪枝有多强？访问结点数（2 维, N=200000） ===\n";
    const int dim = 2;
    const std::size_t n = 200000;
    std::vector<Point> pts = genPoints(n, dim, 424242);
    std::vector<Point> work = pts;
    KdNode* root = build(work, 0, static_cast<int>(work.size()), 0, dim);

    Rng rng(13);
    const int Q = 500;
    long long totalVisited = 0;
    for (int t = 0; t < Q; ++t) {
        Point q(dim);
        for (double& x : q) x = rng.unit();
        totalVisited += nearest(root, q).visited;
    }
    const double avgVisited = double(totalVisited) / Q;
    const double frac = avgVisited / static_cast<double>(n);
    std::cout << "  N = " << n << "，查询 " << Q << " 次\n";
    std::cout << "  KD-树平均访问结点数 = " << std::fixed << std::setprecision(1) << avgVisited
              << "（暴力要访问全部 " << n << " 个）\n";
    std::cout << "  平均只访问 N 的 " << std::setprecision(4) << (100.0 * frac)
              << "% —— 即少访问约 " << std::setprecision(2) << (100.0 * (1.0 - frac))
              << "% 的结点\n\n";

    destroy(root);
}

// ===========================================================================
// 演示 3：维度灾难 —— 维度一高，剪枝失效
// ===========================================================================
static void demo_curse_of_dimensionality() {
    std::cout << "=== 3. ★ 维度灾难：维度越高，KD-树越接近暴力 ===\n";
    std::cout << "  （N = 20000，每个维度查询 200 次，统计平均访问结点数占 N 的比例）\n\n";
    std::cout << "  dim  平均访问结点数   占 N 比例\n";
    std::cout << "  ---  --------------   ---------\n";

    const std::size_t n = 20000;
    double ratioLo = 0.0, ratioHi = 0.0;
    for (int dim : {2, 4, 8, 16, 32}) {
        std::vector<Point> pts = genPoints(n, dim, 1000 + static_cast<std::uint64_t>(dim));
        std::vector<Point> work = pts;
        KdNode* root = build(work, 0, static_cast<int>(work.size()), 0, dim);

        Rng rng(5000 + static_cast<std::uint64_t>(dim));
        const int Q = 200;
        long long totalVisited = 0;
        for (int t = 0; t < Q; ++t) {
            Point q(dim);
            for (double& x : q) x = rng.unit();
            totalVisited += nearest(root, q).visited;
        }
        const double avg = double(totalVisited) / Q;
        const double ratio = avg / n;
        std::cout << std::setw(4) << dim << std::setw(15) << std::fixed << std::setprecision(0)
                  << avg << std::setw(11) << std::setprecision(3) << ratio << "\n";
        if (dim == 2)  ratioLo = ratio;
        if (dim == 32) ratioHi = ratio;
        destroy(root);
    }
    assert(ratioHi > ratioLo);   // 维度升高，必须访问更多结点
    std::cout << "\n  结论：2 维时只访问极少数结点；到 32 维，访问比例逼近 1（几乎全扫）。\n"
                 "  原因：高维空间里「几乎所有点都差不多远」，切分面再也无法有效排除远侧。\n"
                 "  -> 这就是 ANN（近似最近邻）、LSH、HNSW、向量数据库存在的根本原因：\n"
                 "     与其精确搜索，不如用可扩展的结构换一点精度。\n\n";
}

int main() {
    std::cout << "======== 07 KD-树：把二叉树伸进高维空间 ========\n\n";
    demo_correctness();
    demo_pruning();
    demo_curse_of_dimensionality();
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
