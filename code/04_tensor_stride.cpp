// 04_tensor_stride.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 04 章 数组与矩阵：张量视角》
// 主题：用「形状 + 步长（shape + strides）」实现张量的 view / transpose /
//       permute / contiguous —— 这就是 PyTorch 张量的底层机制。
//
//   核心洞见：张量 = 一维连续存储 + 一层「索引到偏移」的地址映射。
//     偏移(index) = offset + Σ_k index[k] · stride[k]
//   只要允许 stride 任意（甚至为 0、为负），transpose / reshape / 广播 / 切片
//   全都可以**不动一个字节的数据**、只改元数据（shape 与 strides）就完成。
//
//   本文件演示：
//     1. 构造 2×3 张量，按 strides 访问；
//     2. transpose(0,1)：只交换 shape 与 strides，**零拷贝**，逻辑值即转置；
//     3. isContiguous()：判断 strides 是否等于行主序的标准 strides；
//     4. contiguous()：把非连续的 view **真正物化**成连续张量（要拷贝）；
//     5. NCHW → NHWC 的 permute：只改 strides，观察 offset 变化；
//     6. 实测：求和「连续张量」 vs 「转置后的跨步 view」—— 后者慢数倍，
//        于是你就懂了「为什么 PyTorch 的 .transpose() 之后常要 .contiguous()」。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 04_tensor_stride.cpp -o tstride && ./tstride
// ---------------------------------------------------------------------------

#include <algorithm>     // std::min, std::swap
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>        // std::numeric_limits
#include <memory>        // std::shared_ptr
#include <numeric>
#include <string>
#include <utility>
#include <vector>

static volatile double g_sink = 0.0;   // 防止死代码消除

// ---------------------------------------------------------------------------
// 迷你张量：共享底层存储的「视图」语义
// ---------------------------------------------------------------------------
class Tensor {
public:
    // 构造一个连续张量，元素填 0,1,2,...（方便观察偏移对应的真实数据）
    explicit Tensor(std::vector<long> shape) : shape_(std::move(shape)) {
        data_    = std::make_shared<std::vector<double>>(numel());
        strides_ = standardStrides(shape_);
        std::iota(data_->begin(), data_->end(), 0.0);
    }

    // ---- 元数据 ----
    long numel()              const { long n = 1; for (long d : shape_) n *= d; return n; }
    const std::vector<long>& shape()   const { return shape_; }
    const std::vector<long>& strides() const { return strides_; }
    long offset()             const { return offset_; }

    // 行主序（row-major / C 风格）的标准 strides：最后一维步长为 1
    static std::vector<long> standardStrides(const std::vector<long>& s) {
        std::vector<long> st(s.size());
        long acc = 1;
        for (long k = static_cast<long>(s.size()) - 1; k >= 0; --k) {
            st[static_cast<std::size_t>(k)] = acc;
            acc *= s[static_cast<std::size_t>(k)];
        }
        return st;
    }

    // 按逻辑索引取真实存储偏移：offset + Σ idx[k]·stride[k]
    long offsetOf(std::initializer_list<long> idx) const {
        assert(idx.size() == shape_.size());
        long o = offset_, k = 0;
        for (long i : idx) { o += i * strides_[static_cast<std::size_t>(k)]; ++k; }
        return o;
    }

    double at(std::initializer_list<long> idx) const { return (*data_)[offsetOf(idx)]; }

    // ---- 视图操作：全部零拷贝（共享 data_）----

    // 交换第 a、b 两维：只 swap shape 与 strides，数据一个字节都不动
    Tensor transpose(long a, long b) const {
        auto sh = shape_, st = strides_;
        std::swap(sh[static_cast<std::size_t>(a)], sh[static_cast<std::size_t>(b)]);
        std::swap(st[static_cast<std::size_t>(a)], st[static_cast<std::size_t>(b)]);
        return Tensor(data_, sh, st, offset_);
    }

    // 按 order 重排所有维度（如 NCHW -> NHWC 即 order = {0,2,3,1}）
    Tensor permute(const std::vector<long>& order) const {
        std::vector<long> sh, st;
        for (long k : order) { sh.push_back(shape_[k]); st.push_back(strides_[k]); }
        return Tensor(data_, sh, st, offset_);
    }

    // 是否为「连续」：strides 恰好等于行主序的标准 strides
    bool isContiguous() const { return strides_ == standardStrides(shape_); }

    // 把非连续视图真正物化成连续张量（**这里才发生数据拷贝**）
    Tensor contiguous() const {
        if (isContiguous()) {
            // 已是连续：直接返回一个共享存储的副本视图（零拷贝）
            return Tensor(data_, shape_, strides_, offset_);
        }
        Tensor out(shape_);                 // 连续的、预置 iota 的骨架
        const long n = numel();
        std::vector<long> idx(shape_.size(), 0);
        for (long lin = 0; lin < n; ++lin) {
            // 把行主序线性下标拆成多维逻辑索引
            long rem = lin;
            for (long k = static_cast<long>(shape_.size()) - 1; k >= 0; --k) {
                idx[static_cast<std::size_t>(k)] = rem % shape_[static_cast<std::size_t>(k)];
                rem /= shape_[static_cast<std::size_t>(k)];
            }
            long src = offset_;
            for (std::size_t k = 0; k < shape_.size(); ++k) src += idx[k] * strides_[k];
            (*out.data_)[static_cast<std::size_t>(lin)] = (*data_)[static_cast<std::size_t>(src)];
        }
        return out;
    }

    void describe(const std::string& label) const {
        std::cout << std::left << std::setw(12) << label << " shape=[";
        for (std::size_t k = 0; k < shape_.size(); ++k) {
            std::cout << shape_[k] << (k + 1 < shape_.size() ? "," : "");
        }
        std::cout << "]  strides=[";
        for (std::size_t k = 0; k < strides_.size(); ++k) {
            std::cout << strides_[k] << (k + 1 < strides_.size() ? "," : "");
        }
        std::cout << "]  contiguous=" << std::boolalpha << isContiguous() << "\n";
    }

private:
    // 内部构造：用于生成共享存储的视图
    Tensor(std::shared_ptr<std::vector<double>> d,
           std::vector<long> shape, std::vector<long> strides, long offset)
        : data_(std::move(d)), shape_(std::move(shape)),
          strides_(std::move(strides)), offset_(offset) {}

    std::shared_ptr<std::vector<double>> data_;
    std::vector<long>                    shape_;
    std::vector<long>                    strides_;
    long                                 offset_ = 0;
};

// ---------------------------------------------------------------------------
// 1~4. transpose / contiguous
// ---------------------------------------------------------------------------
static void demo_transpose() {
    std::cout << "=== 1. 构造 2×3 连续张量（数据 0..5 按行主序铺开）===\n";
    Tensor A({2, 3});
    A.describe("A");
    std::cout << "  底层存储: 0 1 2 3 4 5\n";
    std::cout << "  逻辑视图:\n";
    for (long i = 0; i < 2; ++i) {
        std::cout << "    ";
        for (long j = 0; j < 3; ++j) std::cout << A.at({i, j}) << " ";
        std::cout << "\n";
    }

    std::cout << "\n=== 2. transpose(0,1)：零拷贝，只换 shape 与 strides ===\n";
    Tensor T = A.transpose(0, 1);
    T.describe("A^T");
    std::cout << "  逻辑视图（转置后的值）:\n";
    for (long i = 0; i < 3; ++i) {
        std::cout << "    ";
        for (long j = 0; j < 2; ++j) std::cout << T.at({i, j}) << " ";
        std::cout << "\n";
    }
    // 关键证明：A 与 T 共享同一块底层存储（零拷贝）
    assert(A.at({0, 1}) == T.at({1, 0}));     // A[0][1] == A^T[1][0]
    assert(A.at({1, 2}) == T.at({2, 1}));
    std::cout << "  验证 A[0][1]==" << A.at({0, 1}) << " == A^T[1][0]=" << T.at({1, 0})
              << "  ✓（转置未拷贝数据）\n";

    std::cout << "\n=== 3. isContiguous：转置后**不再连续** ===\n";
    assert(A.isContiguous());
    assert(!T.isContiguous());
    std::cout << "  A 连续 = true，A^T 连续 = false ✓\n";
    std::cout << "  → 因为 A^T 的行 stride=1，而 C 风格要求**最后一维** stride=1\n";

    std::cout << "\n=== 4. contiguous()：物化成连续张量（此时才拷贝）===\n";
    Tensor C = T.contiguous();
    C.describe("T.contig");
    std::cout << "  物化后的逻辑视图与 T 相同，但 strides 变成标准行主序:\n";
    for (long i = 0; i < 3; ++i) {
        std::cout << "    ";
        for (long j = 0; j < 2; ++j) std::cout << C.at({i, j}) << " ";
        std::cout << "\n";
    }
    assert(C.isContiguous());
    assert(C.at({1, 0}) == T.at({1, 0}));
    std::cout << "  contiguous 后连续 = true，且逻辑值与 A^T 逐一相等 ✓\n\n";
}

// ---------------------------------------------------------------------------
// 5. NCHW -> NHWC 的 permute
// ---------------------------------------------------------------------------
static void demo_nchw_nhwc() {
    std::cout << "=== 5. NCHW → NHWC：只改 strides 的「布局切换」 ===\n";
    // N=2, C=3, H=4, W=5
    Tensor x({2, 3, 4, 5});
    x.describe("NCHW");
    Tensor y = x.permute({0, 2, 3, 1});      // NHWC = 取第 0,2,3,1 维
    y.describe("NHWC");
    std::cout << "  同一个 (n,c,h,w) 元素在两套布局下的底层偏移:\n";
    // 逻辑索引 (n=1,c=2,h=3,w=4)
    const long offNCHW = x.offsetOf({1, 2, 3, 4});
    const long offNHWC = y.offsetOf({1, 3, 4, 2});   // NHWC 下同元素索引是 (n,h,w,c)
    std::cout << "    x[1][2][3][4] 偏移 = " << offNCHW
              << "（NCHW）, y[1][3][4][2] 偏移 = " << offNHWC
              << "（NHWC）\n";
    assert(offNCHW == offNHWC);               // 同一物理元素，偏移必然相同
    std::cout << "  → 同一个元素的物理偏移一致，说明 permute 没搬数据，只换了「索引进偏移」的规则 ✓\n";
    std::cout << "  → 这就是 cuDNN 区分 NCHW / NHWC 的本质：\n"
                 "    内核读同一个数，但因为 stride 不同，访存模式（能否合并）完全不同。\n\n";
}

// ---------------------------------------------------------------------------
// 6. 实测：连续 vs 跨步访问
// ---------------------------------------------------------------------------
template <typename Fn>
static double bestOf(Fn&& fn, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) fn();
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

static void demo_stride_perf() {
    std::cout << "=== 6. 求和：连续张量 vs 转置后的跨步 view ===\n";
    const long N = 2048;
    Tensor a({N, N});                        // 2048×2048，连续
    Tensor at = a.transpose(0, 1);           // 非连续 view

    // 两种求和都按「行优先」遍历自己的逻辑下标，加法次数完全相同
    auto sumContig = [&] {
        double s = 0;
        for (long i = 0; i < N; ++i)
            for (long j = 0; j < N; ++j) s += a.at({i, j});
        g_sink = s;
    };
    auto sumStrided = [&] {
        double s = 0;
        for (long i = 0; i < N; ++i)
            for (long j = 0; j < N; ++j) s += at.at({i, j});   // 跨步访问
        g_sink = s;
    };

    const double tc = bestOf(sumContig,  1, 5);
    const double ts = bestOf(sumStrided, 1, 5);

    std::cout << std::left << std::setw(28) << "连续张量 a[i][j]"
              << std::fixed << std::setprecision(3) << (tc * 1e3) << " ms\n";
    std::cout << std::left << std::setw(28) << "跨步 view a^T[i][j]"
              << std::fixed << std::setprecision(3) << (ts * 1e3) << " ms\n";
    std::cout << "跨步 / 连续 耗时比 = " << std::setprecision(2) << (ts / tc) << "x\n";
    std::cout << "（两者加法次数相同，差异全来自 cache：跨步访问每次落在一个"
                 "新的 cache line 上）\n";
    std::cout << "→ 这正是 .transpose() 之后常要 .contiguous() 的原因：\n"
                 "  多花一次拷贝，换回后续算子的顺序访存。\n\n";
}

int main() {
    std::cout << "======== 04 张量的 shape / stride / view ========\n\n";
    demo_transpose();
    demo_nchw_nhwc();
    demo_stride_perf();
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
