// 00_benchmark_chrono.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 0 章 预备：C++ 与开发环境》
// 主题：用 std::chrono 做「科学」的性能测量
//       1. ScopedTimer：RAII 计时器（离开作用域自动打印耗时）
//       2. bestOf(fn, warmup, reps)：热身 + 多次取「最小值」的微基准模板
//       3. volatile sink：防止编译器把「没有副作用」的计算整个删掉
//       4. 时钟分辨率到底是多少：实测可见最小刻度
//       5. 顺序访问 vs 跨步访问：缓存（cache）对有效带宽的影响
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 00_benchmark_chrono.cpp -o bench && ./bench
//
// 为什么必须 -O2？因为 -O0 下代码完全没被优化，测出来的是「编译器有多笨」，
// 而不是「算法有多快」。真正的性能结论必须在优化后的二进制上得出。
//
// 为什么「取最小值」而不是「取平均」？因为任何一次测量都会被操作系统调度、
// 其它进程抢 CPU、缓存冷启动等因素「污染」，这些污染只会让时间变长，
// 不会让时间变短。因此最小值最接近「这段代码在理想条件下的真实代价」。
// ---------------------------------------------------------------------------

#include <algorithm>   // std::min
#include <cassert>
#include <chrono>
#include <cmath>       // std::isfinite
#include <cstddef>
#include <iomanip>     // std::setprecision
#include <iostream>
#include <limits>      // std::numeric_limits
#include <string>
#include <type_traits> // std::is_same_v
#include <utility>     // std::move
#include <vector>

// ===========================================================================
// 0. volatile sink：对抗「as-if 规则」
// ===========================================================================
//
// C++ 标准允许编译器做任何不改变「可观察行为」的优化（as-if 规则）。
// 如果一段计算的结果从来没被使用，编译器有权把它整个删掉 —— 哪怕它跑了几秒。
// 把结果写进一个 volatile 变量，就制造了一个「可观察的副作用」，
// 编译器必须老老实实把计算做完。
//
// 注意：volatile 不是用来做线程同步的（那是 std::atomic 的活），
//       它在这里的唯一作用就是「防止死代码消除（dead code elimination）」。
static volatile double g_sink = 0.0;

// ===========================================================================
// 1. ScopedTimer：RAII 计时器
// ===========================================================================
// 与 00_raii_smart_ptr.cpp 中的思路完全一致：把「资源的生命周期」绑定到
// 对象生命周期上。这里「资源」是「一段被计时的区间」——构造时开始计，析构时打印。
// 好处：无论中间是正常 return、还是抛异常提前离开，计时代码都不会漏掉。
class ScopedTimer {
public:
    explicit ScopedTimer(std::string name)
        : name_(std::move(name)),
          start_(std::chrono::high_resolution_clock::now()) {}

    // 计时器不可拷贝：两个对象共用一个 start_ 没有意义，而且会打印两次
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    ~ScopedTimer() {
        const auto end = std::chrono::high_resolution_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(end - start_).count();
        std::cout << "[ScopedTimer] " << name_ << " 用时 "
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    }

    // 需要的话可以中途读取已经过去多久
    double elapsedSeconds() const {
        const auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ===========================================================================
// 2. bestOf：热身 + 重复取最小值的微基准模板
// ===========================================================================
// 参数说明：
//   fn     —— 被测量的可调用对象（通常是 lambda），要求无参数、返回 void
//   warmup —— 热身次数：先空跑几遍，把指令缓存、分支预测器、页表都「捂热」
//   reps   —— 正式测量次数：取其中最快的一次
// 返回：最优一次的耗时（秒）
//
// 为什么热身很重要？第一次执行时：
//   * 代码页、数据页还没进 TLB / cache，会发生大量缺页与 cache miss；
//   * 目标数组第一次被触摸，页错误（page fault）代价极高。
// 这些都是一次性开销，会严重虚高第一次的结果。
template <typename Fn>
double bestOf(Fn&& fn, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) fn();

    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        best = std::min(best, s);
    }
    return best;
}

// ===========================================================================
// 3. 实测时钟分辨率
// ===========================================================================
// 时钟的「分辨率（resolution）」是最小可分辨的时间间隔，也叫刻度（tick）。
// 如果一次操作只花 5 ns，而时钟刻度是 100 ns，那测出来要么是 0 要么是 100，
// 误差 100% —— 这时必须「跑很多遍再平均」，把单次时间放大到刻度之上。
//
// 这里用最直接的方法估计刻度：连续调用 now() 十万次，记录所有
// 「严格大于 0 的最小差值」。这个值就是我们能观察到的粒度。
static void reportClockResolution() {
    using clock = std::chrono::high_resolution_clock;
    using ns    = std::chrono::nanoseconds;

    ns minDiff = ns::max();
    auto prev = clock::now();
    for (int i = 0; i < 100000; ++i) {
        const auto cur = clock::now();
        const auto d   = std::chrono::duration_cast<ns>(cur - prev);
        if (d.count() > 0 && d < minDiff) minDiff = d;
        prev = cur;
    }

    std::cout << std::left << std::setw(46)
              << "high_resolution_clock 可见最小刻度" << " ≈ "
              << minDiff.count() << " ns\n";

    // 重要事实：high_resolution_clock 在多数实现里就是 steady_clock 或
    // system_clock 的别名。「high resolution」说的是刻度细，不代表单调。
    // 测「耗时」应该优先用 steady_clock（单调递增，不受系统改时间影响）。
    std::cout << std::left << std::setw(46) << "high_resolution_clock == steady_clock"
              << " : " << std::boolalpha
              << std::is_same_v<clock, std::chrono::steady_clock> << "\n";
    std::cout << std::left << std::setw(46) << "high_resolution_clock == system_clock"
              << " : " << std::is_same_v<clock, std::chrono::system_clock> << "\n\n";
}

// ===========================================================================
// 4. 顺序访问 vs 跨步访问：缓存局部性实验
// ===========================================================================
//
// 现代 CPU 访问 DRAM 要 ~100 ns，但访问 L1 cache 只要 ~1 ns。
// 差别来自「空间局部性（spatial locality）」：
//   * 顺序访问：读 a[0] 时，硬件会把 a[0..7] 一整条 cache line（64 字节）
//     都搬进来，之后 7 次访问全是命中；硬件预取器（prefetcher）还会
//     提前把后面几条线拉进 cache。
//   * 跨步访问：stride 超过一条 cache line 时，每次访问都落在一个全新的
//     cache line 上，搬进来的 64 字节里只用 8 字节，其余全浪费；
//     预取器也难以跟上不规律的跳跃，于是 CPU 大量时间在「等内存」。
//
// 实验设计（关键：让两种写法的「加法次数」完全相等，否则不公平）：
//   n = 2^23 = 8388608 个 double（64 MB）
//   顺序：i = 0,1,2,...,n-1              共 n 次加法，stride = 1
//   跨步：stride = 16，每个 pass 走 n/16 次，跑 16 个 pass，
//         共 (n/16)*16 = n 次加法，与顺序完全一致。
// 二者加法次数相同，唯一差别是访存模式 ⇒ 时间差就是 cache 的贡献。
static constexpr std::size_t kSize   = std::size_t(1) << 23; // 8M 个 double
static constexpr std::size_t kStride = 16;                   // 跨步 16 个 double
static constexpr std::size_t kPasses = 16;                   // 跨步重复遍数

static double sumSequential(const std::vector<double>& a) {
    double s = 0.0;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) s += a[i];  // 顺序、连续
    return s;
}

static double sumStrided(const std::vector<double>& a) {
    double s = 0.0;
    const std::size_t n = a.size();
    for (std::size_t p = 0; p < kPasses; ++p)
        for (std::size_t i = 0; i < n; i += kStride) s += a[i];
    return s;
}

// 打印「有效带宽」：把真正参与计算的数据字节数除以时间。
// 得到的是 GB/s（10^9 字节/秒），可以和内存标称带宽（比如 20~50 GB/s）对照。
static void printBandwidth(const char* label, double seconds,
                           std::size_t accesses) {
    const double bytes = static_cast<double>(accesses) * sizeof(double);
    const double gbps  = bytes / seconds / 1e9;
    std::cout << std::left << std::setw(30) << label
              << "时间 = " << std::fixed << std::setprecision(6) << seconds << " s"
              << "，有效带宽 = " << std::setprecision(2) << gbps << " GB/s\n";
}

int main() {
    std::cout << "################ 1. 时钟分辨率 ################\n";
    reportClockResolution();

    std::cout << "################ 2. ScopedTimer(RAII) ################\n";
    {
        ScopedTimer t("构造并填充 1e6 个 int");
        std::vector<int> v(1000000);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int>(i);
        g_sink = static_cast<double>(v[123]);   // 副作用，防止整段被优化掉
    }   // <-- 析构函数在这里打印耗时

    std::cout << "\n################ 3. 顺序 vs 跨步（cache 局部性）################\n";
    std::vector<double> a(kSize);
    for (std::size_t i = 0; i < kSize; ++i) {
        a[i] = static_cast<double>(i % 1000) * 0.5;
    }

    // 两种写法做完全相同的加法次数
    const std::size_t seqAccesses    = kSize;
    const std::size_t stridedAccesses = (kSize / kStride) * kPasses;
    assert(stridedAccesses == seqAccesses);   // 公平性：加法次数必须相等

    const double tSeq = bestOf([&] { g_sink = sumSequential(a); }, /*warmup=*/2, /*reps=*/5);
    const double tStr = bestOf([&] { g_sink = sumStrided(a);    }, /*warmup=*/2, /*reps=*/5);

    printBandwidth("顺序访问 (stride = 1)", tSeq, seqAccesses);
    printBandwidth("跨步访问 (stride = 16)", tStr, stridedAccesses);

    std::cout << "加速比（跨步耗时 / 顺序耗时） = "
              << std::setprecision(2) << (tStr / tSeq) << "x\n";
    std::cout << "说明：二者加法次数完全相同（" << seqAccesses
              << " 次），时间差全部来自 cache 行为。\n";
    std::cout << "在本机上顺序访问显著更快是预期结果；具体倍数与 CPU、"
                 "内存、编译器向量化有关。\n";

    // 「取最小值」的稳健性演示：同一段代码最好的与最差的一次差多少
    {
        double worst = 0.0, best = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 20; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            g_sink = sumSequential(a);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double s = std::chrono::duration<double>(t1 - t0).count();
            best  = std::min(best, s);
            worst = std::max(worst, s);
        }
        std::cout << "\n同一段顺序代码跑 20 次：最快 " << std::setprecision(6) << best
                  << " s，最慢 " << worst << " s，波动 "
                  << std::setprecision(1) << (worst / best) << "x。\n";
        std::cout << "波动说明单次测量不可信，所以 bestOf 取最小值。\n";
    }

    // 逻辑自测：两个求和结果都必须是有限数，且被 sink 吸收（未被优化掉）
    assert(std::isfinite(static_cast<double>(g_sink)));
    {
        const double s1 = sumSequential(a);
        const double s2 = sumSequential(a);
        assert(s1 == s2);                    // 同一输入结果确定
        // 跨步求和只取了一部分元素，因此一般小于总和；这里只要求它非负有限
        const double s3 = sumStrided(a);
        assert(std::isfinite(s3) && s3 > 0.0);
        std::cout << "\n顺序求和 = " << std::setprecision(6) << s1
                  << "（共 " << kSize << " 项）\n";
    }

    std::cout << "\n全部断言通过，程序正常结束。\n";
    return 0;
}
