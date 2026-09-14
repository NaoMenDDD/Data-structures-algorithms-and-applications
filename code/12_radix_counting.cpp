// 12_radix_counting.cpp
// ---------------------------------------------------------------------------
// 配套章节：《第 12 章 排序》
// 主题：不比较的排序 —— 计数排序、基数排序、桶排序
//
//   前面所有排序都在「比较」两个元素。而**比较排序有一个下界**：
//   任何基于比较的排序至少要做 Ω(n log n) 次比较（决策树有 n! 个叶子，
//   树高 >= log2(n!) ≈ n log2 n）。
//
//   想突破这个下界，唯一的路是**不比大小，直接利用键的「结构」**：
//     * 计数排序（counting sort）：键是小范围整数，直接数每个值出现几次。
//       时间 O(n + k)，**稳定**，需 O(k) 额外空间（k = 键的取值范围）。
//     * 基数排序（radix sort）：把键按「位」拆开，从低位到高位做若干轮
//       计数排序。时间 O(d·(n + b))（d = 位数，b = 进制）。
//     * 桶排序（bucket sort）：按键值范围分到若干桶里，各桶内部再排，
//       最后拼起来。均匀分布下**期望 O(n)**。
//
//   这三者的共同前提：**键是整数 / 可离散化，且范围可控**。
//   这是「用空间（或结构知识）换时间」的经典范例。
//
// 编译运行：
//   g++ -std=c++17 -O2 -Wall -Wextra 12_radix_counting.cpp -o 12_radix_counting && ./12_radix_counting
// 消毒器版本：
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined 12_radix_counting.cpp -o rc_san && ./rc_san
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// 带 tag 的元素，用于检测「计数 / 基数排序是否稳定」。
struct Item {
    int key;
    int tag;
};

// ===========================================================================
// 计数排序：键在 [0, maxVal]。O(n + maxVal)，稳定。
//   从后往前填 out[]（配合前缀和），是「稳定」的关键：
//   同一个键的后出现者，会被放到更靠后的位置。
// ===========================================================================
static std::vector<int> countingSort(const std::vector<int>& a, int maxVal) {
    std::vector<int> cnt(static_cast<std::size_t>(maxVal) + 1, 0);
    for (int x : a) ++cnt[static_cast<std::size_t>(x)];
    for (std::size_t v = 1; v < cnt.size(); ++v) cnt[v] += cnt[v - 1];      // 前缀和 -> 每个值的「结束位置」
    std::vector<int> out(a.size());
    for (std::size_t i = a.size(); i-- > 0; ) {                            // 逆序遍历 -> 稳定
        out[static_cast<std::size_t>(--cnt[static_cast<std::size_t>(a[i])])] = a[i];
    }
    return out;
}

// 计数排序（带 tag 版，用来实测稳定性）。
static std::vector<Item> countingSortStable(const std::vector<Item>& a, int maxVal) {
    std::vector<int> cnt(static_cast<std::size_t>(maxVal) + 1, 0);
    for (const Item& x : a) ++cnt[static_cast<std::size_t>(x.key)];
    for (std::size_t v = 1; v < cnt.size(); ++v) cnt[v] += cnt[v - 1];
    std::vector<Item> out(a.size());
    for (std::size_t i = a.size(); i-- > 0; )
        out[static_cast<std::size_t>(--cnt[static_cast<std::size_t>(a[i].key)])] = a[i];
    return out;
}

// ===========================================================================
// 基数排序（LSD，低位优先）：对非负整数，按 BASE 进制逐位做稳定计数排序。
//   某一位的排序必须是**稳定**的，否则高位的次序会被低位打乱。
// ===========================================================================
template <int BASE>
static void radixSort(std::vector<int>& a) {
    if (a.empty()) return;
    const int mx = *std::max_element(a.begin(), a.end());
    std::vector<int> buf(a.size());
    for (long long exp = 1; exp <= mx; exp *= BASE) {                      // exp = 当前处理的「位权」
        int cnt[BASE] = {0};
        for (int x : a) ++cnt[static_cast<int>((x / exp) % BASE)];
        for (int d = 1; d < BASE; ++d) cnt[d] += cnt[d - 1];
        for (std::size_t i = a.size(); i-- > 0; ) {
            const int d = static_cast<int>((a[i] / exp) % BASE);
            buf[static_cast<std::size_t>(--cnt[d])] = a[i];
        }
        a.swap(buf);
    }
}

// ===========================================================================
// 桶排序：把 [0,1) 的浮点数分到 nb 个桶，各桶内部用 std::sort，再拼接。
//   均匀分布下每桶期望 n/nb 个元素 -> 总期望 O(n + n·log(n/nb))。
//   桶数是个「旋钮」：太多 -> 分配开销压过收益；太少 -> 每桶内部排序变贵。
// ===========================================================================
static void bucketSort(std::vector<double>& a, std::size_t nb) {
    const std::size_t n = a.size();
    if (n < 2 || nb < 1) return;
    std::vector<std::vector<double>> buckets(nb);
    for (double x : a) {
        std::size_t b = static_cast<std::size_t>(x * static_cast<double>(nb));
        if (b >= nb) b = nb - 1;                                           // x 恰好 = 1.0 的边界
        buckets[b].push_back(x);
    }
    std::size_t k = 0;
    for (auto& buck : buckets) {
        std::sort(buck.begin(), buck.end());
        for (double x : buck) a[k++] = x;
    }
}

static double msSince(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    std::cout << "======== 12 不比较的排序：计数 / 基数 / 桶 ========\n\n";

    // ---------- 1. 正确性对拍 ----------
    std::cout << "=== 1. ★ 对拍：三种排序 == std::sort ===\n";
    {
        std::mt19937 rng(20260913);
        int tested = 0;
        for (int trial = 0; trial < 2000; ++trial) {
            const std::size_t n = rng() % 200;
            // (a) 计数排序：小范围键
            const int maxVal = 1 + static_cast<int>(rng() % 30);
            std::vector<int> base(n);
            for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<int>(rng() % static_cast<unsigned>(maxVal + 1));
            std::vector<int> ref = base;
            std::sort(ref.begin(), ref.end());
            assert(countingSort(base, maxVal) == ref);

            // (b) 基数排序：中等范围非负键（base 10）
            std::vector<int> r(n);
            for (std::size_t i = 0; i < n; ++i) r[i] = static_cast<int>(rng() % 100000);
            std::vector<int> rref = r;
            std::sort(rref.begin(), rref.end());
            radixSort<10>(r);
            assert(r == rref);

            // (c) 桶排序：[0,1) 浮点
            std::vector<double> d(n);
            for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<double>(rng() % 1000000) / 1000000.0;
            std::vector<double> dref = d;
            std::sort(dref.begin(), dref.end());
            bucketSort(d, std::max<std::size_t>(1, n / 16 + 1));
            assert(d == dref);
            ++tested;
        }
        std::cout << "  " << tested << " 组随机：计数 / 基数(10 进制) / 桶排序全部 == std::sort ✓\n";
    }

    // ---------- 2. ★ 稳定性：计数 / 基数排序都是稳定的 ----------
    std::cout << "\n=== 2. ★ 计数排序与基数排序的稳定性 ===\n";
    {
        std::mt19937 rng(20260914);
        const std::size_t n = 60;
        std::vector<Item> items(n);
        for (std::size_t i = 0; i < n; ++i) { items[i].key = static_cast<int>(rng() % 5); items[i].tag = static_cast<int>(i); }
        const std::vector<Item> sorted = countingSortStable(items, 4);
        bool stable = true;
        for (std::size_t i = 1; i < n; ++i)
            if (sorted[i - 1].key == sorted[i].key && sorted[i - 1].tag > sorted[i].tag) stable = false;
        std::cout << "  计数排序后，相等键是否保持原次序？ " << (stable ? "是（稳定）✓" : "否") << "\n";
        assert(stable);
        std::cout << "  -> 计数排序「逆序回填」保证稳定；基数排序的每轮也用稳定计数排序，\n";
        std::cout << "     从而整体稳定（这也是它必须「低位先排」的原因：高位排序时不能打乱低位的相对次序）。\n";
    }

    // ---------- 3. 演示：按位看基数排序 ----------
    std::cout << "\n=== 3. 基数排序「逐位」过程（base 10）===\n";
    {
        std::vector<int> a = {170, 45, 75, 90, 2, 802, 24, 66, 5};
        std::cout << "  原数组： ";
        for (int x : a) std::cout << x << " ";
        std::cout << "\n";
        radixSort<10>(a);
        std::cout << "  排序后： ";
        for (int x : a) std::cout << x << " ";
        std::cout << "\n";
        std::cout << "  过程：先按个位分桶、再按十位、再按百位（每轮都稳定），三轮后即有序。\n";
        assert(std::is_sorted(a.begin(), a.end()));
    }

    // ---------- 4. ★ 基准：vs std::sort ----------
    std::cout << "\n=== 4. ★ 基准：n = 200,000 ===\n";
    {
        const std::size_t n = 200000;
        std::mt19937 rng(20260913);

        // (a) 键范围小（0..999）-> 计数排序的主场
        {
            std::vector<int> base(n);
            for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<int>(rng() % 1000);
            std::vector<int> v = base;
            auto t0 = std::chrono::steady_clock::now(); std::sort(v.begin(), v.end());
            const double ts = msSince(t0);
            t0 = std::chrono::steady_clock::now(); volatile std::size_t sz = countingSort(base, 999).size();
            const double tc = msSince(t0);
            std::cout << "  键范围 0..999   std::sort " << ts << " ms  |  计数排序 " << tc << " ms\n";
            (void)sz;
        }
        // (b) 32 位随机整数 -> 基数排序（base 256，4 趟）vs std::sort
        {
            std::vector<int> base(n);
            for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<int>(rng() & 0x7fffffff);   // 非负
            std::vector<int> v = base;
            auto t0 = std::chrono::steady_clock::now(); std::sort(v.begin(), v.end());
            const double ts = msSince(t0);
            std::vector<int> w = base;
            t0 = std::chrono::steady_clock::now(); radixSort<256>(w);
            const double tr = msSince(t0);
            assert(w == v);
            std::cout << "  32 位整数       std::sort " << ts << " ms  |  基数排序(base256) " << tr << " ms\n";
        }
        // (c) [0,1) 均匀浮点 -> 桶排序 vs std::sort，并考察「桶数」这个旋钮
        {
            std::vector<double> base(n);
            for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<double>(rng() % 1000000) / 1000000.0;
            std::vector<double> v = base;
            auto t0 = std::chrono::steady_clock::now(); std::sort(v.begin(), v.end());
            const double ts = msSince(t0);
            std::cout << "  [0,1) 均匀浮点  std::sort " << ts << " ms\n";
            for (std::size_t nb : {n / 16, n / 4, n}) {
                std::vector<double> w = base;
                t0 = std::chrono::steady_clock::now(); bucketSort(w, nb);
                const double tb = msSince(t0);
                assert(w == v);
                std::cout << "                 桶排序(桶数=" << nb << ") " << tb << " ms\n";
            }
            std::cout << "  -> 桶数太多（= n）时，二十万个桶对象的分配 / 碎片开销压过了 O(n) 的收益；\n";
            std::cout << "     取 n/16 这种「少量大桶」反而更快——这是桶排序调参的经验点。\n";
        }
    }

    // ---------- 5. 计数排序的「软肋」：键范围一大就崩 ----------
    std::cout << "\n=== 5. ★ 计数排序的适用边界（它需要 O(k) 空间）===\n";
    {
        std::cout << "  计数排序时间 O(n + k)、空间 O(k)，k = 键的取值范围。\n";
        std::cout << "  * n=200000、键在 0..999    ：k=1000，几乎无额外空间，极快（见上）；\n";
        std::cout << "  * n=200000、键在 0..10^9   ：k=10 亿，光计数数组就要 ~4GB —— 直接崩。\n";
        std::cout << "  -> 键范围大时改**基数排序**（只按位、不需 O(k) 空间）或直接比较排序。\n";
    }

    std::cout << "\n=== 6. 复杂度与选择 ===\n"
                 "  * 比较排序下界 Ω(n log n)（决策树）；要更快必须「不比较」，利用键的整数结构；\n"
                 "  * 计数排序：O(n + k)、稳定、空间 O(k)；**仅当 k = O(n)** 时优于比较排序；\n"
                 "  * 基数排序：O(d(n+b))、稳定、空间 O(n+b)；适合定长整数 / 字符串，键范围大也能用；\n"
                 "  * 桶排序：均匀独立同分布下期望 O(n)；分布偏斜时会退化（都挤进一两个桶）；\n"
                 "  * 稳定性的意义：多级排序（先按次关键字、再按主关键字）必须用稳定排序，\n"
                 "    基数排序的正确性正是「每轮稳定」堆出来的。\n\n";
    std::cout << "全部断言通过，程序正常结束。\n";
    return 0;
}
